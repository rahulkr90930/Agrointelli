/**
 * AgroIntelli — Plant Health Community & AI Scanner
 * app.js: Core Application Logic & State Management
 */

(function () {
  'use strict';

  // ══════════════════════════════════════════════════════════════
  // APPLICATION STATE
  // ══════════════════════════════════════════════════════════════
  const state = {
    user: null,                   // Logged in user or null (guest mode)
    activeTab: 'home',            // 'home' | 'scan' | 'plants'
    scanMode: 'single',           // 'single' | 'timeline'
    selectedPlantId: null,        // null by default in guest mode
    selectedDiseaseId: null,      // For highlighting disease cards
    plantsCatalog: [],            // List of 14 supported crops
    currentPlantDiseases: [],     // Diseases for state.selectedPlantId
    communityPosts: [],           // Discussion feed
    activeCommunityFilter: 'all', // 'all' | plant_id
    userRecords: [],              // Journal records from /api/records
    chatHistory: [],              // Chat history for AgroBot
    activeScanResult: null,       // Result of latest single scan
    activeTimelineResult: null,   // Result of latest batch scan
    singleScanFile: null,         // File object for single scan
    singleScanFileUrl: null,      // ObjectURL for single scan original image
    timelineFiles: {              // Files for progression analysis
      day1: null,
      day5: null,
      day10: null
    },
    timelineDays: {               // User configurable days
      day1: 1,
      day5: 5,
      day10: 10
    },
    chatLanguage: 'English',      // 'English' | 'Hindi' | 'Bengali'
    chatTtsEnabled: true,         // Text-to-speech voice output toggle
    activeScanForJournal: null,   // Cached scan to prevent stale form collisions
    activeCheckinRecord: null
  };

  // ══════════════════════════════════════════════════════════════
  // DOM ELEMENT REFERENCES
  // ══════════════════════════════════════════════════════════════
  const dom = {
    // Top Header & Nav
    aiModelBadge: document.getElementById('ai-model-badge'),
    userTopPill: document.getElementById('user-top-pill'),
    userTopName: document.getElementById('user-top-name'),
    navBtns: document.querySelectorAll('.nav-btn'),
    bnavBtns: document.querySelectorAll('.bnav-btn'),

    // Views
    viewHome: document.getElementById('view-home'),
    viewScan: document.getElementById('view-scan'),
    viewPlants: document.getElementById('view-plants'),

    // Home Page Elements
    plantsShelf: document.getElementById('plants-shelf'),
    plantsGrid: document.getElementById('plants-grid'),
    plantEcosystemCard: document.getElementById('plant-ecosystem-card'),
    btnBackToAllPlants: document.getElementById('btn-back-to-all-plants'),
    btnScanSelectedCrop: document.getElementById('btn-scan-selected-crop'),
    ecosystemCropName: document.getElementById('ecosystem-crop-name'),
    ecosystemBotanical: document.getElementById('ecosystem-botanical'),
    ecosystemDesc: document.getElementById('ecosystem-desc'),
    diseaseCardsContainer: document.getElementById('disease-cards-container'),
    homeCommunitySection: document.getElementById('home-community-section'),
    communityFeed: document.getElementById('community-feed'),
    communityFilterPills: document.getElementById('community-filter-pills'),

    // FABs
    fabContainer: document.getElementById('fab-container'),
    fabCreatePost: document.getElementById('fab-create-post'),
    chatFab: document.getElementById('chat-fab'),

    // Scan Page Elements
    scanTabSingle: document.getElementById('scan-tab-single'),
    scanTabTimeline: document.getElementById('scan-tab-timeline'),
    panelSingleScan: document.getElementById('panel-single-scan'),
    panelTimelineScan: document.getElementById('panel-timeline-scan'),
    scanCropSelect: document.getElementById('scan-crop-select'),
    timelineCropSelect: document.getElementById('timeline-crop-select'),
    singleDropZone: document.getElementById('single-drop-zone'),
    singleFileInput: document.getElementById('single-file-input'),
    singleCameraInput: document.getElementById('single-camera-input'),
    singlePreviewBox: document.getElementById('single-preview-box'),
    singlePreviewImg: document.getElementById('single-preview-img'),
    singleRemoveImgBtn: document.getElementById('single-remove-img-btn'),
    btnRunSingleScan: document.getElementById('btn-run-single-scan'),
    singleScanResultsCard: document.getElementById('single-scan-results-card'),

    // Timeline Slots
    slotDay1Input: document.getElementById('slot-day1-input'),
    slotDay5Input: document.getElementById('slot-day5-input'),
    slotDay10Input: document.getElementById('slot-day10-input'),
    slotDay1Preview: document.getElementById('slot-day1-preview'),
    slotDay5Preview: document.getElementById('slot-day5-preview'),
    slotDay10Preview: document.getElementById('slot-day10-preview'),
    btnRunTimelineScan: document.getElementById('btn-run-timeline-scan'),
    timelineResultsCard: document.getElementById('timeline-results-card'),

    // My Plants (Journal) Elements
    plantsJournalGrid: document.getElementById('plants-journal-grid'),
    btnNewPlantEntry: document.getElementById('btn-new-plant-entry'),

    // Modals
    modalUserAccount: document.getElementById('modal-user-account'),
    modalCreatePost: document.getElementById('modal-create-post'),
    modalSaveJournal: document.getElementById('modal-save-journal'),
    modalFollowupScan: document.getElementById('modal-followup-scan'),
    modalLightbox: document.getElementById('modal-lightbox'),
    lightboxImg: document.getElementById('lightbox-img'),

    // Chat Drawer & Voice Controls
    chatDrawer: document.getElementById('chat-drawer'),
    chatCloseBtn: document.getElementById('chat-close-btn'),
    chatMessages: document.getElementById('chat-messages'),
    chatInput: document.getElementById('chat-input'),
    chatSendBtn: document.getElementById('chat-send-btn'),
    chatLangSelect: document.getElementById('chat-lang-select'),
    chatTtsToggle: document.getElementById('chat-tts-toggle'),
    chatVoiceBtn: document.getElementById('chat-voice-btn'),

    // Toast
    toastPill: document.getElementById('toast-pill')
  };

  // ══════════════════════════════════════════════════════════════
  // UTILITY HELPERS
  // ══════════════════════════════════════════════════════════════
  function showToast(message, duration = 3000) {
    if (!dom.toastPill) return;
    dom.toastPill.textContent = message;
    dom.toastPill.classList.add('show');
    clearTimeout(dom.toastPill._timer);
    dom.toastPill._timer = setTimeout(() => {
      dom.toastPill.classList.remove('show');
    }, duration);
  }

  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  function formatRelativeDate(isoStr) {
    if (!isoStr) return 'Recently';
    try {
      const d = new Date(isoStr);
      const now = new Date();
      const diffSec = Math.floor((now - d) / 1000);
      if (diffSec < 60) return 'Just now';
      if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
      if (diffSec < 86400) return `${Math.floor(diffSec / 3600)}h ago`;
      if (diffSec < 604800) return `${Math.floor(diffSec / 86400)}d ago`;
      return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
    } catch {
      return 'Recently';
    }
  }

  // ══════════════════════════════════════════════════════════════
  // USER & AUTHENTICATION
  // ══════════════════════════════════════════════════════════════
  function initUser() {
    try {
      const saved = localStorage.getItem('agro_user');
      if (saved) {
        state.user = JSON.parse(saved);
      }
    } catch (e) {
      console.warn('Failed reading agro_user from localStorage', e);
    }
    updateUserUI();
  }

  function updateUserUI() {
    if (!dom.userTopName) return;
    if (state.user && state.user.username) {
      dom.userTopName.textContent = state.user.username;
    } else {
      dom.userTopName.textContent = 'Guest Grower';
    }
  }

  function logoutUser() {
    state.user = null;
    localStorage.removeItem('agro_user');
    updateUserUI();
    showToast('Signed out. Switched to Guest Mode.');
    closeModal(dom.modalUserAccount);
    renderUserAccountModal();
  }

  // ══════════════════════════════════════════════════════════════
  // PRIMARY NAVIGATION (HOME, SCAN, MY PLANTS)
  // ══════════════════════════════════════════════════════════════
  function switchTab(targetTab) {
    state.activeTab = targetTab;

    // Toggle View visibility
    if (dom.viewHome) dom.viewHome.style.display = (targetTab === 'home') ? 'block' : 'none';
    if (dom.viewScan) dom.viewScan.style.display = (targetTab === 'scan') ? 'block' : 'none';
    if (dom.viewPlants) dom.viewPlants.style.display = (targetTab === 'plants') ? 'block' : 'none';

    // Update Nav Button Active States
    dom.navBtns.forEach(btn => {
      const tab = btn.getAttribute('data-tab');
      btn.classList.toggle('active', tab === targetTab);
    });
    dom.bnavBtns.forEach(btn => {
      const tab = btn.getAttribute('data-tab');
      btn.classList.toggle('active', tab === targetTab);
    });

    // Control AI Model Badge: Only visible when Scan tab is active
    if (dom.aiModelBadge) {
      dom.aiModelBadge.style.display = (targetTab === 'scan') ? 'inline-flex' : 'none';
    }

    // Control FAB "+ Create Post" visibility: only visible on Home view
    if (dom.fabCreatePost) {
      dom.fabCreatePost.style.display = (targetTab === 'home') ? 'inline-flex' : 'none';
    }

    // Tab-specific initializations
    if (targetTab === 'home') {
      if (!state.selectedPlantId) {
        deselectPlant();
      }
    } else if (targetTab === 'scan') {
      initScanView();
    } else if (targetTab === 'plants') {
      loadUserJournal();
    }

    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // ══════════════════════════════════════════════════════════════
  // CROPS CATALOG & SUPPORTED PLANTS
  // ══════════════════════════════════════════════════════════════
  async function fetchPlantsCatalog() {
    try {
      const res = await fetch('/api/plants');
      const data = await res.json();
      if (data && data.plants) {
        state.plantsCatalog = data.plants;
        renderSupportedPlantsGrid();
        populateCropDropdowns();
        renderCommunityFilterPills();
      }
    } catch (err) {
      console.error('Failed to fetch crops catalog:', err);
    }
  }

  function renderSupportedPlantsGrid() {
    if (!dom.plantsGrid) return;
    dom.plantsGrid.innerHTML = '';

    state.plantsCatalog.forEach(crop => {
      const card = document.createElement('div');
      card.className = 'plant-card';
      card.setAttribute('data-plant-id', crop.id);

      const cropImgSrc = crop.image || `/images/crops/${crop.id}.jpg`;
      const fallbackIcon = crop.icon || '🌱';
      const condCount = crop.disease_count || crop.diseases_count || 3;

      card.innerHTML = `
        <div class="plant-card-media">
          <img src="${cropImgSrc}" alt="${escapeHtml(crop.name)}" class="plant-card-img" onerror="this.onerror=null;this.parentElement.innerHTML='<span class=\\'plant-card-icon-fallback\\'>${fallbackIcon}</span>';" />
          <span class="plant-card-emoji-pill">${fallbackIcon}</span>
        </div>
        <div class="plant-card-content">
          <div>
            <div class="plant-card-name">${escapeHtml(crop.name)}</div>
            <div class="plant-card-botanical">${escapeHtml(crop.botanical || '')}</div>
          </div>
          <div class="plant-card-badge">${condCount} Conditions &bull; Tap to explore</div>
        </div>
      `;

      card.addEventListener('click', () => {
        selectPlant(crop.id);
      });
      dom.plantsGrid.appendChild(card);
    });
  }

  function populateCropDropdowns() {
    const selectors = [dom.scanCropSelect, dom.timelineCropSelect, document.getElementById('post-crop-select')];
    selectors.forEach(sel => {
      if (!sel) return;
      const currentVal = sel.value;
      const isPost = (sel.id === 'post-crop-select');
      sel.innerHTML = isPost
        ? `<option value="">Select Crop for Discussion</option>`
        : `<option value="">Select Crop (Optional - Auto-Detect)</option>`;

      state.plantsCatalog.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c.id;
        opt.textContent = `${c.name} (${c.botanical || ''})`;
        sel.appendChild(opt);
      });

      if (currentVal) sel.value = currentVal;
    });
  }

  // ══════════════════════════════════════════════════════════════
  // HOME PAGE: SELECT PLANT & ECOSYSTEM VIEW
  // ══════════════════════════════════════════════════════════════
  async function selectPlant(plantId) {
    if (!plantId) return;
    state.selectedPlantId = plantId;

    const cropMeta = state.plantsCatalog.find(c => c.id === plantId) || {
      id: plantId,
      name: plantId.charAt(0).toUpperCase() + plantId.slice(1),
      botanical: 'Cultivated Specimen',
      description: 'Crop ecosystem details and disease library.'
    };

    // 1. Hide Supported Plants Section
    if (dom.plantsShelf) dom.plantsShelf.style.display = 'none';

    // 2. Show Plant Ecosystem Card
    if (dom.plantEcosystemCard) dom.plantEcosystemCard.style.display = 'block';

    // Update Header Meta & Scan CTA
    if (dom.ecosystemCropName) dom.ecosystemCropName.textContent = cropMeta.name;
    if (dom.ecosystemBotanical) dom.ecosystemBotanical.textContent = cropMeta.botanical || '';
    if (dom.ecosystemDesc) dom.ecosystemDesc.textContent = cropMeta.description || '';
    if (dom.btnScanSelectedCrop) dom.btnScanSelectedCrop.innerHTML = `🔬 Scan ${escapeHtml(cropMeta.name)} Leaf`;

    // Synchronize scan dropdowns with chosen plant
    if (dom.scanCropSelect) dom.scanCropSelect.value = plantId;
    if (dom.timelineCropSelect) dom.timelineCropSelect.value = plantId;

    // Save preference to backend if logged in
    if (state.user && (state.user.user_id || state.user.id || state.user.username)) {
      try {
        fetch('/api/users/preference', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ user_id: getCurrentUserId(), last_selected_plant: plantId })
        });
      } catch (e) {}
    }

    // 3. Load & Render Disease Cards for this crop
    await fetchPlantDiseases(plantId);

    // 4. Filter Community Discussions to this crop
    fetchCommunityPosts(plantId);

    // Smooth scroll to top of ecosystem view
    if (dom.plantEcosystemCard) {
      dom.plantEcosystemCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  function deselectPlant() {
    state.selectedPlantId = null;

    // 1. Hide Plant Ecosystem Card
    if (dom.plantEcosystemCard) dom.plantEcosystemCard.style.display = 'none';

    // 2. Reveal Supported Plants Shelf
    if (dom.plantsShelf) dom.plantsShelf.style.display = 'block';

    // 3. Reset Scan Crop Selectors to Placeholder
    if (dom.scanCropSelect) dom.scanCropSelect.value = '';
    if (dom.timelineCropSelect) dom.timelineCropSelect.value = '';

    // 4. Restore All Community Discussions
    fetchCommunityPosts(null);
  }

  async function fetchPlantDiseases(plantId) {
    if (!dom.diseaseCardsContainer) return;
    dom.diseaseCardsContainer.innerHTML = `
      <div style="grid-column: 1 / -1; padding: 2.5rem; text-align: center; color: var(--text-3);">
        <div class="loader-spinner" style="margin: 0 auto 0.75rem;"></div>
        Loading verified disease profiles for ${escapeHtml(plantId)}...
      </div>
    `;

    try {
      const res = await fetch(`/api/plants/${encodeURIComponent(plantId)}/diseases`);
      const data = await res.json();
      if (data && data.diseases) {
        state.currentPlantDiseases = data.diseases;
        renderDiseaseCards(data.diseases);
      } else {
        dom.diseaseCardsContainer.innerHTML = `<div style="grid-column: 1 / -1; padding: 1.5rem; color: var(--text-3);">No specific disease records found for this crop.</div>`;
      }
    } catch (err) {
      console.error('Error fetching plant diseases:', err);
      dom.diseaseCardsContainer.innerHTML = `<div style="grid-column: 1 / -1; padding: 1.5rem; color: var(--red-600);">Failed to load disease data.</div>`;
    }
  }

  function renderDiseaseCards(diseases) {
    if (!dom.diseaseCardsContainer) return;
    dom.diseaseCardsContainer.innerHTML = '';

    diseases.forEach(d => {
      const card = document.createElement('div');
      card.className = `disease-card ${d.is_healthy ? 'is-healthy' : ''}`;
      card.id = `dc-${d.id}`;

      let badgeClass = 'fungal';
      let badgeLabel = 'Fungal Lesion';
      if (d.is_healthy) {
        badgeClass = 'healthy';
        badgeLabel = 'Healthy Foliage';
      } else if (d.pathogen_type === 'bacterial' || d.id.includes('bacterial')) {
        badgeClass = 'bacterial';
        badgeLabel = 'Bacterial Spot';
      } else if (d.pathogen_type === 'viral' || d.id.includes('virus')) {
        badgeClass = 'viral';
        badgeLabel = 'Viral Disease';
      }

      // Generate reference image thumbnails (prioritize lightweight bundled /images/diseases/)
      const refCount = (d.reference_images && d.reference_images.length) || 3;
      let refImgsHtml = '';
      for (let i = 0; i < Math.min(refCount, 3); i++) {
        const bundledUrl = `/images/diseases/${d.id}/${i}.jpg`;
        const apiFallbackUrl = `/api/reference-images/${d.id}/${i}`;
        refImgsHtml += `
          <div class="dc-ref-item" data-full-url="${bundledUrl}" title="View verified reference photo">
            <img src="${bundledUrl}" alt="${escapeHtml(d.name)} sample" loading="lazy" onerror="this.src='${apiFallbackUrl}';" />
          </div>
        `;
      }

      card.innerHTML = `
        <div>
          <div class="dc-top">
            <h4 class="dc-name">${escapeHtml(d.name)}</h4>
            <span class="dc-badge ${badgeClass}">${badgeLabel}</span>
          </div>
          ${d.pathogen ? `<div class="dc-pathogen">Pathogen: ${escapeHtml(d.pathogen)}</div>` : ''}

          <div class="dc-ref-strip">
            <div class="dc-ref-label">Verified Dataset References</div>
            <div class="dc-ref-scroll">${refImgsHtml}</div>
          </div>

          <div class="dc-desc">${escapeHtml(d.symptoms || d.description || 'Characteristic leaf symptoms.')}</div>
        </div>

        <div class="dc-actions">
          <button class="btn btn-secondary btn-sm btn-dc-filter" data-disease-id="${d.id}" type="button">
            🔍 Discussions
          </button>
          <button class="btn btn-outline-emerald btn-sm btn-dc-ask" data-disease-name="${escapeHtml(d.name)}" type="button">
            💬 Ask AgroBot
          </button>
        </div>
      `;

      // Attach Lightbox click on reference images
      card.querySelectorAll('.dc-ref-item').forEach(thumb => {
        thumb.addEventListener('click', () => {
          const imgEl = thumb.querySelector('img');
          openLightbox(imgEl ? imgEl.src : thumb.getAttribute('data-full-url'));
        });
      });

      // Filter community discussions to this specific disease
      card.querySelector('.btn-dc-filter').addEventListener('click', () => {
        fetchCommunityPosts(state.selectedPlantId, d.id);
        const commSec = document.getElementById('home-community-section');
        if (commSec) commSec.scrollIntoView({ behavior: 'smooth' });
      });

      // Query AgroBot about this disease
      card.querySelector('.btn-dc-ask').addEventListener('click', () => {
        openAgroBot(`What are the recommended prevention and organic treatments for ${d.name}?`);
      });

      dom.diseaseCardsContainer.appendChild(card);
    });
  }

  // ══════════════════════════════════════════════════════════════
  // PLANT HEALTH COMMUNITY DISCUSSIONS
  // ══════════════════════════════════════════════════════════════
  function renderCommunityFilterPills() {
    if (!dom.communityFilterPills) return;
    dom.communityFilterPills.innerHTML = '';

    const allPill = document.createElement('button');
    allPill.className = `filter-pill ${state.activeCommunityFilter === 'all' ? 'active' : ''}`;
    allPill.textContent = 'All Discussions';
    allPill.addEventListener('click', () => {
      state.activeCommunityFilter = 'all';
      renderCommunityFilterPills();
      fetchCommunityPosts(null);
    });
    dom.communityFilterPills.appendChild(allPill);

    state.plantsCatalog.forEach(crop => {
      const pill = document.createElement('button');
      pill.className = `filter-pill ${state.activeCommunityFilter === crop.id ? 'active' : ''}`;
      pill.textContent = `${crop.icon || ''} ${crop.name}`;
      pill.addEventListener('click', () => {
        state.activeCommunityFilter = crop.id;
        renderCommunityFilterPills();
        fetchCommunityPosts(crop.id);
      });
      dom.communityFilterPills.appendChild(pill);
    });
  }

  async function fetchCommunityPosts(plantFilter = null, diseaseFilter = null) {
    if (!dom.communityFeed) return;
    dom.communityFeed.innerHTML = `
      <div style="padding: 2.5rem; text-align: center; color: var(--text-3);">
        <div class="loader-spinner" style="margin: 0 auto 0.75rem;"></div>
        Loading grower discussions...
      </div>
    `;

    try {
      let url = '/api/community/posts';
      const params = new URLSearchParams();
      if (plantFilter) params.append('plant', plantFilter);
      if (diseaseFilter) params.append('disease', diseaseFilter);
      if (params.toString()) url += `?${params.toString()}`;

      const res = await fetch(url);
      const data = await res.json();
      if (data && data.posts) {
        state.communityPosts = data.posts;
        renderCommunityFeed(data.posts);
      } else {
        dom.communityFeed.innerHTML = `<div style="padding: 2rem; text-align: center; color: var(--text-3);">No discussions found for this crop yet. Be the first to start a conversation!</div>`;
      }
    } catch (err) {
      console.error('Failed fetching community posts:', err);
      dom.communityFeed.innerHTML = `<div style="padding: 2rem; text-align: center; color: var(--red-600);">Failed to load community discussions.</div>`;
    }
  }

  function getActiveVoterId() {
    if (state.user && (state.user.id || state.user.username)) {
      return getCurrentUserId();
    }
    let guestId = localStorage.getItem('agro_guest_id');
    if (!guestId) {
      guestId = 'guest_' + Math.random().toString(36).substring(2, 10);
      localStorage.setItem('agro_guest_id', guestId);
    }
    return guestId;
  }

  function getCurrentUserId() {
    if (!state.user) return 'guest_user';
    return String(state.user.user_id || state.user.id || state.user.username || 'guest_user');
  }

  function renderCommunityFeed(posts) {
    if (!dom.communityFeed) return;
    if (!posts || posts.length === 0) {
      dom.communityFeed.innerHTML = `
        <div style="padding: 3rem 1.5rem; text-align: center; background: #ffffff; border-radius: var(--radius-xl); border: 1.5px dashed var(--border);">
          <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">🌾</div>
          <h4 style="color: var(--emerald-900); margin-bottom: 0.35rem;">No Community Posts Yet</h4>
          <p style="color: var(--text-3); font-size: 0.85rem; max-width: 400px; margin: 0 auto 1.25rem;">
            Ask a question or share field observations about your plants with growers worldwide.
          </p>
          <button class="btn btn-primary" onclick="window.agroOpenCreatePost()">
            + Start a Plant Health Discussion
          </button>
        </div>
      `;
      return;
    }

    dom.communityFeed.innerHTML = '';
    const voterId = getActiveVoterId();

    posts.forEach(post => {
      const pid = post.post_id || post.id || post._id;
      post.id = pid;
      post.post_id = pid;

      const card = document.createElement('article');
      card.className = 'community-post-card';
      card.id = `post-${pid}`;

      const userVote = post.voted_users ? post.voted_users[voterId] : null;
      const isUpvoted = (userVote === 'up') || (post.upvoted_by && post.upvoted_by.includes(voterId));
      const isDownvoted = (userVote === 'down');
      const commentsCount = (post.comments && post.comments.length) || post.comment_count || 0;

      card.innerHTML = `
        <div class="post-header">
          <div style="display:flex;align-items:center;gap:0.65rem;">
            <div class="post-avatar">${post.author ? post.author.charAt(0).toUpperCase() : 'G'}</div>
            <div class="post-meta">
              <div class="post-author-name">
                ${escapeHtml(post.author || 'Grower')}
                <span class="post-badge">${escapeHtml(post.author_badge || 'Grower')}</span>
              </div>
              <div class="post-time">${formatRelativeDate(post.created_at)}</div>
            </div>
          </div>
          <div class="post-crop-tag">${escapeHtml(post.plant_name || 'Crop')}</div>
        </div>

        <div class="post-content">${escapeHtml(post.content)}</div>

        ${post.disease_name ? `
          <div style="margin: 0.5rem 0 0.75rem;">
            <span style="font-size: 0.75rem; background: var(--emerald-50); color: var(--emerald-800); border: 1px solid var(--emerald-200); padding: 0.2rem 0.6rem; border-radius: var(--radius-full); font-weight: 600;">
              Topic: ${escapeHtml(post.disease_name)}
            </span>
          </div>
        ` : ''}

        <div class="post-footer">
          <!-- Upvote and Downvote Group (Allowed for both Guests and Users) -->
          <div class="post-vote-group">
            <button class="post-action-btn btn-upvote ${isUpvoted ? 'active' : ''}" data-post-id="${pid}" type="button" title="Upvote / Helpful">
              ▲ <span class="upvote-count">${post.upvotes || 0}</span>
            </button>
            <button class="post-action-btn btn-downvote ${isDownvoted ? 'active' : ''}" data-post-id="${pid}" type="button" title="Downvote">
              ▼ <span class="downvote-count">${post.downvotes || 0}</span>
            </button>
          </div>
          <button class="post-action-btn btn-toggle-comments" data-post-id="${pid}" type="button">
            💬 <span class="comment-count">${commentsCount}</span> Replies
          </button>
        </div>

        <div class="post-comments-drawer" id="comments-drawer-${pid}" style="display: none;">
          <div class="comments-list" id="comments-list-${pid}"></div>
          <form class="comment-input-row" data-post-id="${pid}">
            <input type="text" class="form-input comment-input" placeholder="Share your agronomic advice or reply..." required />
            <button type="submit" class="btn btn-primary" style="padding: 0.4rem 0.9rem; font-size: 0.8rem;">Reply</button>
          </form>
        </div>
      `;

      // Upvote handler (Guests allowed)
      const btnUp = card.querySelector('.btn-upvote');
      if (btnUp) {
        btnUp.addEventListener('click', async () => {
          await handleVotePost(pid, 'up', card);
        });
      }

      // Downvote handler (Guests allowed)
      const btnDown = card.querySelector('.btn-downvote');
      if (btnDown) {
        btnDown.addEventListener('click', async () => {
          await handleVotePost(pid, 'down', card);
        });
      }

      // Toggle comments handler
      const btnComments = card.querySelector('.btn-toggle-comments');
      const drawer = card.querySelector(`#comments-drawer-${pid}`);
      btnComments.addEventListener('click', () => {
        const isHidden = (drawer.style.display === 'none');
        drawer.style.display = isHidden ? 'block' : 'none';
        if (isHidden) {
          loadPostComments(pid, post.comments);
        }
      });

      // Comment submission form: intercept guests and redirect to sign-in modal
      const commentForm = card.querySelector('.comment-input-row');
      const input = commentForm.querySelector('.comment-input');

      input.addEventListener('focus', () => {
        if (!state.user || !state.user.username) {
          showToast('Please sign in or create an account to reply.');
          renderUserAccountModal();
          openModal(dom.modalUserAccount);
        }
      });

      commentForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        if (!state.user || !state.user.username) {
          showToast('Please sign in or create an account to reply.');
          renderUserAccountModal();
          openModal(dom.modalUserAccount);
          return;
        }
        const text = input.value.trim();
        if (!text) return;
        await submitComment(pid, text, input);
      });

      dom.communityFeed.appendChild(card);
    });
  }

  async function handleVotePost(postId, voteType, cardEl) {
    const voterId = getActiveVoterId();
    try {
      const res = await fetch(`/api/community/posts/${postId}/vote`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: voterId, type: voteType })
      });
      const data = await res.json();
      if (data && data.success && data.vote) {
        const upCountSpan = cardEl.querySelector('.upvote-count');
        const downCountSpan = cardEl.querySelector('.downvote-count');
        const btnUp = cardEl.querySelector('.btn-upvote');
        const btnDown = cardEl.querySelector('.btn-downvote');

        if (upCountSpan) upCountSpan.textContent = data.vote.upvotes;
        if (downCountSpan) downCountSpan.textContent = data.vote.downvotes;

        if (btnUp) btnUp.classList.toggle('active', data.vote.user_vote === 'up');
        if (btnDown) btnDown.classList.toggle('active', data.vote.user_vote === 'down');

        if (data.vote.user_vote === 'up') {
          showToast('Marked helpful! ▲');
        } else if (data.vote.user_vote === 'down') {
          showToast('Marked unhelpful. ▼');
        } else {
          showToast('Vote removed.');
        }
      }
    } catch (err) {
      console.error('Vote failed:', err);
    }
  }

  async function loadPostComments(postId, initialComments) {
    const listEl = document.getElementById(`comments-list-${postId}`);
    if (!listEl) return;

    // Render initial comments immediately if available
    if (initialComments && initialComments.length > 0) {
      listEl.innerHTML = '';
      initialComments.forEach(c => {
        const item = document.createElement('div');
        item.className = 'comment-item';
        item.innerHTML = `
          <div class="comment-author">${escapeHtml(c.author || 'Grower')} <span class="comment-user-tag">${escapeHtml(c.author_badge || 'Grower')}</span><span style="font-size: 0.7rem; color: var(--text-4); font-weight: normal; margin-left: 4px;">${formatRelativeDate(c.created_at)}</span></div>
          <div class="comment-text">${escapeHtml(c.content)}</div>
        `;
        listEl.appendChild(item);
      });
    } else {
      listEl.innerHTML = `<div style="padding: 0.5rem; font-size: 0.8rem; color: var(--text-3);">Loading replies...</div>`;
    }

    try {
      const res = await fetch(`/api/community/posts/${postId}/comments`);
      const data = await res.json();
      if (data && data.comments) {
        if (data.comments.length === 0) {
          listEl.innerHTML = `<div style="padding: 0.5rem; font-size: 0.8rem; color: var(--text-3); font-style: italic;">No replies yet. Be the first to share agronomic advice!</div>`;
          return;
        }
        listEl.innerHTML = '';
        data.comments.forEach(c => {
          const item = document.createElement('div');
          item.className = 'comment-item';
          item.innerHTML = `
            <div class="comment-author">${escapeHtml(c.author || 'Grower')} <span class="comment-user-tag">${escapeHtml(c.author_badge || 'Grower')}</span><span style="font-size: 0.7rem; color: var(--text-4); font-weight: normal; margin-left: 4px;">${formatRelativeDate(c.created_at)}</span></div>
            <div class="comment-text">${escapeHtml(c.content)}</div>
          `;
          listEl.appendChild(item);
        });
      }
    } catch (e) {
      if (!initialComments || initialComments.length === 0) {
        listEl.innerHTML = `<div style="color: var(--red-600); font-size: 0.8rem;">Failed to load comments.</div>`;
      }
    }
  }

  async function submitComment(postId, content, inputEl) {
    if (!state.user || !state.user.username) {
      showToast('Please sign in or create an account to reply to fellow growers.');
      openModal(dom.modalUserAccount);
      return;
    }

    const author = state.user.username;
    try {
      const res = await fetch(`/api/community/posts/${postId}/comments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          author,
          author_badge: 'Grower',
          user_id: getCurrentUserId(),
          content
        })
      });
      const data = await res.json();
      if (data && data.success) {
        inputEl.value = '';
        showToast('Reply posted successfully!');
        loadPostComments(postId);
        const postCard = document.getElementById(`post-${postId}`);
        if (postCard) {
          const countEl = postCard.querySelector('.comment-count');
          if (countEl) countEl.textContent = parseInt(countEl.textContent || '0', 10) + 1;
        }
      }
    } catch (e) {
      showToast('Failed to post reply.');
    }
  }

  // ══════════════════════════════════════════════════════════════
  // SCAN PAGE: SINGLE SCAN & TIMELINE PROGRESSION
  // ══════════════════════════════════════════════════════════════
  function initScanView() {
    if (dom.scanTabSingle && dom.scanTabTimeline) {
      dom.scanTabSingle.onclick = () => switchScanMode('single');
      dom.scanTabTimeline.onclick = () => switchScanMode('timeline');
    }

    if (dom.scanCropSelect && !dom.scanCropSelect.value) {
      dom.scanCropSelect.value = state.selectedPlantId || '';
    }
    if (dom.timelineCropSelect && !dom.timelineCropSelect.value) {
      dom.timelineCropSelect.value = state.selectedPlantId || '';
    }

    initSingleScanHandlers();
    initTimelineScanHandlers();
  }

  function switchScanMode(mode) {
    state.scanMode = mode;
    if (dom.scanTabSingle) dom.scanTabSingle.classList.toggle('active', mode === 'single');
    if (dom.scanTabTimeline) dom.scanTabTimeline.classList.toggle('active', mode === 'timeline');

    if (dom.panelSingleScan) dom.panelSingleScan.style.display = (mode === 'single') ? 'block' : 'none';
    if (dom.panelTimelineScan) dom.panelTimelineScan.style.display = (mode === 'timeline') ? 'block' : 'none';
  }

  // --- Single Scan Handlers ---
  function initSingleScanHandlers() {
    if (!dom.singleDropZone) return;

    ['dragenter', 'dragover'].forEach(name => {
      dom.singleDropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dom.singleDropZone.classList.add('drag-active');
      });
    });
    ['dragleave', 'drop'].forEach(name => {
      dom.singleDropZone.addEventListener(name, (e) => {
        e.preventDefault();
        dom.singleDropZone.classList.remove('drag-active');
      });
    });
    dom.singleDropZone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files[0]) handleSingleFileSelected(files[0]);
    });

    if (dom.singleFileInput) {
      dom.singleFileInput.addEventListener('change', () => {
        if (dom.singleFileInput.files[0]) handleSingleFileSelected(dom.singleFileInput.files[0]);
      });
    }
    if (dom.singleCameraInput) {
      dom.singleCameraInput.addEventListener('change', () => {
        if (dom.singleCameraInput.files[0]) handleSingleFileSelected(dom.singleCameraInput.files[0]);
      });
    }

    if (dom.singleRemoveImgBtn) {
      dom.singleRemoveImgBtn.addEventListener('click', () => {
        state.singleScanFile = null;
        state.singleScanFileUrl = null;
        if (dom.singleFileInput) dom.singleFileInput.value = '';
        if (dom.singleCameraInput) dom.singleCameraInput.value = '';
        if (dom.singlePreviewBox) dom.singlePreviewBox.style.display = 'none';
        if (dom.singleDropZone) dom.singleDropZone.style.display = 'flex';
        if (dom.btnRunSingleScan) dom.btnRunSingleScan.disabled = true;
      });
    }

    if (dom.btnRunSingleScan) {
      dom.btnRunSingleScan.addEventListener('click', runSingleDiagnostic);
    }
  }

  function handleSingleFileSelected(file) {
    if (!file || !file.type.startsWith('image/')) {
      showToast('Please upload a valid image file (JPG or PNG).');
      return;
    }

    state.singleScanFile = file;
    state.singleScanFileUrl = URL.createObjectURL(file);

    if (dom.singlePreviewImg) dom.singlePreviewImg.src = state.singleScanFileUrl;
    if (dom.singlePreviewBox) dom.singlePreviewBox.style.display = 'flex';
    if (dom.singleDropZone) dom.singleDropZone.style.display = 'none';
    if (dom.btnRunSingleScan) dom.btnRunSingleScan.disabled = false;
  }

  async function runSingleDiagnostic() {
    if (!state.singleScanFile) {
      showToast('Please select or capture a leaf photo first.');
      return;
    }

    if (dom.btnRunSingleScan) {
      dom.btnRunSingleScan.disabled = true;
      dom.btnRunSingleScan.innerHTML = `<span class="loader-spinner-sm"></span> Analyzing Botanical Foliage...`;
    }

    const selectedCrop = (dom.scanCropSelect && dom.scanCropSelect.value) ? dom.scanCropSelect.value.trim() : '';
    const useWeather = document.getElementById('scan-weather-toggle')?.checked ?? true;
    const isField = document.getElementById('scan-field-toggle')?.checked ?? true;
    const customCity = (dom.scanCityInput && dom.scanCityInput.value) ? dom.scanCityInput.value.trim() : '';

    const formData = new FormData();
    formData.append('image', state.singleScanFile);
    formData.append('mode', isField ? 'field' : 'lab');
    formData.append('use_weather', useWeather ? 'true' : 'false');
    formData.append('architecture', 'mobilenet');
    formData.append('plant', selectedCrop); // Crop filter condition
    if (customCity) {
      formData.append('city', customCity);
    }

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();

      if (!res.ok || data.error) {
        showToast(data.error || 'Diagnostic evaluation failed.');
        return;
      }

      state.activeScanResult = data;
      renderSingleScanResult(data);
      showToast('Diagnostic completed successfully!');
    } catch (err) {
      console.error('Prediction failed:', err);
      showToast('Diagnostic request failed. Check server connection.');
    } finally {
      if (dom.btnRunSingleScan) {
        dom.btnRunSingleScan.disabled = false;
        dom.btnRunSingleScan.innerHTML = `🔬 Run Leaf Diagnostic`;
      }
    }
  }

  function renderSingleScanResult(result) {
    if (!dom.singleScanResultsCard) return;
    dom.singleScanResultsCard.style.display = 'block';

    const confPct = Number(result.confidence_pct || (result.confidence ? result.confidence * 100 : 0)).toFixed(1);
    const confTierRaw = String(result.confidence_tier || '').toLowerCase();
    const confTier = confTierRaw.includes('high') ? 'HIGH' : confTierRaw.includes('low') ? 'LOW' : (Number(confPct) >= 85 ? 'HIGH' : Number(confPct) >= 65 ? 'MEDIUM' : 'LOW');
    const predLabel = (result.prediction || 'Botanical Analysis').replace(/_/g, ' ');
    const cropName = result.plant ? result.plant.toUpperCase() : 'DETECTED PLANT';
    const isHealthy = Boolean(result.prediction && result.prediction.endsWith('_healthy'));
    const archName = (result.architecture || 'mobilenet').includes('efficient') ? '🎯 EfficientNet-B0' : '⚡ MobileNetV3';

    // SVG Confidence Ring Gauge calculation
    const P = 207.35;
    const strokeDash = P - (P * Math.min(Number(confPct), 100) / 100);
    const ringStroke = confPct >= 85 ? 'var(--emerald-600)' : confPct >= 65 ? 'var(--amber-500)' : 'var(--red-500)';

    // Grad-CAM Data (Keys: image, affected_pct, category, description, is_healthy)
    const gc = result.gradcam || {};
    const gcImage = gc.image || '';
    const lesionPct = Number(gc.affected_pct != null ? gc.affected_pct : 0).toFixed(1);
    const gcCategory = gc.category || (isHealthy || lesionPct < 1 ? 'Healthy Tissue' : lesionPct < 10 ? 'Mild Surface Damage' : lesionPct < 28 ? 'Moderate Spread' : 'Severe Blight');
    const gcSeverityClass = (isHealthy || lesionPct < 1) ? 'healthy' : lesionPct < 10 ? 'mild' : lesionPct < 28 ? 'moderate' : 'severe';

    // Weather & Microclimate Risk
    const risk = result.spread_risk || {};
    const riskLevel = risk.level || 'LOW';
    const riskExpl = risk.explanation || 'Optimal microclimate observed.';
    const wx = result.weather || {};
    const wxTemp = wx.temp_c != null ? `${wx.temp_c}°C` : '26°C';
    const wxHum = wx.humidity_pct != null ? `${wx.humidity_pct}%` : '65%';
    const wxRain = wx.rain_1h_mm != null ? `${wx.rain_1h_mm} mm/h` : '0 mm/h';

    // Care Advice
    const rawAdvice = typeof result.advice === 'string' ? result.advice : '';
    let organicAdvice = '';
    let chemicalAdvice = rawAdvice;
    if (rawAdvice.includes('Organic / biological options:')) {
      const parts = rawAdvice.split('Organic / biological options:');
      chemicalAdvice = parts[0].trim();
      organicAdvice = parts[1].trim();
    }

    // Top 3 Differential Bars
    let top3Html = '';
    if (result.top3 && result.top3.length) {
      top3Html = result.top3.map(([cls, prob]) => {
        const pPct = (prob * 100).toFixed(1);
        return `
          <div class="diff-item">
            <div class="diff-meta">
              <span>${escapeHtml(cls.replace(/_/g, ' '))}</span>
              <span>${pPct}%</span>
            </div>
            <div class="diff-track">
              <div class="diff-fill" style="width: ${pPct}%;"></div>
            </div>
          </div>
        `;
      }).join('');
    }

    dom.singleScanResultsCard.innerHTML = `
      <!-- Hero Diagnostic Header -->
      <div class="res-hero">
        <div>
          <div class="res-crop-tag">${escapeHtml(cropName)} DIAGNOSTIC REPORT</div>
          <h3 class="res-disease-title ${isHealthy ? 'healthy' : 'disease'}">${escapeHtml(predLabel.toUpperCase())}</h3>
          <div class="res-badges-wrap">
            <span class="btn-sm btn-outline-emerald" style="font-weight:700;">${archName}</span>
            ${result.forced ? `<span class="btn-sm" style="background:#fef3c7;color:#d97706;font-weight:700;">⚡ Forced Mode</span>` : ''}
          </div>
        </div>

        <!-- Confidence Gauge Ring -->
        <div class="gauge-wrap" title="Confidence Score">
          <svg class="g-svg" width="90" height="90">
            <circle class="g-bg" cx="45" cy="45" r="33"></circle>
            <circle class="g-fill" id="res-conf-ring" cx="45" cy="45" r="33"
                    stroke="${ringStroke}" stroke-dasharray="${P}" stroke-dashoffset="${strokeDash}"></circle>
          </svg>
          <div class="gauge-pct">${confPct}%</div>
        </div>
      </div>

      <!-- Diagnostic Certainty Banner -->
      <div class="tier-banner ${confTier.toLowerCase()}">
        ${confTier === 'HIGH'
          ? `<span>✅ <strong>High Diagnostic Certainty (${confPct}%)</strong> — Clear characteristic symptoms match verified dataset patterns.</span>`
          : confTier === 'MEDIUM'
          ? `<span>⚠️ <strong>Likely Indication (${confPct}%)</strong> — Probable symptoms detected. Cross-reference with physical inspection.</span>`
          : `<span>🔍 <strong>Low Confidence (${confPct}%)</strong> — Lighting or leaf angle may be suboptimal. Consider taking a closer photo.</span>`}
      </div>

      <!-- Grad-CAM Attention Heatmap Section -->
      ${gcImage ? `
        <div class="gc-section-card">
          <div class="gc-header">
            <div>
              <h4 class="gc-title">🎯 Explainable AI: Leaf Damage &amp; Attention Heatmap (Grad-CAM)</h4>
              <div class="gc-sub">Visual attention highlighting foliar regions that determined the diagnosis</div>
            </div>
            ${state.singleScanFileUrl ? `
              <button class="btn btn-secondary btn-sm" id="btn-toggle-gc" type="button">
                🔄 Toggle: Original Photo
              </button>
            ` : ''}
          </div>

          <!-- Bounded Image Container (Never Goes Off Screen) -->
          <div class="gc-media-wrap">
            <img id="gc-main-img" src="${gcImage}" alt="Grad-CAM Lesion Heatmap" class="gc-display-img" />
            <span class="gc-media-badge" id="gc-media-badge-tag">Grad-CAM Overlay</span>
          </div>

          <!-- Lesion Coverage Bar -->
          <div class="gc-metrics-strip">
            <div class="gc-metric-row">
              <span class="gc-metric-label">Estimated Lesion Surface Coverage:</span>
              <span class="gc-metric-val" style="color: ${isHealthy ? 'var(--emerald-600)' : 'var(--red-600)'};">${lesionPct}%</span>
            </div>
            <div class="gc-bar-track">
              <div class="gc-bar-fill ${gcSeverityClass}" style="width: ${Math.min(Number(lesionPct), 100)}%;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:var(--text-3);margin-top:4px;">
              <span>0% Healthy</span>
              <span>Severity Level: <strong>${escapeHtml(gcCategory)}</strong></span>
              <span>100% Full Leaf</span>
            </div>
          </div>
          ${gc.description ? `<p class="gc-desc-note">${escapeHtml(gc.description)}</p>` : ''}
        </div>
      ` : ''}

      <!-- Environmental Spread Risk -->
      <div class="risk-card">
        <div class="risk-header">
          <span class="risk-title">🌦️ Microclimate Disease Spread Risk</span>
          <span class="risk-level-badge ${riskLevel}">${riskLevel} SPREAD RISK</span>
        </div>
        <div class="risk-weather-row">
          <span>🌡️ Temp: ${wxTemp}</span>
          <span>💧 Humidity: ${wxHum}</span>
          <span>🌧️ Rain: ${wxRain}</span>
        </div>
        <p class="risk-explanation">${escapeHtml(riskExpl)}</p>
      </div>

      <!-- Agronomic Treatment & Care Protocol -->
      ${rawAdvice ? `
        <div class="care-card">
          <h4 class="care-title">🌱 Agronomic Management &amp; Treatment Protocol</h4>
          ${organicAdvice ? `<div class="care-item"><strong>🌿 Organic / Biological Solutions:</strong> ${escapeHtml(organicAdvice)}</div>` : ''}
          ${chemicalAdvice ? `<div class="care-item"><strong>🧪 Fungicidal &amp; Targeted Sprays:</strong> ${escapeHtml(chemicalAdvice)}</div>` : ''}
          <div class="care-item"><strong>🛡️ Prevention Protocol:</strong> Prune infected foliage, avoid overhead watering, and ensure good row aeration.</div>
        </div>
      ` : ''}

      <!-- Top 3 Classifier Probabilities -->
      ${top3Html ? `
        <div class="top3-card">
          <h4 class="top3-header">📋 Classifier Distribution (Top Differential Diagnoses)</h4>
          ${top3Html}
        </div>
      ` : ''}

      <!-- Action Buttons -->
      <div class="res-actions-strip">
        <button class="btn btn-primary btn-save-to-journal" type="button">
          📖 Save to My Plants Journal
        </button>
        <button class="btn btn-outline-emerald btn-compare-crop-card" type="button">
          🌿 View Crop Disease Library
        </button>
        <button class="btn btn-secondary btn-discuss-bot" type="button">
          💬 Discuss with AgroBot
        </button>
      </div>
    `;

    // Hook Grad-CAM View Toggle
    const toggleGcBtn = dom.singleScanResultsCard.querySelector('#btn-toggle-gc');
    const gcImgEl = dom.singleScanResultsCard.querySelector('#gc-main-img');
    const gcBadgeEl = dom.singleScanResultsCard.querySelector('#gc-media-badge-tag');

    if (toggleGcBtn && gcImgEl && state.singleScanFileUrl) {
      let showingHeatmap = true;
      toggleGcBtn.addEventListener('click', () => {
        showingHeatmap = !showingHeatmap;
        gcImgEl.src = showingHeatmap ? gcImage : state.singleScanFileUrl;
        gcBadgeEl.textContent = showingHeatmap ? 'Grad-CAM Overlay' : 'Original Leaf Photo';
        toggleGcBtn.textContent = showingHeatmap ? '🔄 View Original Leaf' : '🎯 View Grad-CAM Heatmap';
      });
    }

    // Hook Action Buttons
    dom.singleScanResultsCard.querySelector('.btn-save-to-journal').addEventListener('click', () => {
      openSaveJournalModal(result);
    });

    const btnCompareCrop = dom.singleScanResultsCard.querySelector('.btn-compare-crop-card');
    if (btnCompareCrop) {
      btnCompareCrop.addEventListener('click', () => {
        const cropId = (result.plant || state.selectedPlantId || 'tomato').toLowerCase();
        agroViewCropLibrary(cropId, result.prediction);
      });
    }

    dom.singleScanResultsCard.querySelector('.btn-discuss-bot').addEventListener('click', () => {
      openAgroBot(`I just scanned my leaf and it was diagnosed with ${predLabel} (${confPct}% confidence). What should I do next?`, result);
    });

    dom.singleScanResultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // --- Timeline Progression Handlers ---
  function initTimelineScanHandlers() {
    // 1. Manual Day Steppers (− / +)
    document.querySelectorAll('.btn-day-step').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.preventDefault();
        e.stopPropagation();
        const slot = btn.getAttribute('data-slot');
        const step = parseInt(btn.getAttribute('data-step'), 10) || 1;
        if (!state.timelineDays) {
          state.timelineDays = { day1: 1, day5: 5, day10: 10 };
        }
        const current = state.timelineDays[slot] != null ? state.timelineDays[slot] : (slot === 'day1' ? 1 : slot === 'day5' ? 5 : 10);
        const nextVal = Math.max(1, Math.min(180, current + step));
        state.timelineDays[slot] = nextVal;
        const lbl = document.getElementById(`slot-${slot}-label`);
        if (lbl) lbl.textContent = `Day ${nextVal}`;
      });
    });

    ['day1', 'day5', 'day10'].forEach(slot => {
      const input = document.getElementById(`slot-${slot}-input`);
      const preview = document.getElementById(`slot-${slot}-preview`);
      const drop = document.getElementById(`slot-${slot}-drop`);

      if (input) {
        input.addEventListener('change', () => {
          if (input.files && input.files[0]) {
            state.timelineFiles[slot] = input.files[0];
            const reader = new FileReader();
            reader.onload = (e) => {
              if (preview) {
                preview.src = e.target.result;
                preview.style.display = 'block';
              }
              if (drop) drop.classList.add('has-file');
              checkTimelineReady();
            };
            reader.readAsDataURL(input.files[0]);
          }
        });
      }
    });

    if (dom.btnRunTimelineScan) {
      dom.btnRunTimelineScan.addEventListener('click', runTimelineProgression);
    }
  }

  function checkTimelineReady() {
    const ready = !!(state.timelineFiles.day1 && state.timelineFiles.day5);
    if (dom.btnRunTimelineScan) dom.btnRunTimelineScan.disabled = !ready;
  }

  async function runTimelineProgression() {
    if (!state.timelineFiles.day1) {
      showToast('Please upload at least Day 1 and Day 5 leaf scans.');
      return;
    }

    if (dom.btnRunTimelineScan) {
      dom.btnRunTimelineScan.disabled = true;
      dom.btnRunTimelineScan.innerHTML = `<span class="loader-spinner-sm"></span> Computing Progression Dynamics...`;
    }

    const selectedCrop = (dom.timelineCropSelect && dom.timelineCropSelect.value) ? dom.timelineCropSelect.value.trim() : '';
    const formData = new FormData();

    formData.append('day1', state.timelineFiles.day1);
    if (state.timelineFiles.day5) formData.append('day5', state.timelineFiles.day5);
    if (state.timelineFiles.day10) formData.append('day10', state.timelineFiles.day10);

    const day1Num = (state.timelineDays && state.timelineDays.day1) || 1;
    const day5Num = (state.timelineDays && state.timelineDays.day5) || 5;
    const day10Num = (state.timelineDays && state.timelineDays.day10) || 10;

    const labels = [`Day ${day1Num}`];
    if (state.timelineFiles.day5) labels.push(`Day ${day5Num}`);
    if (state.timelineFiles.day10) labels.push(`Day ${day10Num}`);

    formData.append('labels', JSON.stringify(labels));
    formData.append('plant', selectedCrop);
    formData.append('use_weather', 'true');
    formData.append('mode', 'field');

    try {
      const res = await fetch('/batch_predict', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (!res.ok || data.error) {
        showToast(data.error || 'Progression evaluation failed.');
        return;
      }

      state.activeTimelineResult = data;
      renderTimelineProgressionResult(data);
      showToast('Progression analysis completed!');
    } catch (err) {
      console.error('Batch progression failed:', err);
      showToast('Progression request failed.');
    } finally {
      if (dom.btnRunTimelineScan) {
        dom.btnRunTimelineScan.disabled = false;
        dom.btnRunTimelineScan.innerHTML = `📈 Run Timeline Progression Analysis`;
      }
    }
  }

  function renderTimelineProgressionResult(data) {
    if (!dom.timelineResultsCard) return;
    dom.timelineResultsCard.style.display = 'block';

    const items = data.items || [];
    const summary = data.summary || {};
    const trajectory = summary.trajectory || 'STABLE';

    let trajBadge = 'badge-med';
    let trajIcon = '⚖️';
    if (trajectory.toLowerCase().includes('improv') || trajectory.toLowerCase().includes('recovering')) {
      trajBadge = 'badge-high';
      trajIcon = '🌿';
    } else if (trajectory.toLowerCase().includes('worsen') || trajectory.toLowerCase().includes('progressing')) {
      trajBadge = 'badge-low';
      trajIcon = '⚠️';
    }

    let itemsHtml = items.map(item => {
      const res = item.result || {};
      const score = item.severity_score !== undefined ? Math.round(item.severity_score * 100) : 50;
      const pred = res.prediction ? res.prediction.replace(/_/g, ' ') : 'Condition';
      const delta = (item.delta_from_previous !== null && item.delta_from_previous !== undefined)
        ? (item.delta_from_previous > 0 ? `+${Math.round(item.delta_from_previous * 100)}%` : `${Math.round(item.delta_from_previous * 100)}%`)
        : 'Baseline';

      return `
        <div class="timeline-step-card">
          <div class="ts-label">${escapeHtml(item.label)}</div>
          <div class="ts-score">${score}% Severity</div>
          <div class="ts-delta">${delta}</div>
          <div class="ts-pred">${escapeHtml(pred)}</div>
        </div>
      `;
    }).join('');

    dom.timelineResultsCard.innerHTML = `
      <div class="res-hero">
        <div>
          <div class="res-crop-tag">MULTI-DAY DISEASE TRAJECTORY</div>
          <h3 class="res-disease-title">Progression Tracking Report</h3>
        </div>
        <div class="tier-banner ${trajectory.toLowerCase().includes('worsen') ? 'low' : 'high'}" style="margin-bottom:0;">
          ${trajIcon} <strong>${escapeHtml(trajectory.toUpperCase())}</strong>
        </div>
      </div>

      <div class="timeline-steps-row">
        ${itemsHtml}
      </div>

      <div style="background: var(--surface-subtle); padding: 1.25rem; border-radius: var(--radius-lg); border: 1.5px solid var(--border); margin: 1.25rem 0;">
        <h4 style="font-size: 0.95rem; color: var(--emerald-900); margin-bottom: 0.5rem;">Agronomic Trajectory Assessment</h4>
        <p style="font-size: 0.85rem; color: var(--text-2); line-height: 1.5;">${escapeHtml(summary.notes || 'Sequential comparative leaf evaluation completed.')}</p>
      </div>

      <div class="res-actions-strip">
        <button class="btn btn-secondary" onclick="window.agroOpenAgroBot('Can you analyze my plant timeline progression results and recommend recovery protocols?')">
          💬 Discuss Trajectory with AgroBot
        </button>
      </div>
    `;

    dom.timelineResultsCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ══════════════════════════════════════════════════════════════
  // MY PLANTS: PERSONAL JOURNAL
  // ══════════════════════════════════════════════════════════════
  async function loadUserJournal() {
    if (!dom.plantsJournalGrid) return;
    const userId = getCurrentUserId();

    dom.plantsJournalGrid.innerHTML = `
      <div style="grid-column: 1 / -1; padding: 3rem; text-align: center; color: var(--text-3);">
        <div class="loader-spinner" style="margin: 0 auto 0.75rem;"></div>
        Loading your tracked crop records...
      </div>
    `;

    try {
      const res = await fetch(`/api/records?user_id=${encodeURIComponent(userId)}`);
      const data = await res.json();
      if (data && data.records) {
        state.userRecords = data.records;
        renderJournalGrid(data.records);
      } else {
        renderEmptyJournal();
      }
    } catch (err) {
      console.error('Failed to load user records:', err);
      renderEmptyJournal();
    }
  }

  function renderEmptyJournal() {
    if (!dom.plantsJournalGrid) return;
    dom.plantsJournalGrid.innerHTML = `
      <div style="grid-column: 1 / -1; padding: 3.5rem 1.5rem; text-align: center; background: #ffffff; border-radius: var(--radius-xl); border: 1.5px dashed var(--border);">
        <div style="font-size: 3rem; margin-bottom: 0.75rem;">🪴</div>
        <h3 style="color: var(--emerald-900); margin-bottom: 0.35rem;">Your Plant Journal is Empty</h3>
        <p style="color: var(--text-3); font-size: 0.88rem; max-width: 420px; margin: 0 auto 1.5rem;">
          Keep track of your garden plants, monitor leaf health over time, and compare symptom progression.
        </p>
        <button class="btn btn-primary" onclick="window.agroSwitchTab('scan')">
          📸 Scan a Plant to Start Tracking
        </button>
      </div>
    `;
  }

  function renderJournalGrid(records) {
    if (!dom.plantsJournalGrid) return;
    if (!records || records.length === 0) {
      renderEmptyJournal();
      return;
    }

    dom.plantsJournalGrid.innerHTML = '';
    records.forEach(rec => {
      const card = document.createElement('div');
      card.className = 'journal-card';

      const timeline = rec.timeline || [];
      const latest = timeline.length ? timeline[timeline.length - 1] : {};
      const status = rec.status || latest.prediction || 'Healthy';
      const isHealthy = status.toLowerCase().includes('healthy');

      card.innerHTML = `
        <div class="journal-card-header">
          <div>
            <h4 class="journal-crop-name">${escapeHtml(rec.plant_name || 'My Plant')}</h4>
            <div class="journal-crop-variety">${escapeHtml(rec.variety || 'Tracked Specimen')}</div>
          </div>
          <span class="journal-status-pill ${isHealthy ? 'status-healthy' : 'status-disease'}">
            ${escapeHtml(status.replace(/_/g, ' '))}
          </span>
        </div>

        <div class="journal-meta-row">
          <span>📅 Updated: ${formatRelativeDate(rec.updated_at || rec.created_at)}</span>
          <span>🔄 Check-ins: ${timeline.length}</span>
        </div>

        ${rec.notes ? `<div class="journal-notes">"${escapeHtml(rec.notes)}"</div>` : ''}

        <section class="journal-timeline" aria-label="Plant scan timeline">
          <h5 class="journal-timeline-title">Scan timeline</h5>
          ${timeline.map((entry, index) => `
            <article class="journal-timeline-entry">
              <div class="journal-timeline-marker">${index + 1}</div>
              <div class="journal-timeline-content">
                <div class="journal-timeline-heading">
                  <strong>${escapeHtml(entry.day_label || `Check-in ${index + 1}`)}</strong>
                  <time>${escapeHtml(entry.date || '')}</time>
                </div>
                <div class="journal-timeline-diagnosis">${escapeHtml(String(entry.prediction || 'Unknown').replace(/_/g, ' '))}</div>
                <div class="journal-timeline-meta">
                  ${entry.confidence_pct != null ? `${Number(entry.confidence_pct).toFixed(1)}% confidence · ` : ''}
                  ${entry.affected_pct != null ? `${Number(entry.affected_pct).toFixed(1)}% affected` : 'Area not recorded'}
                </div>
                ${entry.explanation ? `<p class="journal-timeline-note">${escapeHtml(entry.explanation)}</p>` : ''}
              </div>
            </article>
          `).join('')}
        </section>

        <div class="journal-card-footer">
          <button class="btn btn-secondary btn-checkin" type="button">
            📸 Add Follow-up Scan
          </button>
        </div>
      `;

      const checkinButton = card.querySelector('.btn-checkin');
      checkinButton.addEventListener('click', () => openFollowupScanModal(rec));

      dom.plantsJournalGrid.appendChild(card);
    });
  }

  function timelineDayForDate(record, selectedDate) {
    const timeline = record.timeline || [];
    const baseline = timeline[0] || {};
    const baselineDate = String(baseline.date || record.created_at || '').slice(0, 10);
    const baselineTime = Date.parse(`${baselineDate}T00:00:00Z`);
    const selectedTime = Date.parse(`${selectedDate}T00:00:00Z`);
    if (!Number.isFinite(baselineTime) || !Number.isFinite(selectedTime)) return 'Day 1';
    const dayNumber = Math.max(1, Math.floor((selectedTime - baselineTime) / 86400000) + 1);
    return `Day ${dayNumber}`;
  }

  function openFollowupScanModal(record) {
    const form = document.getElementById('form-followup-scan');
    const result = document.getElementById('checkin-result');
    const dateInput = document.getElementById('checkin-date');
    const dayInput = document.getElementById('checkin-day');
    if (!form || !dateInput || !dayInput || !dom.modalFollowupScan) return;

    state.activeCheckinRecord = record;
    form.reset();
    if (result) {
      result.hidden = true;
      result.innerHTML = '';
    }
    dateInput.value = new Date(Date.now() - new Date().getTimezoneOffset() * 60000).toISOString().slice(0, 10);
    dayInput.value = timelineDayForDate(record, dateInput.value);
    openModal(dom.modalFollowupScan);
    document.getElementById('checkin-image').focus();
  }

  function initFollowupScanForm() {
    const form = document.getElementById('form-followup-scan');
    const dateInput = document.getElementById('checkin-date');
    const dayInput = document.getElementById('checkin-day');
    const result = document.getElementById('checkin-result');
    if (!form || !dateInput || !dayInput || !result) return;

    dateInput.addEventListener('change', () => {
      if (state.activeCheckinRecord && dateInput.value) {
        dayInput.value = timelineDayForDate(state.activeCheckinRecord, dateInput.value);
      }
    });

    form.addEventListener('submit', async (event) => {
      event.preventDefault();
      const record = state.activeCheckinRecord;
      const recordId = record && (record.record_id || record._id || record.id);
      const image = form.querySelector('.checkin-image').files[0];
      if (!recordId || !image) {
        showToast('Select a leaf photo before adding a follow-up scan.');
        return;
      }

      const submitButton = form.querySelector('.checkin-submit');
      submitButton.disabled = true;
      submitButton.textContent = 'Analyzing follow-up...';
      result.hidden = true;
      const formData = new FormData();
      formData.append('image', image);
      formData.append('day_label', dayInput.value);
      formData.append('date', dateInput.value);
      formData.append('notes', form.querySelector('.checkin-notes').value.trim());
      formData.append('use_weather', 'true');
      formData.append('mode', 'field');

      try {
        const response = await fetch(`/api/records/${encodeURIComponent(recordId)}/checkin`, {
          method: 'POST',
          body: formData
        });
        const data = await response.json();
        if (!response.ok || !data.success) {
          showToast(data.error || 'Could not add the follow-up scan.');
          return;
        }

        const checkin = data.latest_checkin || {};
        const diagnostic = data.diagnostic || {};
        const gradcam = diagnostic.gradcam || {};
        const gradcamImage = checkin.gradcam_image || gradcam.image || '';
        const comparison = data.comparison || {};
        const risk = diagnostic.spread_risk || {};
        const weather = checkin.weather || diagnostic.weather || {};
        const detail = (label, value) => `
          <div class="checkin-result-detail">
            <strong>${escapeHtml(label)}</strong>${escapeHtml(String(value))}
          </div>
        `;
        const topClasses = Array.isArray(diagnostic.top3)
          ? diagnostic.top3.slice(0, 3).map(item => {
            const name = Array.isArray(item) ? item[0] : '';
            const probability = Array.isArray(item) ? Number(item[1]) : NaN;
            return name
              ? `${String(name).replace(/_/g, ' ')}${Number.isFinite(probability) ? ` (${(probability * 100).toFixed(1)}%)` : ''}`
              : '';
          }).filter(Boolean).join(' · ')
          : '';
        const detailsHtml = [
          detail('Crop tracked', record.plant_name || diagnostic.plant || 'Unknown'),
          detail('Confidence', `${Number(checkin.confidence_pct || diagnostic.confidence_pct || 0).toFixed(1)}%`),
          detail('Affected leaf area', `${Number(checkin.affected_pct || 0).toFixed(1)}% · ${gradcam.category || checkin.severity || 'Uncategorized'}`),
          detail('Change since previous scan', `${Number(comparison.delta || 0) > 0 ? '+' : ''}${Number(comparison.delta || 0).toFixed(1)} percentage points`),
          detail('Progression', checkin.status_tag || comparison.status_tag || checkin.verdict || 'Recorded'),
          detail('Spread risk', risk.level ? `${risk.level}${risk.score != null ? ` (${Number(risk.score).toFixed(0)}/100)` : ''}` : 'Unavailable'),
          detail('Weather at scan', [
            weather.temp_c != null ? `${weather.temp_c}°C` : '',
            weather.humidity_pct != null ? `${weather.humidity_pct}% humidity` : '',
            weather.rain_1h_mm != null ? `${weather.rain_1h_mm} mm rain` : ''
          ].filter(Boolean).join(' · ') || 'Unavailable'),
          detail('Model', diagnostic.architecture || 'Plant disease classifier')
        ].join('');
        const advice = diagnostic.advice || checkin.advice || '';
        result.innerHTML = `
          <strong>${escapeHtml(String(checkin.prediction || diagnostic.prediction || 'Diagnosis complete').replace(/_/g, ' '))}</strong>
          <div>${escapeHtml(checkin.day_label || dayInput.value)} · ${escapeHtml(checkin.date || dateInput.value)}${comparison.days_elapsed != null ? ` · ${Number(comparison.days_elapsed)} days since prior scan` : ''}</div>
          <div class="checkin-result-details">${detailsHtml}</div>
          ${comparison.explanation || risk.explanation
            ? `<p class="checkin-result-advice"><strong>Progress notes:</strong> ${escapeHtml(comparison.explanation || risk.explanation)}</p>`
            : ''}
          ${topClasses ? `<p class="checkin-result-advice"><strong>Other likely classes:</strong> ${escapeHtml(topClasses)}</p>` : ''}
          ${advice ? `<p class="checkin-result-advice"><strong>Care guidance:</strong> ${escapeHtml(advice)}</p>` : ''}
          ${gradcamImage
            ? `<img class="checkin-result-image" src="${escapeHtml(gradcamImage)}" alt="Grad-CAM heatmap for the follow-up scan" />`
            : '<p>Grad-CAM image was not available for this scan.</p>'}
        `;
        result.hidden = false;
        showToast('Follow-up scan added to this plant timeline.');
        await loadUserJournal();
      } catch (error) {
        console.error('Follow-up scan failed:', error);
        showToast('Network error adding the follow-up scan.');
      } finally {
        submitButton.disabled = false;
        submitButton.textContent = 'Analyze and add to timeline';
      }
    });
  }

  // ══════════════════════════════════════════════════════════════
  // AGROBOT AGRONOMIC CHATBOT DRAWER
  // ══════════════════════════════════════════════════════════════
  // ══════════════════════════════════════════════════════════════
  // AGROBOT AGRONOMIC CHATBOT DRAWER (Voice Input & Output)
  // ══════════════════════════════════════════════════════════════
  function initChatbot() {
    if (dom.chatFab) {
      dom.chatFab.addEventListener('click', toggleAgroBot);
    }
    if (dom.chatCloseBtn) {
      dom.chatCloseBtn.addEventListener('click', () => {
        if (dom.chatDrawer) dom.chatDrawer.classList.remove('open');
        if ('speechSynthesis' in window) window.speechSynthesis.cancel();
      });
    }
    if (dom.chatSendBtn) {
      dom.chatSendBtn.addEventListener('click', sendChatMessage);
    }
    if (dom.chatInput) {
      dom.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          sendChatMessage();
        }
      });
    }

    // 1. Language Selection (English, Hindi, Bengali)
    if (dom.chatLangSelect) {
      dom.chatLangSelect.addEventListener('change', () => {
        state.chatLanguage = dom.chatLangSelect.value;
        const flag = state.chatLanguage === 'Hindi' ? '🇮🇳 हिन्दी' : state.chatLanguage === 'Bengali' ? '🇧🇩 বাংলা' : '🇬🇧 English';
        showToast(`AgroBot language: ${flag}`);
      });
    }

    // 2. Voice Output (TTS Toggle)
    if (dom.chatTtsToggle) {
      dom.chatTtsToggle.addEventListener('click', () => {
        state.chatTtsEnabled = !state.chatTtsEnabled;
        dom.chatTtsToggle.classList.toggle('active', state.chatTtsEnabled);
        dom.chatTtsToggle.textContent = state.chatTtsEnabled ? '🔊' : '🔇';
        showToast(state.chatTtsEnabled ? 'Voice output enabled 🔊' : 'Voice output muted 🔇');
        if (!state.chatTtsEnabled && 'speechSynthesis' in window) {
          window.speechSynthesis.cancel();
        }
      });
    }

    // 3. Voice Input (Microphone Speech Recognition)
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRec && dom.chatVoiceBtn) {
      const recognition = new SpeechRec();
      recognition.continuous = false;
      recognition.interimResults = false;
      let isListening = false;

      recognition.onstart = () => {
        isListening = true;
        dom.chatVoiceBtn.classList.add('recording');
        if (dom.chatInput) dom.chatInput.placeholder = `Listening in ${state.chatLanguage}... Speak now`;
      };

      recognition.onresult = (e) => {
        const transcript = e.results && e.results[0] && e.results[0][0] && e.results[0][0].transcript;
        if (transcript && dom.chatInput) {
          dom.chatInput.value = transcript;
          setTimeout(() => sendChatMessage(), 250);
        }
      };

      recognition.onerror = () => {
        isListening = false;
        dom.chatVoiceBtn.classList.remove('recording');
        if (dom.chatInput) dom.chatInput.placeholder = "Ask about disease treatments, organic care...";
      };

      recognition.onend = () => {
        isListening = false;
        dom.chatVoiceBtn.classList.remove('recording');
        if (dom.chatInput) dom.chatInput.placeholder = "Ask about disease treatments, organic care...";
      };

      dom.chatVoiceBtn.addEventListener('click', () => {
        if (isListening) {
          recognition.stop();
        } else {
          const lCode = state.chatLanguage === 'Hindi' ? 'hi-IN' : state.chatLanguage === 'Bengali' ? 'bn-IN' : 'en-US';
          recognition.lang = lCode;
          try {
            recognition.start();
          } catch (err) {
            recognition.stop();
          }
        }
      });
    } else if (dom.chatVoiceBtn) {
      dom.chatVoiceBtn.addEventListener('click', () => {
        showToast('Voice input is supported in Chrome, Edge, and modern browsers.');
      });
    }
  }

  function speakChatMessage(text, lang = 'English') {
    if (!('speechSynthesis' in window)) return;
    try {
      window.speechSynthesis.cancel();
      const clean = text.replace(/[*#_`>]/g, '').replace(/https?:\/\/\S+/g, '').trim();
      if (!clean) return;
      const utt = new SpeechSynthesisUtterance(clean);
      const lCode = lang === 'Hindi' ? 'hi-IN' : lang === 'Bengali' ? 'bn-IN' : 'en-US';
      utt.lang = lCode;
      utt.rate = 1.0;
      window.speechSynthesis.speak(utt);
    } catch (e) {
      console.warn('TTS error:', e);
    }
  }

  function toggleAgroBot() {
    if (!dom.chatDrawer) return;
    const isOpen = dom.chatDrawer.classList.toggle('open');
    if (isOpen) {
      if (dom.chatMessages && dom.chatMessages.children.length === 0) {
        appendChatMessage('bot', "Hello! I'm AgroBot, your precision AI agronomist. Ask about crop diseases, organic remedies, or scan a leaf photo for a grounded diagnosis.");
      }
      if (dom.chatInput) dom.chatInput.focus();
    } else {
      if ('speechSynthesis' in window) window.speechSynthesis.cancel();
    }
  }

  function openAgroBot(initialPrompt = '', scanContext = null) {
    if (!dom.chatDrawer) return;
    dom.chatDrawer.classList.add('open');
    if (scanContext) {
      state.activeScanResult = scanContext;
    }
    if (initialPrompt && dom.chatInput) {
      dom.chatInput.value = initialPrompt;
      sendChatMessage();
    }
  }

  async function sendChatMessage() {
    if (!dom.chatInput) return;
    const text = dom.chatInput.value.trim();
    if (!text) return;

    dom.chatInput.value = '';
    appendChatMessage('user', text);

    const context = {};
    if (state.activeScanResult) {
      context.prediction = state.activeScanResult.prediction;
      context.plant_name = state.activeScanResult.plant;
      context.confidence_pct = state.activeScanResult.confidence_pct;
      if (state.activeScanResult.gradcam) {
        context.affected_pct = state.activeScanResult.gradcam.affected_pct;
        context.gradcam = {
          affected_pct: state.activeScanResult.gradcam.affected_pct,
          category: state.activeScanResult.gradcam.category
        };
      }
      context.weather = state.activeScanResult.weather || {};
    }

    const payload = {
      message: text,
      history: state.chatHistory.slice(-6),
      context: context,
      language: state.chatLanguage || 'English',
      user_id: getCurrentUserId()
    };

    const loadingId = appendChatMessage('bot', 'AgroBot is analyzing agronomic knowledge...');

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      removeChatMessage(loadingId);
      if (data && data.reply) {
        appendChatMessage('bot', data.reply);
        state.chatHistory.push({ role: 'user', content: text });
        state.chatHistory.push({ role: 'assistant', content: data.reply });
        if (state.chatTtsEnabled) {
          speakChatMessage(data.reply, state.chatLanguage);
        }
      } else {
        appendChatMessage('bot', "I couldn't process that question right now. Please try again.");
      }
    } catch (err) {
      removeChatMessage(loadingId);
      appendChatMessage('bot', 'Network error connecting to AgroBot.');
    }
  }

  function appendChatMessage(sender, text) {
    if (!dom.chatMessages) return;
    const id = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 5);
    const msg = document.createElement('div');
    msg.className = `chat-msg ${sender}`;
    msg.id = id;

    if (sender === 'bot') {
      msg.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 0.5rem;">
          <div class="chat-message-content" style="flex: 1;">${formatChatMessage(text)}</div>
          <button type="button" class="btn-bubble-tts" title="Listen" style="background:none;border:none;cursor:pointer;font-size:0.85rem;padding:0;opacity:0.75;flex-shrink:0;">🔊</button>
        </div>
      `;
      const btnSpeak = msg.querySelector('.btn-bubble-tts');
      if (btnSpeak) {
        btnSpeak.addEventListener('click', () => speakChatMessage(text, state.chatLanguage));
      }
    } else {
      msg.innerHTML = escapeHtml(text).replace(/\n/g, '<br/>');
    }

    dom.chatMessages.appendChild(msg);
    dom.chatMessages.scrollTop = dom.chatMessages.scrollHeight;
    return id;
  }

  function formatChatMessage(text) {
    const lines = escapeHtml(text).split('\n');
    const html = [];
    let inList = false;
    const formatInline = (line) => line
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>');
    for (const line of lines) {
      const heading = line.match(/^#{1,6}\s+(.+)$/);
      const bullet = line.match(/^\s*(?:[-*]|\d+\.)\s+(.+)$/);
      if (bullet) {
        if (!inList) {
          html.push('<ul>');
          inList = true;
        }
        html.push(`<li>${formatInline(bullet[1])}</li>`);
        continue;
      }
      if (inList) {
        html.push('</ul>');
        inList = false;
      }
      if (heading) {
        html.push(`<strong class="chat-message-heading">${formatInline(heading[1])}</strong>`);
      } else if (line.trim()) {
        html.push(`<p>${formatInline(line)}</p>`);
      }
    }
    if (inList) html.push('</ul>');
    return html.join('');
  }

  function removeChatMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  // ══════════════════════════════════════════════════════════════
  // MODALS & POPUPS
  // ══════════════════════════════════════════════════════════════
  function openModal(modalEl) {
    if (!modalEl) return;
    modalEl.classList.add('open');
    document.body.style.overflow = 'hidden';
  }

  function closeModal(modalEl) {
    if (!modalEl) return;
    modalEl.classList.remove('open');
    document.body.style.overflow = '';
  }

  function initModals() {
    document.querySelectorAll('.modal-backdrop').forEach(modal => {
      modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal(modal);
      });
      const closeBtn = modal.querySelector('.modal-close-btn');
      if (closeBtn) closeBtn.addEventListener('click', () => closeModal(modal));
    });

    if (dom.userTopPill) {
      dom.userTopPill.addEventListener('click', () => {
        renderUserAccountModal();
        openModal(dom.modalUserAccount);
      });
    }

    if (dom.fabCreatePost) {
      dom.fabCreatePost.addEventListener('click', () => {
        openCreatePostModal();
      });
    }

    if (dom.btnNewPlantEntry) {
      dom.btnNewPlantEntry.addEventListener('click', () => {
        switchTab('scan');
      });
    }

    if (dom.modalLightbox) {
      dom.modalLightbox.addEventListener('click', () => {
        closeModal(dom.modalLightbox);
      });
    }
  }

  function openLightbox(imgUrl) {
    if (!dom.modalLightbox || !dom.lightboxImg) return;
    dom.lightboxImg.src = imgUrl;
    openModal(dom.modalLightbox);
  }

  // --- User Account Modal ---
  function renderUserAccountModal() {
    if (!dom.modalUserAccount) return;
    const body = dom.modalUserAccount.querySelector('.modal-body');
    if (!body) return;

    if (state.user && state.user.username) {
      body.innerHTML = `
        <div style="text-align: center; margin-bottom: 1.5rem;">
          <div style="width: 64px; height: 64px; border-radius: 50%; background: linear-gradient(135deg, var(--emerald-600), var(--emerald-800)); color: #fff; font-size: 1.75rem; font-weight: 800; display: flex; align-items: center; justify-content: center; margin: 0 auto 0.75rem;">
            ${state.user.username.charAt(0).toUpperCase()}
          </div>
          <h3 style="color: var(--emerald-900); font-size: 1.25rem;">${escapeHtml(state.user.username)}</h3>
          <p style="color: var(--text-3); font-size: 0.85rem;">${escapeHtml(state.user.email || 'Registered Grower')}</p>
        </div>

        <div style="background: var(--surface-subtle); border: 1.5px solid var(--border); border-radius: var(--radius-lg); padding: 1rem; margin-bottom: 1.5rem;">
          <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 0.5rem;">
            <span style="color: var(--text-3);">Journal Records:</span>
            <strong style="color: var(--emerald-900);">${state.userRecords.length}</strong>
          </div>
          <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
            <span style="color: var(--text-3);">Default Crop:</span>
            <strong style="color: var(--emerald-900);">${state.selectedPlantId ? state.selectedPlantId.toUpperCase() : 'None Selected'}</strong>
          </div>
        </div>

        <button class="btn btn-secondary" id="btn-user-logout" style="width: 100%;" type="button">
          🚪 Sign Out
        </button>
      `;

      body.querySelector('#btn-user-logout').addEventListener('click', logoutUser);
    } else {
      body.innerHTML = `
        <div class="auth-tabs">
          <button class="auth-tab-btn active" id="tab-auth-login" type="button">Sign In</button>
          <button class="auth-tab-btn" id="tab-auth-register" type="button">Create Account</button>
        </div>

        <form id="form-auth-login">
          <div class="form-field">
            <label class="form-label">Username or Email</label>
            <input type="text" id="login-identifier" class="form-input" required autocomplete="username" />
          </div>
          <div class="form-field">
            <label class="form-label">Password</label>
            <input type="password" id="login-password" class="form-input" required autocomplete="current-password" />
          </div>
          <button type="submit" class="btn btn-primary" style="width: 100%; margin-top: 0.5rem;">Sign In</button>
        </form>

        <form id="form-auth-register" style="display: none;">
          <div class="form-field">
            <label class="form-label">Username</label>
            <input type="text" id="reg-username" class="form-input" required autocomplete="username" />
          </div>
          <div class="form-field">
            <label class="form-label">Email</label>
            <input type="email" id="reg-email" class="form-input" required autocomplete="email" />
          </div>
          <div class="form-field">
            <label class="form-label">Password</label>
            <input type="password" id="reg-password" class="form-input" required autocomplete="new-password" />
          </div>
          <button type="submit" class="btn btn-primary" style="width: 100%; margin-top: 0.5rem;">Create Grower Account</button>
        </form>
      `;

      const tabLogin = body.querySelector('#tab-auth-login');
      const tabReg = body.querySelector('#tab-auth-register');
      const formLogin = body.querySelector('#form-auth-login');
      const formReg = body.querySelector('#form-auth-register');

      tabLogin.onclick = () => {
        tabLogin.classList.add('active');
        tabReg.classList.remove('active');
        formLogin.style.display = 'block';
        formReg.style.display = 'none';
      };
      tabReg.onclick = () => {
        tabReg.classList.add('active');
        tabLogin.classList.remove('active');
        formReg.style.display = 'block';
        formLogin.style.display = 'none';
      };

      formLogin.onsubmit = async (e) => {
        e.preventDefault();
        const ident = body.querySelector('#login-identifier').value.trim();
        const pwd = body.querySelector('#login-password').value.trim();
        try {
          const res = await fetch('/api/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ identifier: ident, password: pwd })
          });
          const data = await res.json();
          if (res.ok && data.user) {
            state.user = data.user;
            localStorage.setItem('agro_user', JSON.stringify(data.user));
            updateUserUI();
            showToast(`Welcome back, ${data.user.username}!`);
            closeModal(dom.modalUserAccount);
            loadUserJournal();
          } else {
            showToast(data.error || 'Login failed.');
          }
        } catch (err) {
          showToast('Network error during login.');
        }
      };

      formReg.onsubmit = async (e) => {
        e.preventDefault();
        const username = body.querySelector('#reg-username').value.trim();
        const email = body.querySelector('#reg-email').value.trim();
        const password = body.querySelector('#reg-password').value.trim();
        try {
          const res = await fetch('/api/auth/register', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email, password })
          });
          const data = await res.json();
          if (res.ok && data.user) {
            state.user = data.user;
            localStorage.setItem('agro_user', JSON.stringify(data.user));
            updateUserUI();
            showToast(`Account created! Welcome, ${data.user.username}.`);
            closeModal(dom.modalUserAccount);
            loadUserJournal();
          } else {
            showToast(data.error || 'Registration failed.');
          }
        } catch (err) {
          showToast('Network error during registration.');
        }
      };
    }
  }

  // --- Create Post Modal ---
  function openCreatePostModal() {
    if (!state.user || !state.user.username) {
      showToast('Please sign in or create an account to start a discussion.');
      renderUserAccountModal();
      openModal(dom.modalUserAccount);
      return;
    }
    if (!dom.modalCreatePost) return;
    populateCropDropdowns();
    const sel = document.getElementById('post-crop-select');
    if (sel && state.selectedPlantId) {
      sel.value = state.selectedPlantId;
    }
    openModal(dom.modalCreatePost);
  }

  function initCreatePostForm() {
    const form = document.getElementById('form-create-post');
    if (!form) return;
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const content = document.getElementById('post-content-input').value.trim();
      const plantId = document.getElementById('post-crop-select').value || 'tomato';
      const diseaseName = document.getElementById('post-disease-input').value.trim();
      const author = (state.user && state.user.username) || 'Guest Grower';

      if (!content) {
        showToast('Please enter your post or question.');
        return;
      }

      const submitBtn = form.querySelector('button[type="submit"]');
      if (submitBtn) submitBtn.disabled = true;

      try {
        const res = await fetch('/api/community/posts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            author,
            content,
            plant_id: plantId,
            disease_name: diseaseName || 'Field Observation'
          })
        });
        const data = await res.json();
        if (data && data.success) {
          form.reset();
          closeModal(dom.modalCreatePost);
          showToast('Your post has been shared with the community!');
          fetchCommunityPosts(state.selectedPlantId);
        } else {
          showToast(data.error || 'Could not post.');
        }
      } catch (err) {
        showToast('Error sharing post.');
      } finally {
        if (submitBtn) submitBtn.disabled = false;
      }
    });
  }

  // --- Save Scan to Journal Modal ---
  function openSaveJournalModal(scanResult) {
    if (!dom.modalSaveJournal) return;
    state.activeScanForJournal = scanResult;
    const cropInput = document.getElementById('journal-crop-input');
    const statusInput = document.getElementById('journal-status-input');
    if (cropInput) {
      cropInput.value = scanResult.plant
        ? scanResult.plant.charAt(0).toUpperCase() + scanResult.plant.slice(1)
        : '';
    }
    if (statusInput) {
      statusInput.value = scanResult.prediction
        ? scanResult.prediction.replace(/_/g, ' ')
        : '';
    }
    openModal(dom.modalSaveJournal);
  }

  function initSaveJournalForm() {
    const form = document.getElementById('form-save-journal');
    if (!form) return;
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      const plantName = document.getElementById('journal-crop-input').value.trim() || 'My Plant';
      const variety = document.getElementById('journal-variety-input').value.trim();
      const notes = document.getElementById('journal-notes-input').value.trim();
      const userId = getCurrentUserId();

      const scan = state.activeScanForJournal;
      if (!scan || !scan.prediction) {
        showToast('No completed diagnostic is available to save.');
        return;
      }
      const payload = {
        user_id: userId,
        plant_name: plantName,
        variety: variety || 'Field Specimen',
        status: scan.prediction || 'Tracked Leaf',
        notes: notes,
        day_label: 'Day 1',
        diagnostic: scan
      };

      try {
        const res = await fetch('/api/records', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (res.ok && data.success) {
          form.reset();
          state.activeScanForJournal = null;
          closeModal(dom.modalSaveJournal);
          showToast('Plant saved to your personal journal!');
          loadUserJournal();
        } else {
          showToast(data.error || 'Failed saving to journal.');
        }
      } catch (e) {
        showToast('Error saving to journal.');
      }
    });
  }

  function agroScanCrop(plantId) {
    const targetCrop = plantId || state.selectedPlantId || '';
    if (targetCrop) {
      state.selectedPlantId = targetCrop;
      if (dom.scanCropSelect) dom.scanCropSelect.value = targetCrop;
      if (dom.timelineCropSelect) dom.timelineCropSelect.value = targetCrop;
    }
    switchTab('scan');
    const cropMeta = state.plantsCatalog.find(c => c.id === targetCrop);
    const cropName = cropMeta ? cropMeta.name : 'Crop';
    showToast(`Scanner ready for ${cropName}! Upload photo or capture leaf.`);
  }

  async function agroViewCropLibrary(cropId, diseaseId) {
    switchTab('home');
    await selectPlant(cropId);
    if (diseaseId) {
      setTimeout(() => {
        const dCard = document.getElementById(`dc-${diseaseId}`);
        if (dCard) {
          dCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
          dCard.style.outline = '3px solid var(--emerald-500)';
          dCard.style.borderRadius = 'var(--radius-lg)';
          setTimeout(() => { dCard.style.outline = ''; }, 3500);
        }
      }, 400);
    }
    const cropMeta = state.plantsCatalog.find(c => c.id === cropId);
    const cropName = cropMeta ? cropMeta.name : cropId;
    showToast(`Viewing verified ${cropName} Disease Library.`);
  }

  // ══════════════════════════════════════════════════════════════
  // EVENT LISTENERS INITIALIZATION
  // ══════════════════════════════════════════════════════════════
  function bindGlobalEvents() {
    dom.navBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const tab = btn.getAttribute('data-tab');
        switchTab(tab);
      });
    });

    dom.bnavBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        const tab = btn.getAttribute('data-tab');
        switchTab(tab);
      });
    });

    if (dom.btnBackToAllPlants) {
      dom.btnBackToAllPlants.addEventListener('click', deselectPlant);
    }

    if (dom.btnScanSelectedCrop) {
      dom.btnScanSelectedCrop.addEventListener('click', () => {
        agroScanCrop(state.selectedPlantId);
      });
    }



    window.agroSwitchTab = switchTab;
    window.agroSelectPlant = selectPlant;
    window.agroDeselectPlant = deselectPlant;
    window.agroScanCrop = agroScanCrop;
    window.agroViewCropLibrary = agroViewCropLibrary;
    window.agroOpenAgroBot = openAgroBot;
    window.agroOpenCreatePost = openCreatePostModal;
  }

  // ══════════════════════════════════════════════════════════════
  // BOOTSTRAP APPLICATION
  // ══════════════════════════════════════════════════════════════
  async function init() {
    initUser();
    bindGlobalEvents();
    initModals();
    initFollowupScanForm();
    initChatbot();
    initCreatePostForm();
    initSaveJournalForm();

    await fetchPlantsCatalog();
    await fetchCommunityPosts(null);

    switchTab('home');
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
