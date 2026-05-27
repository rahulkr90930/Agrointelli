document.addEventListener('DOMContentLoaded', () => {
    // Elements
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const btnBrowse = document.getElementById('btn-browse');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const btnRemoveImage = document.getElementById('btn-remove-image');
    
    const btnAnalyze = document.getElementById('btn-analyze');
    const btnSpinner = document.getElementById('btn-spinner');
    const fieldModeToggle = document.getElementById('field-mode-toggle');
    
    const emptyState = document.getElementById('empty-state');
    const skeletonState = document.getElementById('skeleton-state');
    const dashboardResults = document.getElementById('dashboard-results');
    
    const predictedDisease = document.getElementById('predicted-disease');
    const scientificName = document.getElementById('scientific-name');
    const confidenceBadge = document.getElementById('confidence-badge');
    const confidencePercentage = document.getElementById('confidence-percentage');
    const alternatesList = document.getElementById('alternates-list');
    
    const severityRadial = document.getElementById('severity-radial');
    const lesionPercentage = document.getElementById('lesion-percentage');
    const severityLevel = document.getElementById('severity-level');
    
    const qualityStatusBadge = document.getElementById('quality-status-badge');
    const barBrightness = document.getElementById('bar-brightness');
    const valBrightness = document.getElementById('val-brightness');
    const barContrast = document.getElementById('bar-contrast');
    const valContrast = document.getElementById('val-contrast');
    const barSharpness = document.getElementById('bar-sharpness');
    const valSharpness = document.getElementById('val-sharpness');
    const qualityWarningsBox = document.getElementById('quality-warnings-box');
    const adviceText = document.getElementById('advice-text');

    let uploadedFile = null;

    // Toggle Labels Active State
    fieldModeToggle.addEventListener('change', () => {
        const labels = document.querySelectorAll('.mode-label');
        labels.forEach(l => l.classList.toggle('active'));
    });

    // Browse Button
    btnBrowse.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    // File Input Selection
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and Drop
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        e.preventDefault();
        dropZone.classList.add('dragover');
    }

    function unhighlight(e) {
        e.preventDefault();
        dropZone.classList.remove('dragover');
    }

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            fileInput.files = files;
            handleFileSelect();
        }
    });

    function handleFileSelect() {
        const file = fileInput.files[0];
        if (!file) return;

        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (PNG, JPG, BMP, WEBP).');
            return;
        }

        uploadedFile = file;

        // Render Preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove('hidden');
            btnAnalyze.classList.remove('disabled');
            btnAnalyze.removeAttribute('disabled');
        };
        reader.readAsDataURL(file);
    }

    // Remove Image
    btnRemoveImage.addEventListener('click', (e) => {
        e.stopPropagation();
        resetImageUpload();
    });

    function resetImageUpload() {
        uploadedFile = null;
        fileInput.value = '';
        imagePreview.src = '#';
        previewContainer.classList.add('hidden');
        btnAnalyze.classList.add('disabled');
        btnAnalyze.setAttribute('disabled', 'true');
        
        // Return to empty state
        dashboardResults.classList.remove('visible');
        setTimeout(() => {
            dashboardResults.classList.add('hidden');
            emptyState.classList.remove('hidden');
        }, 300);
    }

    // Run Neural Inference
    btnAnalyze.addEventListener('click', async () => {
        if (!uploadedFile) return;

        // UI States
        btnAnalyze.classList.add('disabled');
        btnAnalyze.setAttribute('disabled', 'true');
        btnSpinner.classList.remove('hidden');
        
        emptyState.classList.add('hidden');
        dashboardResults.classList.remove('visible');
        dashboardResults.classList.add('hidden');
        skeletonState.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', uploadedFile);
        
        const fieldMode = fieldModeToggle.checked;
        
        try {
            const response = await fetch(`/predict?field_mode=${fieldMode}`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('API server returned an error.');
            }

            const data = await response.json();
            renderResults(data);

        } catch (error) {
            console.error('Inference Error:', error);
            alert(`Analysis failed: ${error.message}`);
            skeletonState.classList.add('hidden');
            emptyState.classList.remove('hidden');
        } finally {
            btnAnalyze.classList.remove('disabled');
            btnAnalyze.removeAttribute('disabled');
            btnSpinner.classList.add('hidden');
        }
    });

    // Formatting Helpers
    function cleanClassName(className) {
        // e.g. Tomato___Tomato_Yellow_Leaf_Curl_Virus -> Tomato: Yellow Leaf Curl Virus
        // e.g. Apple___Apple_scab -> Apple: Apple Scab
        if (!className) return 'Unknown Plant';
        
        const parts = className.split('___');
        let plant = parts[0] || '';
        let disease = parts[1] || 'Healthy';
        
        // Clean plant name
        plant = plant.replace(/_/g, ' ');
        
        // Clean disease name
        disease = disease.replace(/_/g, ' ');
        // If disease starts with plant name, remove redundancy
        if (disease.toLowerCase().startsWith(plant.toLowerCase())) {
            disease = disease.substring(plant.length).trim();
        }
        
        // Title Case Helper
        const toTitleCase = (str) => str.replace(/\b\w/g, c => c.toUpperCase());
        
        plant = toTitleCase(plant);
        disease = toTitleCase(disease);
        
        if (disease === '' || disease.toLowerCase() === 'healthy') {
            return `${plant} (Healthy)`;
        }
        
        return `${plant}: ${disease}`;
    }

    function getScientificName(className) {
        const map = {
            'apple': 'Malus domestica',
            'cherry': 'Prunus avium',
            'grape': 'Vitis vinifera',
            'peach': 'Prunus persica',
            'potato': 'Solanum tuberosum',
            'strawberry': 'Fragaria × ananassa',
            'tomato': 'Solanum lycopersicum',
            'pepper': 'Capsicum annuum',
            'corn': 'Zea mays',
            'squash': 'Cucurbita pepo',
            'orange': 'Citrus sinensis'
        };
        
        const firstWord = (className || '').split('___')[0].toLowerCase();
        return map[firstWord] || 'Plantae';
    }

    // Render Results to UI
    function renderResults(res) {
        skeletonState.classList.add('hidden');
        
        // 1. Predicted Disease
        const rawPrediction = res.prediction;
        predictedDisease.innerText = cleanClassName(rawPrediction);
        scientificName.innerText = getScientificName(rawPrediction);
        
        // 2. Confidence & Mode Badge
        const conf = res.confidence;
        confidencePercentage.innerText = `${(conf * 100).toFixed(1)}%`;
        
        confidenceBadge.innerText = res.mode;
        confidenceBadge.className = 'badge'; // reset
        if (conf >= 0.85) {
            confidenceBadge.classList.add('success');
        } else if (conf >= 0.55) {
            confidenceBadge.classList.add('medium');
        } else {
            confidenceBadge.classList.add('low');
        }
        
        // 3. Top-3 Alternatives
        alternatesList.innerHTML = '';
        res.top3.forEach(([name, p]) => {
            const row = document.createElement('div');
            row.className = 'alt-row';
            row.innerHTML = `
                <div class="alt-info">
                    <span class="alt-name">${cleanClassName(name)}</span>
                    <span class="alt-pct">${(p * 100).toFixed(1)}%</span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" style="width: ${(p * 100).toFixed(1)}%"></div>
                </div>
            `;
            alternatesList.appendChild(row);
        });
        
        // 4. Severity Proxy Radial
        const sev = res.severity_proxy;
        const ratio = sev.lesion_ratio;
        lesionPercentage.innerText = `${(ratio * 100).toFixed(1)}%`;
        severityLevel.innerText = sev.severity.toUpperCase();
        
        // SVG Radial stroke dashoffset calculation
        const radius = 42;
        const circumference = 2 * Math.PI * radius; // ~263.89
        const offset = circumference - (ratio * circumference);
        severityRadial.style.strokeDasharray = `${circumference}`;
        severityRadial.style.strokeDashoffset = offset;
        
        // Radial color based on severity
        if (sev.severity.toLowerCase().includes('healthy')) {
            severityRadial.style.stroke = 'var(--success)';
        } else if (sev.severity.toLowerCase().includes('early') || sev.severity.toLowerCase().includes('moderate')) {
            severityRadial.style.stroke = 'var(--warning)';
        } else {
            severityRadial.style.stroke = 'var(--danger)';
        }
        
        // 5. Image Quality Check
        const q = res.quality;
        qualityStatusBadge.className = 'quality-status'; // reset
        if (q.ok) {
            qualityStatusBadge.innerHTML = `<i class="fa-solid fa-circle-check"></i> Good Capture Quality`;
            qualityStatusBadge.classList.add('success');
            qualityWarningsBox.classList.add('hidden');
        } else {
            qualityStatusBadge.innerHTML = `<i class="fa-solid fa-circle-exclamation"></i> Quality Warnings`;
            qualityStatusBadge.classList.add('warning');
            
            // Populate warnings
            qualityWarningsBox.innerHTML = `<strong>Warnings:</strong> ${q.warnings.join(', ')}`;
            qualityWarningsBox.classList.remove('hidden');
        }
        
        // Normalize metrics to percentage of 255
        const brightnessPct = (q.brightness / 255 * 100).toFixed(0);
        const contrastPct = (q.contrast / 100 * 100).toFixed(0); // contrast usually maxes lower
        const sharpnessPct = (q.sharpness / 40 * 100).toFixed(0); // sharpness maxes around 30-40 in standard sets
        
        barBrightness.style.width = `${Math.min(brightnessPct, 100)}%`;
        valBrightness.innerText = q.brightness.toFixed(0);
        
        barContrast.style.width = `${Math.min(contrastPct, 100)}%`;
        valContrast.innerText = q.contrast.toFixed(0);
        
        barSharpness.style.width = `${Math.min(sharpnessPct, 100)}%`;
        valSharpness.innerText = q.sharpness.toFixed(1);
        
        // 6. Care Advice
        adviceText.innerText = res.advice;
        
        // Reveal Dashboard
        dashboardResults.classList.remove('hidden');
        setTimeout(() => {
            dashboardResults.classList.add('visible');
        }, 50);
    }
});
