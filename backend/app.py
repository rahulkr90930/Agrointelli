"""
AgroIntelli — Modular Flask REST Controller
Connects frontend requests to dedicated Python backend modules:
- database.py   : MongoDB Atlas persistence & user authentication
- weather.py    : OpenWeatherMap microclimate integration & progression evaluation
- gradcam.py    : Explainable AI saliency maps & lesion quantification
- inference.py  : Dual-model deep learning inference (MobileNetV3 & EfficientNet-B0)
- batch.py      : Multi-image chronological trajectory progression
"""

import sys
import uuid
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

# Suppress repetitive /health heartbeat log lines in Werkzeug console
class _HealthFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return "/health" not in msg

logging.getLogger("werkzeug").addFilter(_HealthFilter())

# Reconfigure stdout/stderr to UTF-8 on Windows
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

from pathlib import Path

# Add backend directory and project root to sys.path for direct script execution
backend_dir = Path(__file__).parent.resolve()
root_dir = backend_dir.parent.resolve()
for p in [str(backend_dir), str(root_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Import modular backend sub-systems
try:
    from .database import db_store
    from .weather import fetch_live_weather_snapshot, evaluate_weather_progression, parse_day_num
    from .inference import (
        load_model, run_prediction, predict_with_context, decode_uploaded_image,
        available_models, class_names, TF_AVAILABLE, model, normalize_plant_tag
    )
    from .batch import batch_progress_summary, normalize_labels
except (ImportError, ValueError):
    from database import db_store
    from weather import fetch_live_weather_snapshot, evaluate_weather_progression, parse_day_num
    from inference import (
        load_model, run_prediction, predict_with_context, decode_uploaded_image,
        available_models, class_names, TF_AVAILABLE, model, normalize_plant_tag
    )
    from batch import batch_progress_summary, normalize_labels

frontend_dir = Path(__file__).parent.parent / "frontend"
app = Flask(__name__, static_folder=str(frontend_dir), static_url_path="")
CORS(app)

# Initialize neural models and label mappings on startup
load_model()


# ── Root & Static Endpoints ───────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    """
    Root route: Unconditionally serves the interactive AgroIntelli Web Application.
    """
    frontend_file = frontend_dir / "index.html"
    if frontend_file.exists():
        from flask import send_file
        return send_file(str(frontend_file))
    return jsonify({"error": "frontend/index.html not found"}), 404


@app.route("/app", methods=["GET"])
def web_app():
    frontend_file = frontend_dir / "index.html"
    from flask import send_file
    return send_file(str(frontend_file))


@app.route("/styles.css", methods=["GET"])
def serve_styles():
    css_file = frontend_dir / "styles.css"
    if css_file.exists():
        from flask import send_file
        return send_file(str(css_file), mimetype="text/css")
    return "", 404


@app.route("/app.js", methods=["GET"])
def serve_app_js():
    js_file = frontend_dir / "app.js"
    if js_file.exists():
        from flask import send_file
        return send_file(str(js_file), mimetype="application/javascript")
    return "", 404


@app.route("/api", methods=["GET"])
@app.route("/api/status", methods=["GET"])
def api_status():
    """Returns JSON API and model availability telemetry."""
    try:
        from .inference import available_models as av_models, class_names as c_names, TF_AVAILABLE as tf_avail, model as m
    except (ImportError, ValueError):
        from inference import available_models as av_models, class_names as c_names, TF_AVAILABLE as tf_avail, model as m

    return jsonify({
        "name": "AgroIntelli Modular API",
        "status": "online",
        "modules": ["database", "weather", "gradcam", "inference", "batch"],
        "models_available": list(av_models.keys()),
        "mobilenet_ready": "mobilenet" in av_models or m is not None,
        "efficientnet_ready": "efficientnet" in av_models,
        "model_loaded": len(av_models) > 0 or m is not None,
        "classes_count": len(c_names),
        "mongo_connected": db_store.is_mongo
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": len(available_models) > 0 or model is not None,
        "mobilenet_ready": "mobilenet" in available_models or model is not None,
        "efficientnet_ready": "efficientnet" in available_models,
        "tf_available": TF_AVAILABLE,
        "classes": len(class_names),
        "storage_mode": "MongoDB" if db_store.is_mongo else "Resilient Local",
        "timestamp": datetime.now().isoformat()
    })


@app.route("/classes", methods=["GET"])
def get_classes():
    return jsonify({
        "classes": class_names or [
            "corn_common_rust", "corn_healthy",
            "pepper_bacterial_spot", "pepper_healthy",
            "potato_early_blight", "potato_healthy", "potato_late_blight",
            "tomato_early_blight", "tomato_healthy", "tomato_late_blight",
        ]
    })


@app.route("/weather", methods=["GET"])
@app.route("/api/weather", methods=["GET"])
def get_weather():
    """Fetch live weather for caller location or queried city via weather module."""
    lat = request.args.get("lat")
    lon = request.args.get("lon")
    city = request.args.get("city")
    payload = fetch_live_weather_snapshot(lat=lat, lon=lon, city=city)
    return jsonify(payload)


# ── Agronomic Knowledge Base Endpoints (for Chatbots & LLM Agents) ────────────

@app.route("/api/knowledge", methods=["GET"])
def list_knowledge():
    """
    GET /api/knowledge
    Exposes structured disease knowledge from CSV for external chatbots, RAG, and LLMs.
    """
    try:
        from .inference import get_all_knowledge
    except (ImportError, ValueError):
        from inference import get_all_knowledge
    store = get_all_knowledge()
    return jsonify({
        "success": True,
        "count": len(store),
        "knowledge": store
    })


@app.route("/api/knowledge/<disease_id>", methods=["GET"])
def get_disease_knowledge(disease_id):
    """
    GET /api/knowledge/<disease_id>
    Fetch complete treatment, organic remedies, pathogen, and environmental risks for a disease.
    """
    try:
        from .inference import get_knowledge_record
    except (ImportError, ValueError):
        from inference import get_knowledge_record
    rec = get_knowledge_record(disease_id)
    if not rec:
        return jsonify({"error": f"No knowledge record found for '{disease_id}'"}), 404
    return jsonify({"success": True, "record": rec})


# ── Plant Knowledge Ecosystem & Reference Images ─────────────────────────────

DATASET_RAW_DIR = Path(__file__).parent.parent / "notebooks" / "agrointelli_data" / "data" / "raw" / "plantvillage" / "Plant_leave_diseases_dataset_with_augmentation"

PLANTVILLAGE_DIR_MAP = {
    "apple_apple_scab": "Apple___Apple_scab",
    "apple_black_rot": "Apple___Black_rot",
    "apple_cedar_apple_rust": "Apple___Cedar_apple_rust",
    "apple_healthy": "Apple___healthy",
    "blueberry_healthy": "Blueberry___healthy",
    "cherry_healthy": "Cherry___healthy",
    "cherry_powdery_mildew": "Cherry___Powdery_mildew",
    "corn_gray_leaf_spot": "Corn___Cercospora_leaf_spot Gray_leaf_spot",
    "corn_common_rust": "Corn___Common_rust",
    "corn_healthy": "Corn___healthy",
    "corn_northern_leaf_blight": "Corn___Northern_Leaf_Blight",
    "grape_black_rot": "Grape___Black_rot",
    "grape_esca_black_measles": "Grape___Esca_(Black_Measles)",
    "grape_healthy": "Grape___healthy",
    "grape_leaf_blight": "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "orange_haunglongbing_citrus_greening": "Orange___Haunglongbing_(Citrus_greening)",
    "peach_bacterial_spot": "Peach___Bacterial_spot",
    "peach_healthy": "Peach___healthy",
    "pepper_bacterial_spot": "Pepper,_bell___Bacterial_spot",
    "pepper_healthy": "Pepper,_bell___healthy",
    "potato_early_blight": "Potato___Early_blight",
    "potato_healthy": "Potato___healthy",
    "potato_late_blight": "Potato___Late_blight",
    "raspberry_healthy": "Raspberry___healthy",
    "soybean_healthy": "Soybean___healthy",
    "squash_powdery_mildew": "Squash___Powdery_mildew",
    "strawberry_healthy": "Strawberry___healthy",
    "strawberry_leaf_scorch": "Strawberry___Leaf_scorch",
    "tomato_bacterial_spot": "Tomato___Bacterial_spot",
    "tomato_early_blight": "Tomato___Early_blight",
    "tomato_healthy": "Tomato___healthy",
    "tomato_late_blight": "Tomato___Late_blight",
    "tomato_leaf_mold": "Tomato___Leaf_Mold",
    "tomato_septoria_leaf_spot": "Tomato___Septoria_leaf_spot",
    "tomato_spider_mites_two_spotted_spider_mite": "Tomato___Spider_mites Two-spotted_spider_mite",
    "tomato_target_spot": "Tomato___Target_Spot",
    "tomato_mosaic_virus": "Tomato___Tomato_mosaic_virus",
    "tomato_yellow_leaf_curl_virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
}

CROPS_CATALOG = {
    "tomato": {
        "name": "Tomato",
        "botanical": "Solanum lycopersicum",
        "description": "High-value nightshade crop. Extremely vulnerable to early & late blights, bacterial spots, and leaf molds in humid conditions.",
        "icon": "🍅"
    },
    "potato": {
        "name": "Potato",
        "botanical": "Solanum tuberosum",
        "description": "Starchy tuber crop. Frequently threatened by aggressive Phytophthora late blight and Alternaria early blight.",
        "icon": "🥔"
    },
    "corn": {
        "name": "Corn",
        "botanical": "Zea mays",
        "description": "Staple cereal grain. Monitored for airborne common rust pustules, gray leaf spot lesions, and northern leaf blight.",
        "icon": "🌽"
    },
    "apple": {
        "name": "Apple",
        "botanical": "Malus domestica",
        "description": "Temperate tree fruit orchard crop. Susceptible to apple scab defoliation, black rot fruit mummies, and cedar rust.",
        "icon": "🍎"
    },
    "grape": {
        "name": "Grape",
        "botanical": "Vitis vinifera",
        "description": "Perennial vineyard climbing vine. Sensitive to black rot berries, esca black measles trunk decay, and pseudocercospora blight.",
        "icon": "🍇"
    },
    "pepper": {
        "name": "Pepper",
        "botanical": "Capsicum annuum",
        "description": "Warm-season bell pepper vegetable. Prone to bacterial spot epidemics spread by wind-driven rain splash.",
        "icon": "🫑"
    },
    "blueberry": {
        "name": "Blueberry",
        "botanical": "Vaccinium corymbosum",
        "description": "Acid-loving perennial berry shrub with glossy dark-green leaves and high nutrient requirements.",
        "icon": "🫐"
    },
    "cherry": {
        "name": "Cherry",
        "botanical": "Prunus avium",
        "description": "Stone fruit tree crop. Frequently impacted by powdery mildew white mycelial growth on young leaves.",
        "icon": "🍒"
    },
    "orange": {
        "name": "Orange",
        "botanical": "Citrus sinensis",
        "description": "Subtropical evergreen citrus tree threatened by psyllid-vectored Huanglongbing (Citrus Greening HLB).",
        "icon": "🍊"
    },
    "peach": {
        "name": "Peach",
        "botanical": "Prunus persica",
        "description": "Temperate stone fruit species. Affected by Xanthomonas bacterial spot causing shot-hole leaf lesions and defoliation.",
        "icon": "🍑"
    },
    "raspberry": {
        "name": "Raspberry",
        "botanical": "Rubus idaeus",
        "description": "Perennial cane fruit with compound leaves requiring trellis ventilation and steady soil moisture.",
        "icon": "🍓"
    },
    "soybean": {
        "name": "Soybean",
        "botanical": "Glycine max",
        "description": "High-protein legume crop with trifoliate leaves, vital for nitrogen fixation and soil health.",
        "icon": "🌱"
    },
    "squash": {
        "name": "Squash",
        "botanical": "Cucurbita pepo",
        "description": "Broad-leaf cucurbit crop highly prone to foliar powdery mildew fungus during late summer.",
        "icon": "🎃"
    },
    "strawberry": {
        "name": "Strawberry",
        "botanical": "Fragaria × ananassa",
        "description": "Low-growing perennial berry crop. Prone to fungal leaf scorch blotches in rainy or overhead-irrigated beds.",
        "icon": "🍓"
    }
}

_img_files_cache = {}

def get_disease_img_files(disease_id):
    if disease_id in _img_files_cache:
        return _img_files_cache[disease_id]
    # 1. Check bundled lightweight images in frontend/images/diseases/ (suitable for GitHub)
    bundled_dir = frontend_dir / "images" / "diseases" / disease_id
    if bundled_dir.exists():
        files = sorted(list(bundled_dir.glob("*.jpg")) + list(bundled_dir.glob("*.png")) + list(bundled_dir.glob("*.JPG")))
        if files:
            _img_files_cache[disease_id] = files
            return files

    # 2. Fall back to raw PlantVillage dataset directory if available
    folder = PLANTVILLAGE_DIR_MAP.get(disease_id)
    if not folder or not DATASET_RAW_DIR.exists():
        return []
    p = DATASET_RAW_DIR / folder
    if not p.exists():
        return []
    files = sorted(list(p.glob("*.JPG")) + list(p.glob("*.jpg")) + list(p.glob("*.png")))
    _img_files_cache[disease_id] = files
    return files



@app.route("/api/reference-images/<disease_id>/<int:img_idx>", methods=["GET"])
def get_reference_image(disease_id, img_idx):
    """Serves authentic reference leaf images from the real PlantVillage dataset."""
    from flask import send_file
    files = get_disease_img_files(disease_id)
    if not files or img_idx < 0 or img_idx >= len(files):
        return jsonify({"error": "Reference image not found"}), 404
    return send_file(str(files[img_idx]), mimetype="image/jpeg")


@app.route("/api/plants", methods=["GET"])
def list_plants():
    """
    GET /api/plants
    Returns all supported agricultural plants, their condition count, metadata,
    and supported diseases with reference image links and treatment overviews.
    """
    try:
        from .inference import get_all_knowledge
    except (ImportError, ValueError):
        from inference import get_all_knowledge

    knowledge_store = get_all_knowledge()
    posts = db_store.get_community_posts()

    # Precalculate post counts per plant and per disease
    plant_post_counts = {}
    disease_post_counts = {}
    for p in posts:
        pid = (p.get("plant_id") or "").lower()
        did = (p.get("disease_id") or "").lower()
        if pid:
            plant_post_counts[pid] = plant_post_counts.get(pid, 0) + 1
        if did:
            disease_post_counts[did] = disease_post_counts.get(did, 0) + 1

    plants_list = []
    for plant_id, meta in CROPS_CATALOG.items():
        # Find all disease records for this plant
        plant_diseases = []
        for d_id, rec in knowledge_store.items():
            if d_id == "background_without_leaves":
                continue
            crop_name = rec.get("crop", "").strip().lower()
            if crop_name == plant_id or d_id.startswith(f"{plant_id}_"):
                c_name = rec.get("common_name", d_id.replace("_", " ").title())
                # Clean name: remove redundant plant name prefix if present
                clean_name = c_name
                if clean_name.lower().startswith(meta["name"].lower()):
                    clean_name = clean_name[len(meta["name"]):].strip()
                if not clean_name:
                    clean_name = c_name

                img_files = get_disease_img_files(d_id)
                ref_urls = [f"/api/reference-images/{d_id}/{i}" for i in range(min(5, len(img_files)))]

                is_h = str(rec.get("is_healthy", "False")).lower() == "true"
                plant_diseases.append({
                    "id": d_id,
                    "disease_id": d_id,
                    "name": clean_name,
                    "full_name": c_name,
                    "is_healthy": is_h,
                    "pathogen": rec.get("pathogen", ""),
                    "symptoms": rec.get("risk_description", ""),
                    "treatment": rec.get("treatment_protocol", ""),
                    "organic_remedies": rec.get("organic_remedies", ""),
                    "prevention": rec.get("prevention", ""),
                    "optimal_temp": f"{rec.get('temp_min', 18)}–{rec.get('temp_max', 30)}°C",
                    "min_humidity": f"{rec.get('humidity_min', 60)}%",
                    "rain_sensitive": str(rec.get("rain_sensitive", "False")).lower() == "true",
                    "reference_images": ref_urls,
                    "community_posts_count": disease_post_counts.get(d_id, 0)
                })

        # Sort diseases: infected first, healthy last
        plant_diseases.sort(key=lambda x: (x["is_healthy"], x["name"]))

        # Image representative of the plant (prefer healthy class if available)
        healthy_d_id = f"{plant_id}_healthy"
        healthy_imgs = get_disease_img_files(healthy_d_id)
        if healthy_imgs:
            rep_img = f"/api/reference-images/{healthy_d_id}/0"
        elif plant_diseases and plant_diseases[0]["reference_images"]:
            rep_img = plant_diseases[0]["reference_images"][0]
        else:
            rep_img = None

        crop_img_url = f"/images/crops/{plant_id}.jpg" if (frontend_dir / "images" / "crops" / f"{plant_id}.jpg").exists() else rep_img

        plants_list.append({
            "id": plant_id,
            "name": meta["name"],
            "botanical": meta["botanical"],
            "description": meta["description"],
            "icon": meta["icon"],
            "image": crop_img_url,
            "disease_count": len(plant_diseases),
            "diseases": plant_diseases,
            "community_posts_count": plant_post_counts.get(plant_id, 0)
        })

    return jsonify({
        "success": True,
        "count": len(plants_list),
        "plants": plants_list
    })


@app.route("/api/plants/<plant_id>/diseases", methods=["GET"])
def list_plant_diseases(plant_id):
    """GET /api/plants/<plant_id>/diseases: Returns condition details for a specific crop."""
    target = plant_id.strip().lower()
    if target not in CROPS_CATALOG:
        return jsonify({"error": f"Plant '{plant_id}' is not recognized"}), 404

    try:
        from .inference import get_all_knowledge
    except (ImportError, ValueError):
        from inference import get_all_knowledge

    knowledge_store = get_all_knowledge()
    meta = CROPS_CATALOG[target]
    plant_diseases = []

    for d_id, rec in knowledge_store.items():
        if d_id == "background_without_leaves":
            continue
        crop_name = rec.get("crop", "").strip().lower()
        if crop_name == target or d_id.startswith(f"{target}_"):
            c_name = rec.get("common_name", d_id.replace("_", " ").title())
            clean_name = c_name
            if clean_name.lower().startswith(meta["name"].lower()):
                clean_name = clean_name[len(meta["name"]):].strip()
            if not clean_name:
                clean_name = c_name

            img_files = get_disease_img_files(d_id)
            ref_urls = [f"/api/reference-images/{d_id}/{i}" for i in range(min(5, len(img_files)))]

            plant_diseases.append({
                "id": d_id,
                "disease_id": d_id,
                "name": clean_name,
                "full_name": c_name,
                "is_healthy": str(rec.get("is_healthy", "False")).lower() == "true",
                "pathogen": rec.get("pathogen", ""),
                "symptoms": rec.get("risk_description", ""),
                "treatment": rec.get("treatment_protocol", ""),
                "organic_remedies": rec.get("organic_remedies", ""),
                "prevention": rec.get("prevention", ""),
                "reference_images": ref_urls
            })

    plant_diseases.sort(key=lambda x: (x["is_healthy"], x["name"]))
    return jsonify({
        "success": True,
        "plant_id": target,
        "plant_name": meta["name"],
        "count": len(plant_diseases),
        "diseases": plant_diseases
    })


# ── Plant Health Community Endpoints ─────────────────────────────────────────

@app.route("/api/community/posts", methods=["GET"])
def get_community_posts_endpoint():
    plant = request.args.get("plant")
    disease = request.args.get("disease")
    posts = db_store.get_community_posts(plant=plant, disease=disease)
    for p in posts:
        pid = p.get("post_id") or str(p.get("_id", ""))
        p["id"] = pid
        p["post_id"] = pid
        p.setdefault("comments", [])
        p.setdefault("comment_count", len(p.get("comments", [])))
    return jsonify({
        "success": True,
        "count": len(posts),
        "posts": posts
    })


@app.route("/api/community/posts", methods=["POST"])
def create_community_post_endpoint():
    data = request.get_json(silent=True) or {}
    author = (data.get("author") or "Guest Grower").strip()
    content = (data.get("content") or "").strip()
    if not content:
        return jsonify({"error": "Post text content is required."}), 400

    plant_id = (data.get("plant_id") or data.get("plant") or "tomato").strip().lower()
    disease_id = (data.get("disease_id") or data.get("disease") or "").strip().lower()

    plant_meta = CROPS_CATALOG.get(plant_id, {"name": plant_id.capitalize()})
    disease_name = data.get("disease_name")
    if not disease_name and disease_id:
        disease_name = disease_id.replace("_", " ").title()
    elif not disease_name:
        disease_name = "General Health Question"

    post = {
        "author": author,
        "author_badge": data.get("author_badge") or ("Grower" if author != "Guest Grower" else "Community Member"),
        "plant_id": plant_id,
        "plant_name": plant_meta["name"],
        "disease_id": disease_id,
        "disease_name": disease_name,
        "content": content,
        "image": data.get("image")
    }

    saved = db_store.save_community_post(post)
    return jsonify({"success": True, "post": saved}), 201


@app.route("/api/community/posts/<post_id>/vote", methods=["POST"])
def vote_post_endpoint(post_id):
    data = request.get_json(silent=True) or {}
    vote_type = data.get("type", "up")
    user_id = data.get("user_id", "guest")
    res = db_store.vote_community_post(post_id, vote_type=vote_type, user_id=user_id)
    if not res:
        return jsonify({"error": "Post not found"}), 404
    return jsonify({"success": True, "vote": res})


@app.route("/api/community/posts/<post_id>/comments", methods=["GET"])
def get_comments_endpoint(post_id):
    comments = db_store.get_post_comments(post_id)
    return jsonify({"success": True, "comments": comments})


@app.route("/api/community/posts/<post_id>/comments", methods=["POST"])
def add_comment_endpoint(post_id):
    data = request.get_json(silent=True) or {}
    author = (data.get("author") or "Community Member").strip()
    content = (data.get("content") or "").strip()
    if not content:
        return jsonify({"error": "Comment text is required."}), 400

    user_id = (data.get("user_id") or "guest").strip()
    author_badge = (data.get("author_badge") or ("Grower" if author not in ("Community Member", "Guest Grower") else "Community Member")).strip()
    comment = {
        "user_id": user_id,
        "author": author,
        "author_badge": author_badge,
        "content": content
    }
    res = db_store.add_post_comment(post_id, comment)
    if not res:
        return jsonify({"error": "Post not found"}), 404
    return jsonify({"success": True, "comment": res}), 201


# ── User Preference Endpoints (Last Selected Plant) ──────────────────────────

@app.route("/api/users/preference", methods=["POST"])
@app.route("/api/users/me/preferences", methods=["PATCH", "POST"])
def update_user_preferences():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id") or request.headers.get("X-User-Id")
    plant = data.get("last_selected_plant") or data.get("plant")
    if not user_id or not plant:
        return jsonify({"error": "user_id and last_selected_plant are required"}), 400

    clean_plant = plant.strip().lower()
    updated = db_store.update_user_preference(user_id, clean_plant)
    return jsonify({
        "success": True,
        "user_id": user_id,
        "last_selected_plant": clean_plant,
        "user": updated
    })



@app.route("/api/chat", methods=["POST"])
def chat():
    """
    POST /api/chat
    Agronomic chatbot query grounded in live leaf diagnosis or MongoDB historical records,
    Grad-CAM lesion coverage, weather, and scientific knowledge base.
    """
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "Message is required"}), 400

    context = data.get("context", {}) or {}
    history = data.get("history", [])
    user_id = data.get("user_id") or request.headers.get("X-User-Id") or "guest"

    # Query MongoDB for user's past records
    try:
        from .database import db_store
    except (ImportError, ValueError):
        from database import db_store

    user_records = []
    try:
        user_records = db_store.get_user_records(user_id)
    except Exception as e:
        print(f"MongoDB past records retrieval notice: {e}")

    # Build MongoDB history summary
    if user_records:
        history_lines = []
        for i, rec in enumerate(user_records[:6], 1):
            p_name = rec.get("plant_name", "Unknown Crop")
            status = rec.get("status", "Unknown Condition")
            up_at = (rec.get("updated_at") or "")[:10]
            timeline = rec.get("timeline", [])
            timeline_str = ""
            if timeline:
                latest_entry = timeline[-1]
                pred_label = latest_entry.get("prediction", status)
                conf = latest_entry.get("confidence_pct", 0)
                aff = latest_entry.get("affected_pct")
                aff_str = f", Lesion: {aff:.1f}%" if aff is not None else ""
                w = latest_entry.get("weather") or {}
                w_str = f", {w.get('temp_c')}°C {w.get('humidity_pct')}% hum" if w else ""
                timeline_str = f" [Check-ins: {len(timeline)} | Latest: {pred_label} ({conf:.0f}% conf{aff_str}{w_str})]"
            history_lines.append(f"{i}. Date: {up_at} | Plant: {p_name} | Disease: {status}{timeline_str}")

        context["mongo_history_summary"] = "\n".join(history_lines)
        context["mongo_records_count"] = len(user_records)

    # If NO live photo is uploaded, ground chatbot using the latest MongoDB record
    pred = context.get("prediction")
    if not pred and user_records:
        latest_rec = user_records[0]
        context["is_historical"] = True
        context["plant_name"] = latest_rec.get("plant_name", "Tracked Crop")
        timeline = latest_rec.get("timeline", [])
        if timeline:
            latest_entry = timeline[-1]
            pred = latest_entry.get("prediction") or latest_rec.get("status")
            context["prediction"] = pred
            context["confidence_pct"] = latest_entry.get("confidence_pct", 90)
            context["affected_pct"] = latest_entry.get("affected_pct", None)
            context["weather"] = latest_entry.get("weather") or {}
            context["timeline_notes"] = f"Tracked across {len(timeline)} check-ins in MongoDB ({latest_rec.get('plant_name')})"
        else:
            pred = latest_rec.get("status")
            context["prediction"] = pred

    # Automatically enrich with scientific knowledge base from CSV
    if pred and "knowledge_record" not in context:
        try:
            from .inference import get_knowledge_record
        except (ImportError, ValueError):
            from inference import get_knowledge_record
        rec = get_knowledge_record(pred)
        if rec:
            context["knowledge_record"] = rec

    language = data.get("language", "English")

    try:
        from .chatbot import generate_chat_response
    except (ImportError, ValueError):
        from chatbot import generate_chat_response

    result = generate_chat_response(message, scan_context=context, history=history, language=language)
    return jsonify({
        "success": True,
        "reply": result["reply"],
        "grounded": result["grounded"],
        "model_used": result["model_used"],
        "language": language,
        "used_mongo_history": bool(user_records),
        "is_historical": context.get("is_historical", False),
        "mongo_records_count": len(user_records),
        "latest_crop": user_records[0].get("plant_name") if user_records else None
    })


# ── Diagnostics & Predictions ────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Upload leaf image, select architecture (mobilenet / efficientnet),
    and receive top-3 predictions, Grad-CAM heatmap, and affected area percentage.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided in upload"}), 400

    file = request.files["image"]
    img_bgr = decode_uploaded_image(file)
    if img_bgr is None:
        return jsonify({"error": "Could not decode image. Use a valid JPG or PNG."}), 400

    field_mode = request.form.get("mode", "field") == "field"
    use_w_param = request.form.get("use_weather") or request.form.get("weather") or "true"
    use_weather = use_w_param.lower() in ("true", "1", "yes")
    arch = request.form.get("architecture") or request.args.get("arch")
    force_param = request.form.get("force") or request.args.get("force") or "false"
    force_diagnostic = str(force_param).lower() in ("true", "1", "yes")
    plant = request.form.get("plant") or request.args.get("plant")
    city = request.form.get("city") or request.args.get("city")
    lat = request.form.get("lat") or request.args.get("lat")
    lon = request.form.get("lon") or request.args.get("lon")

    weather = None
    if use_weather:
        weather_payload = fetch_live_weather_snapshot(lat=lat, lon=lon, city=city)
        weather = weather_payload.get("weather")

    try:
        result = run_prediction(img_bgr, field_mode=field_mode, weather=weather, model_choice=arch, force=force_diagnostic, plant=plant)
        result["weather"] = weather
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    POST /batch_predict
    Evaluates multi-image progression across user-defined timeline days.
    """
    uploaded_files = request.files.getlist("images")
    if not uploaded_files:
        for key in ("day1", "day5", "day10"):
            f = request.files.get(key)
            if f and f.filename:
                uploaded_files.append(f)

    if not uploaded_files:
        return jsonify({"error": "No images provided for batch analysis"}), 400

    import json
    raw_labels = request.form.get("labels", "[]")
    try:
        labels = json.loads(raw_labels) if raw_labels else []
        if not isinstance(labels, list):
            labels = []
    except Exception:
        labels = []

    labels = normalize_labels(labels, len(uploaded_files))
    field_mode = request.form.get("mode", "field") == "field"
    use_weather = request.form.get("use_weather", "false").lower() == "true"
    arch = request.form.get("architecture") or request.args.get("arch")
    force_param = request.form.get("force") or request.args.get("force") or "false"
    force_diagnostic = str(force_param).lower() in ("true", "1", "yes")
    plant = request.form.get("plant") or request.args.get("plant")

    session_weather = None
    if use_weather:
        weather_payload = fetch_live_weather_snapshot()
        if weather_payload.get("success"):
            session_weather = weather_payload.get("weather")

    items = []
    for idx, (label, file) in enumerate(zip(labels, uploaded_files)):
        img_bgr = decode_uploaded_image(file)
        if img_bgr is None:
            return jsonify({"error": f"Could not decode image for {label}. Use JPG or PNG."}), 400

        result = predict_with_context(img_bgr, field_mode=field_mode, weather=session_weather, model_choice=arch, force=force_diagnostic, plant=plant)
        item = {
            "label": label,
            "index": idx + 1,
            "filename": file.filename,
            "result": result,
            "severity_score": round(float(result.get("severity_score", 0.0)), 4),
            "weather": session_weather,
        }
        items.append(item)

    # Step deltas
    prev_score = None
    for item in items:
        item["delta_from_previous"] = None if prev_score is None else round(item["severity_score"] - prev_score, 4)
        prev_score = item["severity_score"]

    summary = batch_progress_summary(items, weather=session_weather)

    return jsonify({
        "items": items,
        "summary": summary,
        "weather": session_weather,
        "model_used": arch or "mobilenet"
    })


# ── User Authentication ───────────────────────────────────────────────────────

@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    email = data.get("email", "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password required."}), 400

    user, err = db_store.create_user(username, password, email)
    if err:
        return jsonify({"error": err}), 409
    return jsonify({"success": True, "user": user})


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json(silent=True) or {}
    identifier = (data.get("identifier") or data.get("username") or data.get("email") or "").strip()
    password = data.get("password", "").strip()

    if not identifier or not password:
        return jsonify({"error": "Username/email and password required."}), 400

    from werkzeug.security import check_password_hash
    user = db_store.get_user(identifier)
    if not user or not check_password_hash(user.get("password_hash", ""), password):
        return jsonify({"error": "Invalid username or password."}), 401

    clean = {k: v for k, v in user.items() if k not in ("password_hash", "_id")}
    clean.setdefault("last_selected_plant", user.get("last_selected_plant", "tomato"))
    return jsonify({"success": True, "user": clean})


# ── MongoDB Leaf Journal & Re-Check Operations ────────────────────────────────

@app.route("/api/records", methods=["GET"])
def list_records():
    user_id = request.args.get("user_id", "guest").strip()
    records = db_store.get_user_records(user_id)
    return jsonify({
        "success": True,
        "count": len(records),
        "records": records,
        "storage_mode": "MongoDB" if db_store.is_mongo else "Resilient Local"
    })


@app.route("/api/records/save", methods=["POST"])
@app.route("/api/records", methods=["POST"])
def save_record():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id", "guest").strip()
    plant_name = data.get("plant_name", "").strip() or "Tracked Crop Leaf"
    day_label = data.get("day_label", "Day 1").strip()
    notes = data.get("notes", "").strip()

    prediction_data = data.get("diagnostic") or data.get("timeline_entry") or {}
    if not isinstance(prediction_data, dict) or not prediction_data.get("prediction"):
        return jsonify({"error": "A completed diagnostic payload is required to save this journal entry."}), 400

    record_id = data.get("record_id") or f"leaf_{uuid.uuid4().hex[:10]}"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    day_num = parse_day_num(day_label) or 1

    pred_name = prediction_data.get("prediction", "Unknown")
    gradcam_data = prediction_data.get("gradcam")
    if not isinstance(gradcam_data, dict):
        gradcam_data = {}
    if "affected_pct" not in gradcam_data and "affected_pct" in prediction_data:
        gradcam_data["affected_pct"] = prediction_data["affected_pct"]
    severity_data = prediction_data.get("severity")
    if not isinstance(severity_data, dict):
        severity_data = {}
    aff_pct = gradcam_data.get("affected_pct", 0.0)
    sev_cat = gradcam_data.get("category", severity_data.get("severity", "Unknown"))
    gradcam_img = gradcam_data.get("image")
    weather_snap = prediction_data.get("weather")

    initial_checkin = {
        "checkin_id": uuid.uuid4().hex[:8],
        "day_number": day_num,
        "day_label": day_label,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": now_str,
        "prediction": pred_name,
        "confidence": prediction_data.get("confidence", 0.0),
        "confidence_pct": prediction_data.get("confidence_pct", 0.0),
        "affected_pct": aff_pct,
        "severity": sev_cat,
        "delta_from_previous": 0.0,
        "rate_per_day": 0.0,
        "verdict": "BASELINE",
        "status_tag": "🌱 Baseline Diagnosis",
        "explanation": f"Baseline leaf health assessment recorded ({day_label}).",
        "gradcam_image": gradcam_img,
        "weather": weather_snap,
        "notes": notes,
        "advice": prediction_data.get("advice", "")
    }

    record = {
        "record_id": record_id,
        "user_id": user_id,
        "plant_name": plant_name,
        "variety": data.get("variety", "").strip(),
        "created_at": now_str,
        "status": pred_name,
        "initial_prediction": pred_name,
        "latest_prediction": pred_name,
        "latest_affected_pct": aff_pct,
        "latest_severity": sev_cat,
        "latest_day": day_label,
        "latest_day_number": day_num,
        "latest_verdict": "Baseline scan logged.",
        "thumbnail": gradcam_img,
        "timeline": [initial_checkin],
        "notes": notes
    }

    saved = db_store.save_leaf_record(record)
    return jsonify({"success": True, "record": saved})


@app.route("/api/records/<record_id>", methods=["GET"])
def get_record(record_id):
    rec = db_store.get_leaf_record(record_id)
    if not rec:
        return jsonify({"error": "Record not found"}), 404
    return jsonify({"success": True, "record": rec})


@app.route("/api/records/<record_id>/checkin", methods=["POST"])
def recheck_leaf(record_id):
    """
    Submits follow-up photo for an existing tracked leaf,
    evaluates weather progression, computes spread delta across elapsed days, and updates MongoDB.
    """
    rec = db_store.get_leaf_record(record_id)
    if not rec:
        return jsonify({"error": "Leaf record not found"}), 404

    if "image" not in request.files:
        return jsonify({"error": "No leaf photo uploaded"}), 400

    file = request.files["image"]
    img_bgr = decode_uploaded_image(file)
    if img_bgr is None:
        return jsonify({"error": "Could not decode uploaded photo."}), 400

    day_label = request.form.get("day_label", "Follow-up Scan").strip()
    custom_date = request.form.get("date", "").strip()
    if not custom_date:
        custom_date = datetime.now().strftime("%Y-%m-%d")
    now_str = f"{custom_date} {datetime.now().strftime('%H:%M')}"

    notes = request.form.get("notes", "").strip()
    use_weather = request.form.get("use_weather", "false").lower() == "true"
    mode = request.form.get("mode", "field") == "field"
    arch = request.form.get("architecture") or request.args.get("arch")

    weather = None
    if use_weather:
        weather_payload = fetch_live_weather_snapshot()
        if weather_payload.get("success"):
            weather = weather_payload.get("weather")

    timeline = rec.get("timeline", [])
    baseline_entry = next(
        (
            entry
            for entry in timeline
            if isinstance(entry, dict) and entry.get("prediction")
        ),
        None,
    )
    baseline_prediction = (
        (baseline_entry or {}).get("prediction")
        or rec.get("initial_prediction")
    )
    crop_filter = normalize_plant_tag(request.form.get("plant"))
    if not crop_filter:
        crop_filter = normalize_plant_tag(rec.get("plant_name", ""))
    if not crop_filter or not any(
        class_name.startswith(f"{crop_filter}_") for class_name in class_names
    ):
        crop_filter = normalize_plant_tag(baseline_prediction)
    if crop_filter and not any(
        class_name.startswith(f"{crop_filter}_") for class_name in class_names
    ):
        crop_filter = None
    try:
        diag = run_prediction(
            img_bgr,
            field_mode=mode,
            weather=weather,
            model_choice=arch,
            plant=crop_filter
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422

    prev_entry = timeline[-1] if timeline else None

    new_pred = diag.get("prediction", "Unknown")
    new_aff = float(diag.get("gradcam", {}).get("affected_pct", 0.0))
    new_sev = diag.get("gradcam", {}).get("category", "Unknown")

    prev_pred = prev_entry.get("prediction", new_pred) if prev_entry else new_pred
    prev_aff  = float(prev_entry.get("affected_pct", new_aff)) if prev_entry else new_aff

    prev_day_num = parse_day_num(prev_entry.get("day_label", "Day 1")) if prev_entry else 1
    curr_day_num = parse_day_num(day_label)

    # Calculate calendar difference if available
    date_diff = None
    if prev_entry:
        prev_date_str = prev_entry.get("date") or (prev_entry.get("timestamp", "").split(" ")[0] if prev_entry.get("timestamp") else None)
        if prev_date_str and custom_date:
            try:
                d_prev = datetime.strptime(prev_date_str[:10], "%Y-%m-%d")
                d_curr = datetime.strptime(custom_date[:10], "%Y-%m-%d")
                diff = (d_curr - d_prev).days
                if diff > 0:
                    date_diff = diff
            except Exception:
                pass

    if curr_day_num is None:
        curr_day_num = (prev_day_num or 1) + (date_diff if date_diff else 7)

    if curr_day_num and prev_day_num and (curr_day_num > prev_day_num):
        days_elapsed = curr_day_num - prev_day_num
    elif date_diff is not None and date_diff > 0:
        days_elapsed = date_diff
    else:
        days_elapsed = max(curr_day_num - (prev_day_num or 1), 1) if curr_day_num else 1

    prev_weather = prev_entry.get("weather") if prev_entry else None
    eval_res = evaluate_weather_progression(
        prev_weather=prev_weather,
        curr_weather=weather,
        days_elapsed=days_elapsed,
        prev_pred=prev_pred,
        curr_pred=new_pred,
        prev_aff=prev_aff,
        curr_aff=new_aff,
        curr_day_label=day_label
    )

    verdict = eval_res["verdict"]
    status_tag = eval_res["status_tag"]
    expl = eval_res["explanation"]
    delta = eval_res["delta"]
    rate_per_day = eval_res["rate_per_day"]

    checkin_entry = {
        "checkin_id": uuid.uuid4().hex[:8],
        "day_number": curr_day_num,
        "day_label": day_label,
        "days_elapsed": days_elapsed,
        "date": custom_date,
        "timestamp": now_str,
        "prediction": new_pred,
        "confidence": diag.get("confidence", 0.0),
        "confidence_pct": diag.get("confidence_pct", 0.0),
        "affected_pct": new_aff,
        "severity": new_sev,
        "delta_from_previous": delta,
        "rate_per_day": rate_per_day,
        "verdict": verdict,
        "status_tag": status_tag,
        "explanation": expl,
        "gradcam_image": diag.get("gradcam", {}).get("image"),
        "weather": weather,
        "notes": notes,
        "advice": diag.get("advice", "")
    }

    timeline.append(checkin_entry)
    rec["timeline"] = timeline
    rec["latest_prediction"] = new_pred
    rec["status"] = new_pred
    rec["latest_affected_pct"] = new_aff
    rec["latest_severity"] = new_sev
    rec["latest_day"] = day_label
    rec["latest_day_number"] = curr_day_num
    rec["latest_verdict"] = expl

    saved = db_store.save_leaf_record(rec)
    prev_img = (prev_entry.get("gradcam_image") or rec.get("thumbnail")) if prev_entry else rec.get("thumbnail")

    return jsonify({
        "success": True,
        "crop_filter": crop_filter,
        "record": saved,
        "latest_checkin": checkin_entry,
        "diagnostic": diag,
        "comparison": {
            "previous_day": prev_entry.get("day_label", "Baseline") if prev_entry else "Baseline",
            "previous_date": (
                (prev_entry.get("date") or prev_entry.get("timestamp", "Initial Scan"))
                if prev_entry else "Initial Scan"
            ),
            "previous_image": prev_img,
            "previous_affected": prev_aff,
            "previous_prediction": prev_pred,
            "previous_weather": prev_weather,
            "current_day": day_label,
            "current_date": custom_date,
            "current_image": diag.get("gradcam", {}).get("image"),
            "current_affected": new_aff,
            "current_prediction": new_pred,
            "current_weather": weather,
            "days_elapsed": days_elapsed,
            "rate_per_day": rate_per_day,
            "delta": delta,
            "verdict": verdict,
            "status_tag": status_tag,
            "explanation": expl,
            "advice": diag.get("advice", "")
        }
    })


@app.route("/api/records/<record_id>", methods=["DELETE", "OPTIONS"])
def delete_record(record_id):
    if request.method == "OPTIONS":
        return jsonify({"success": True}), 200
    user_id = request.args.get("user_id", "").strip()
    if not user_id:
        return jsonify({"success": False, "error": "A user ID is required to delete a plant record."}), 400
    deleted = db_store.delete_record(record_id, user_id)
    if not deleted:
        return jsonify({"success": False, "deleted": False, "error": "Plant record not found for this user."}), 404
    return jsonify({"success": True, "deleted": True, "message": "Record removed."})


# ── Startup ───────────────────────────────────────────────────────────────────

def start_server():
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🌿 AgroIntelli backend running at http://0.0.0.0:{port}")
    print("   Frontend: open frontend/index.html in a browser or visit / in browser")
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    start_server()
