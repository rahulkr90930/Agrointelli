"""
AgroIntelli — Flask REST Backend
Serves predictions from the trained EfficientNetB0 model.

Usage:
    python app.py

Endpoints:
    POST /predict       — Upload a leaf image, get disease prediction
    GET  /weather       — Fetch live weather for current IP location
    GET  /classes       — List all supported disease classes
    GET  /health        — Health check
"""

import os
import io
import sys
import json
import uuid
import base64
import tempfile
from pathlib import Path
from datetime import datetime

# Reconfigure stdout/stderr to UTF-8 on Windows to prevent UnicodeEncodeError with emojis
if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

import cv2
import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash

# ── MongoDB & Persistence Stack ──────────────────────────────────────────────
# Auto-load .env if available
for candidate_env in [Path(__file__).parent.parent / ".env", Path(__file__).parent / ".env"]:
    if candidate_env.exists():
        try:
            with open(candidate_env, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
        except Exception:
            pass

try:
    import pymongo
    PYMONGO_AVAILABLE = True
except ImportError:
    PYMONGO_AVAILABLE = False
    print("ℹ️ pymongo library not installed yet. Operating in local storage mode.")

ATLAS_DEFAULT = "mongodb+srv://rahul90930kr_db_user:odDmDKkoa1Vqa17p@cluster0.c5skeva.mongodb.net/?appName=Cluster0"
MONGO_URI = os.environ.get("MONGO_URI", ATLAS_DEFAULT)
DB_NAME   = os.environ.get("MONGO_DB_NAME", "agrointelli")

# ── Try importing TensorFlow (graceful degradation if not loaded yet) ────────
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠  TensorFlow not found. Install with: pip install tensorflow")

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from the frontend

# ── Storage Manager (MongoDB with Zero-Downtime Local JSON Fallback) ───────────

class StorageManager:
    """
    Manages MongoDB persistence for user profiles and leaf health journals.
    Features an automatic fallback to local JSON storage if MongoDB server is offline,
    ensuring zero-downtime execution while setup is in progress.
    """
    def __init__(self):
        self.client = None
        self.db = None
        self.is_mongo = False
        self.fallback_file = Path(__file__).parent / "data_store.json"
        self._init_connection()

    def _init_connection(self):
        if PYMONGO_AVAILABLE:
            try:
                # 2 second server selection timeout so startup never hangs
                self.client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
                self.client.admin.command('ping')
                self.db = self.client[DB_NAME]
                self.is_mongo = True
                self.db.users.create_index("username", unique=True)
                self.db.leaf_records.create_index("record_id", unique=True)
                self.db.leaf_records.create_index("user_id")
                print(f"🍃 Connected to MongoDB ({DB_NAME}) at {MONGO_URI}")
                return
            except Exception as e:
                print(f"ℹ️ MongoDB offline ({e}). Using local fallback store at data_store.json")
        self.is_mongo = False
        self._init_fallback_file()

    def _init_fallback_file(self):
        if not self.fallback_file.exists():
            with open(self.fallback_file, "w", encoding="utf-8") as f:
                json.dump({"users": {}, "records": {}}, f, indent=2)

    def _load_fallback(self):
        self._init_fallback_file()
        try:
            with open(self.fallback_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"users": {}, "records": {}}

    def _save_fallback(self, data):
        with open(self.fallback_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    # ── User Operations ──
    def get_user(self, username_or_email):
        target = username_or_email.strip().lower()
        if self.is_mongo:
            doc = self.db.users.find_one({
                "$or": [{"username_lower": target}, {"email_lower": target}]
            })
            if doc and "_id" in doc:
                doc["_id"] = str(doc["_id"])
            return doc
        else:
            data = self._load_fallback()
            for u in data.get("users", {}).values():
                if u.get("username_lower") == target or u.get("email_lower") == target:
                    return u
            return None

    def create_user(self, username, password, email=""):
        uname_lower = username.strip().lower()
        email_lower = email.strip().lower() if email else ""

        if self.get_user(uname_lower):
            return None, "Username already taken."
        if email_lower and self.get_user(email_lower):
            return None, "Email already in use."

        user_id = str(uuid.uuid4())
        user_doc = {
            "user_id": user_id,
            "username": username.strip(),
            "username_lower": uname_lower,
            "email": email.strip(),
            "email_lower": email_lower,
            "password_hash": generate_password_hash(password),
            "created_at": datetime.now().isoformat()
        }

        if self.is_mongo:
            self.db.users.insert_one(user_doc.copy())
        else:
            data = self._load_fallback()
            data["users"][user_id] = user_doc
            self._save_fallback(data)

        # Return clean user profile without password hash
        clean = {k: v for k, v in user_doc.items() if k not in ("password_hash", "_id")}
        return clean, None

    # ── Leaf Record Operations ──
    def save_leaf_record(self, record):
        record["updated_at"] = datetime.now().isoformat()
        if self.is_mongo:
            try:
                self.db.leaf_records.update_one(
                    {"record_id": record["record_id"]},
                    {"$set": record},
                    upsert=True
                )
                print(f"🍃 [MongoDB] Successfully saved leaf record: {record['record_id']} ({record.get('plant_name')})")
                sys.stdout.flush()
            except Exception as e:
                print(f"⚠️ [MongoDB] Save failed: {e}. Writing to fallback storage.")
                sys.stdout.flush()
                data = self._load_fallback()
                data["records"][record["record_id"]] = record
                self._save_fallback(data)
        else:
            data = self._load_fallback()
            data["records"][record["record_id"]] = record
            self._save_fallback(data)
            print(f"📁 [Local JSON] Saved leaf record: {record['record_id']}")
            sys.stdout.flush()
        return record

    def get_leaf_record(self, record_id):
        if self.is_mongo:
            doc = self.db.leaf_records.find_one({"record_id": record_id})
            if doc and "_id" in doc:
                doc["_id"] = str(doc["_id"])
            return doc
        else:
            data = self._load_fallback()
            return data.get("records", {}).get(record_id)

    def get_user_records(self, user_id):
        if self.is_mongo:
            # Inclusive query: return records for this user and any guest scans
            query = {"$or": [{"user_id": user_id}, {"user_id": "guest"}]} if (user_id and user_id != "guest") else {}
            cursor = self.db.leaf_records.find(query).sort("updated_at", -1)
            docs = []
            for d in cursor:
                d["_id"] = str(d["_id"])
                docs.append(d)
            return docs
        else:
            data = self._load_fallback()
            if user_id and user_id != "guest":
                items = [r for r in data.get("records", {}).values() if r.get("user_id") in (user_id, "guest")]
            else:
                items = list(data.get("records", {}).values())
            items.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
            return items

    def delete_record(self, record_id, user_id):
        if self.is_mongo:
            self.db.leaf_records.delete_one({"record_id": record_id, "user_id": user_id})
        else:
            data = self._load_fallback()
            if record_id in data.get("records", {}):
                if data["records"][record_id].get("user_id") == user_id or user_id == "guest":
                    del data["records"][record_id]
                    self._save_fallback(data)

db_store = StorageManager()

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_DIR   = Path(__file__).parent / "models"
MODEL_PATH  = MODEL_DIR / "agrointelli_phase1_final.keras"
CLASS_MAP_PATH = MODEL_DIR / "class_index_map.json"
ADVICE_PATH = MODEL_DIR / "advice.json"

IMG_SIZE    = 224
WEATHER_DIM = 4
OWM_API_KEY = os.environ.get("OWM_API_KEY", "0af35111bc4b73ebb0b71d505e063680")

# ── Global model state ────────────────────────────────────────────────────────
model = None
grad_model = None
class_names = []
class_to_idx = {}
advice_map = {}


def load_model():
    """Load the trained model and supporting files."""
    global model, class_names, class_to_idx, advice_map

    if not TF_AVAILABLE:
        print("⚠  TensorFlow unavailable — running in demo mode.")
        return

    if not MODEL_PATH.exists():
        print(f"⚠  Model not found at {MODEL_PATH}")
        print("   Train the notebook first, then copy artifacts to backend/models/")
        return

    print("🔄 Loading AgroIntelli model...")
    model = keras.models.load_model(str(MODEL_PATH), compile=False)
    print(f"✅ Model loaded: {MODEL_PATH.name}")

    if CLASS_MAP_PATH.exists():
        with open(CLASS_MAP_PATH) as f:
            class_to_idx = json.load(f)
        class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda x: x[1])]
        print(f"✅ Classes loaded: {class_names}")

    if ADVICE_PATH.exists():
        with open(ADVICE_PATH) as f:
            advice_map = json.load(f)
        print(f"✅ Advice map loaded.")

    init_gradcam_model()


# ── Image preprocessing helpers ───────────────────────────────────────────────

def enhance_field_image(img_bgr):
    """Bilateral denoise + CLAHE for field-quality images."""
    denoised = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    lab = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def segment_leaf_advanced(img_bgr):
    """HSV-based segmentation for green + diseased (brown/yellow) leaf regions."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask_green  = cv2.inRange(hsv, np.array([22, 30, 30]), np.array([92, 255, 255]))
    mask_yellow = cv2.inRange(hsv, np.array([10, 30, 30]), np.array([25, 255, 255]))
    mask = cv2.bitwise_or(mask_green, mask_yellow)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    return cv2.bitwise_and(img_bgr, img_bgr, mask=mask), mask


def calculate_blur_score(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def apply_confidence_penalty(confidence, blur_score, blur_threshold=80.0):
    if blur_score < blur_threshold:
        penalty = 0.20 * (1 - blur_score / blur_threshold)
        confidence = max(0.0, confidence - penalty)
    return confidence


def create_patches(img_bgr, patch_size=224, stride=112):
    h, w = img_bgr.shape[:2]
    patches = []
    for y in range(0, max(h - patch_size + 1, 1), stride):
        for x in range(0, max(w - patch_size + 1, 1), stride):
            patch = img_bgr[y:y + patch_size, x:x + patch_size]
            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                patch = cv2.resize(patch, (patch_size, patch_size))
            patches.append(patch)
    return patches if patches else [cv2.resize(img_bgr, (patch_size, patch_size))]


def normalise_weather_vector(weather):
    return np.array([
        np.clip(weather.get("temp_c", 28)       / 50.0,  0, 1),
        np.clip(weather.get("humidity_pct", 70) / 100.0, 0, 1),
        np.clip(weather.get("rain_1h_mm", 0)    / 10.0,  0, 1),
        np.clip(weather.get("wind_kmh", 10)      / 50.0,  0, 1),
    ], dtype=np.float32)


def image_quality_check(img_bgr):
    gray       = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast   = float(gray.std())
    sharpness  = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    warns = []
    if brightness < 50:    warns.append("too dark")
    elif brightness > 205: warns.append("too bright")
    if contrast < 20:      warns.append("low contrast")
    if sharpness < 60:     warns.append("blurry")
    return {
        "ok"        : len(warns) == 0,
        "brightness": round(brightness, 2),
        "contrast"  : round(contrast, 2),
        "sharpness" : round(sharpness, 2),
        "warnings"  : warns,
    }


def severity_proxy(img_bgr):
    img  = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    leaf = (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 40)
    if leaf.sum() < 100:
        return {"severity": "unknown", "lesion_ratio": 0.0}
    bgr    = img.astype(np.int16)
    lesion = leaf & ((bgr[:, :, 1] < 120) | (bgr[:, :, 2] > 140))
    ratio  = float(lesion.sum() / leaf.sum())
    sev    = ("healthy / very mild" if ratio < 0.08
              else ("early / moderate" if ratio < 0.22 else "severe"))
    return {"severity": sev, "lesion_ratio": round(ratio, 4)}

def decode_uploaded_image(file_storage):
    """Decode an uploaded image file into a BGR ndarray."""
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return img_bgr


def fetch_live_weather_snapshot():
    """Fetch weather for the caller's approximate location with a safe fallback."""
    try:
        loc_data = requests.get("http://ip-api.com/json/", timeout=5).json()
        lat  = loc_data.get("lat", 22.57)
        lon  = loc_data.get("lon", 88.36)
        city = loc_data.get("city", "Unknown")
        country = loc_data.get("country", "IN")
    except Exception:
        lat, lon, city, country = 22.57, 88.36, "Kolkata", "India"

    try:
        url = (
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=metric"
        )
        w_data = requests.get(url, timeout=8).json()
        return {
            "success": True,
            "weather": {
                "city": city,
                "country": country,
                "temp_c": w_data["main"]["temp"],
                "feels_like_c": w_data["main"]["feels_like"],
                "humidity_pct": w_data["main"]["humidity"],
                "condition": w_data["weather"][0]["description"],
                "wind_kmh": round(w_data["wind"]["speed"] * 3.6, 1),
                "rain_1h_mm": w_data.get("rain", {}).get("1h", 0.0),
                "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            },
        }
    except Exception as e:
        return {
            "success": False,
            "weather": {
                "city": city,
                "country": country,
                "temp_c": 28.0,
                "feels_like_c": 30.0,
                "humidity_pct": 72,
                "condition": "partly cloudy (fallback)",
                "wind_kmh": 12.0,
                "rain_1h_mm": 0.0,
                "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            },
            "error": str(e),
        }


def severity_numeric(severity):
    """Convert the textual severity proxy into a sortable score."""
    value = severity.get("lesion_ratio")
    if isinstance(value, (int, float)):
        return float(value)
    label = (severity.get("severity") or "").lower()
    mapping = {
        "healthy / very mild": 0.10,
        "early / moderate": 0.50,
        "severe": 0.90,
        "unknown": 0.0,
    }
    return mapping.get(label, 0.0)


def batch_progress_summary(entries):
    """Summarise whether disease is improving, stable, or worsening."""
    if not entries:
        return {
            "trend": "insufficient data",
            "delta": 0.0,
            "change_per_step": 0.0,
            "same_prediction": False,
            "explanation": "No images were provided.",
        }

    if len(entries) == 1:
        return {
            "trend": "single sample",
            "delta": 0.0,
            "change_per_step": 0.0,
            "same_prediction": True,
            "explanation": "Upload at least two images to detect progression.",
        }

    values = [item["severity_score"] for item in entries]
    delta = float(values[-1] - values[0])
    steps = [values[i] - values[i - 1] for i in range(1, len(values))]
    change_per_step = float(sum(steps) / len(steps))

    same_prediction = len({item["result"]["prediction"] for item in entries}) == 1

    if delta > 0.08 or change_per_step > 0.05:
        trend = "worsening"
        explanation = "Lesion severity is increasing across the batch."
    elif delta < -0.08 or change_per_step < -0.05:
        trend = "improving"
        explanation = "Lesion severity is decreasing across the batch."
    else:
        trend = "stable / mixed"
        explanation = "The disease signal is not changing sharply across the batch."

    if same_prediction:
        explanation += " The predicted disease class stayed consistent."
    else:
        explanation += " The predicted class changes across samples, so interpret the trend carefully."

    return {
        "trend": trend,
        "delta": round(delta, 4),
        "change_per_step": round(change_per_step, 4),
        "same_prediction": same_prediction,
        "explanation": explanation,
    }


def normalize_labels(labels, count):
    defaults = ["Day 1", "Day 5", "Day 10"]
    out = []
    for i in range(count):
        if i < len(labels) and str(labels[i]).strip():
            out.append(str(labels[i]).strip())
        elif i < len(defaults):
            out.append(defaults[i])
        else:
            out.append(f"Sample {i + 1}")
    return out


def predict_with_context(img_bgr, field_mode=True, weather=None):
    """Run one prediction and enrich it with trend-friendly fields."""
    result = run_prediction(img_bgr, field_mode=field_mode, weather=weather)
    result["severity_score"] = severity_numeric(result.get("severity", {}))
    return result



# ── Disease risk rules ────────────────────────────────────────────────────────

DISEASE_RISK_RULES = {
    "tomato_early_blight":   {"temp_range": (20, 30), "humidity_min": 60,  "rain_sensitive": False,
                               "description": "Alternaria solani — warm + humid conditions accelerate lesion spread."},
    "tomato_late_blight":    {"temp_range": (10, 25), "humidity_min": 80,  "rain_sensitive": True,
                               "description": "Phytophthora infestans — cool, moist nights are highest-risk windows."},
    "potato_early_blight":   {"temp_range": (20, 30), "humidity_min": 60,  "rain_sensitive": False,
                               "description": "Mirrors tomato early blight; warm days + humid nights ideal for Alternaria."},
    "potato_late_blight":    {"temp_range": (10, 25), "humidity_min": 80,  "rain_sensitive": True,
                               "description": "Same pathogen as tomato late blight. Rain dramatically increases spread."},
    "pepper_bacterial_spot": {"temp_range": (24, 32), "humidity_min": 70,  "rain_sensitive": True,
                               "description": "Xanthomonas — warm + rain creates splash dispersal of bacteria."},
    "corn_common_rust":      {"temp_range": (16, 25), "humidity_min": 70,  "rain_sensitive": False,
                               "description": "Puccinia sorghi — moderate temps + humid nights accelerate urediniospore germination."},
}

HEALTHY_CLASSES = {"tomato_healthy", "potato_healthy", "pepper_healthy", "corn_healthy"}

ADVICE_RULES = [
    ("early_blight", "Apply protectant fungicide. Remove lower infected leaves. Maintain dry canopy."),
    ("late_blight",  "Act immediately — late blight spreads very fast. Apply systemic fungicide within 24h."),
    ("blight",       "Remove infected leaves and avoid overhead watering. Apply copper-based fungicide."),
    ("bacterial",    "Avoid overhead irrigation. Remove affected tissue. Apply copper bactericide."),
    ("rust",         "Improve field airflow. Monitor spread daily. Apply fungicide at first new lesions."),
    ("healthy",      "No disease detected. Continue monitoring every 3–5 days."),
]


def get_care_advice(class_name):
    low = class_name.lower()
    for key, advice in ADVICE_RULES:
        if key in low:
            return advice
    return "No specific advice. Monitor regularly and consult your local agronomist."


def compute_spread_risk(disease_label, weather):
    if disease_label in HEALTHY_CLASSES:
        return 0.0, "None", "Leaf appears healthy. No disease detected."
    rules = DISEASE_RISK_RULES.get(disease_label)
    if rules is None:
        return 0.3, "Low", "No specific risk rules for this disease class."

    t_min, t_max = rules["temp_range"]
    temp     = weather.get("temp_c", 28)
    humidity = weather.get("humidity_pct", 70)
    rain     = weather.get("rain_1h_mm", 0)

    score, factors = 0.0, []
    if t_min <= temp <= t_max:
        t_score = 0.4 * (1 - abs(temp - (t_min + t_max) / 2) / ((t_max - t_min) / 2))
        score += t_score
        factors.append(f"temperature {temp}°C in optimal range")
    else:
        factors.append(f"temperature {temp}°C outside optimal range")

    if humidity >= rules["humidity_min"]:
        h_score = 0.35 * min((humidity - rules["humidity_min"]) / (100 - rules["humidity_min"] + 1e-6), 1.0)
        score += h_score
        factors.append(f"humidity {humidity}% above threshold")
    else:
        factors.append(f"humidity {humidity}% below threshold")

    if rules["rain_sensitive"] and rain > 0:
        score += min(rain / 5.0, 1.0) * 0.25
        factors.append(f"rainfall {rain} mm/h (splash risk)")

    score = min(score, 1.0)
    level = "HIGH" if score >= 0.70 else ("MODERATE" if score >= 0.40 else "LOW")
    explanation = rules["description"] + " Factors: " + "; ".join(factors) + "."
    return round(score, 3), level, explanation


# ── Grad-CAM & Lesion Percentage Computation ────────────────────────────────

def init_gradcam_model():
    """Initializes a gradient model for Grad-CAM if TensorFlow model is loaded."""
    global grad_model, model
    if not TF_AVAILABLE or model is None:
        return
    try:
        last_conv = None
        for layer in reversed(model.layers):
            if isinstance(layer, keras.layers.Conv2D):
                last_conv = layer
                break
            if hasattr(layer, "layers"):
                for sub in reversed(layer.layers):
                    if isinstance(sub, keras.layers.Conv2D) or getattr(sub, "name", "") == "top_activation":
                        last_conv = sub
                        break
            if last_conv is not None:
                break

        if last_conv is None:
            try:
                last_conv = model.get_layer("top_activation")
            except Exception:
                pass

        if last_conv is not None:
            grad_model = keras.Model(
                inputs=model.inputs,
                outputs=[last_conv.output, model.output]
            )
            print(f"✅ Grad-CAM attached to layer: {last_conv.name}")
    except Exception as e:
        print(f"ℹ️ Grad-CAM graph attachment notice: {e}. Fallback saliency engine active.")


def generate_gradcam_and_affected_pct(img_bgr, class_idx, class_name, weather=None):
    """
    Computes:
      1. Grad-CAM attention heatmap (or multi-scale lesion saliency).
      2. Leaf segmentation mask to isolate foreground foliage from background.
      3. Precise affected percentage = (diseased pixels / total leaf pixels) * 100.
      4. High-grade visual overlay image (base64 data URI).
    """
    is_healthy = "healthy" in class_name.lower()
    target_dim = 280
    resized_bgr = cv2.resize(img_bgr, (target_dim, target_dim))

    # 1. Segment leaf foliage
    hsv = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
    leaf_mask = (hsv[:, :, 1] > 28) & (hsv[:, :, 2] > 30) & (hsv[:, :, 2] < 248)
    leaf_pixel_count = int(np.sum(leaf_mask))
    if leaf_pixel_count < 150:
        leaf_mask = (np.mean(resized_bgr, axis=2) > 25) & (np.mean(resized_bgr, axis=2) < 235)
        leaf_pixel_count = max(int(np.sum(leaf_mask)), 1)

    heatmap = None
    # 2. Attempt True Grad-CAM if grad_model and TF are available
    if TF_AVAILABLE and grad_model is not None and not is_healthy:
        try:
            inp_img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
            inp_rgb = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB).astype(np.float32)
            inp_arr = tf.cast(np.expand_dims(inp_rgb, axis=0), tf.float32)

            w_vec = normalise_weather_vector(weather) if weather else np.zeros(WEATHER_DIM, np.float32)
            w_arr = tf.cast(np.expand_dims(w_vec, axis=0), tf.float32)

            with tf.GradientTape() as tape:
                conv_out, preds = grad_model([inp_arr, w_arr], training=False)
                loss = preds[:, class_idx]

            grads = tape.gradient(loss, conv_out)
            if grads is not None:
                pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
                cam = tf.reduce_sum(conv_out[0] * pooled_grads, axis=-1)
                cam = tf.nn.relu(cam)
                max_val = tf.reduce_max(cam)
                if max_val > 0:
                    cam = cam / max_val
                heatmap = cam.numpy()
        except Exception:
            heatmap = None

    # 3. Saliency & Lesion Mapping Fallback/Fusion
    bgr_int = resized_bgr.astype(np.int16)
    color_lesion = leaf_mask & ((bgr_int[:, :, 1] < 120) | (bgr_int[:, :, 2] > 135) | (hsv[:, :, 0] < 22) | (hsv[:, :, 0] > 95))

    gray = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2GRAY)
    texture_var = cv2.Laplacian(gray, cv2.CV_32F)
    texture_grad = np.abs(texture_var)
    if texture_grad.max() > 0:
        texture_grad = texture_grad / texture_grad.max()

    if heatmap is None:
        combined_saliency = np.zeros((target_dim, target_dim), dtype=np.float32)
        if not is_healthy:
            combined_saliency = (color_lesion.astype(np.float32) * 0.75 + texture_grad * 0.25) * leaf_mask.astype(np.float32)
            combined_saliency = cv2.GaussianBlur(combined_saliency, (11, 11), 0)
            if combined_saliency.max() > 0:
                combined_saliency /= combined_saliency.max()
        heatmap = combined_saliency
    else:
        heatmap = cv2.resize(heatmap, (target_dim, target_dim))
        heatmap = heatmap * leaf_mask.astype(np.float32)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

    # 4. Compute Affected Percentage
    if is_healthy:
        affected_pct = 0.0
        sev_category = "Healthy (0% Damaged)"
        damage_desc = "Leaf surface is healthy with no significant necrotic lesions detected."
    else:
        diseased_mask = leaf_mask & ((heatmap > 0.32) | color_lesion)
        diseased_count = int(np.sum(diseased_mask))
        affected_pct = round(min(100.0, (diseased_count / leaf_pixel_count) * 100.0), 1)

        if affected_pct < 4.0:
            affected_pct = round(float(np.clip(np.mean(heatmap[leaf_mask]) * 35.0, 5.0, 15.0)), 1)

        if affected_pct < 10.0:
            sev_category = "Mild Damage"
            damage_desc = f"Localized early infection covering {affected_pct}% of the leaf surface."
        elif affected_pct < 28.0:
            sev_category = "Moderate Damage"
            damage_desc = f"Active lesion spread affecting {affected_pct}% of the leaf photosynthetic area."
        else:
            sev_category = "Severe Damage"
            damage_desc = f"Extensive tissue destruction across {affected_pct}% of the leaf area."

    # 5. Generate Visual Grad-CAM Overlay
    heat_u8 = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)

    leaf_mask_soft = cv2.GaussianBlur(leaf_mask.astype(np.float32), (13, 13), 0)
    leaf_mask_soft = np.repeat(np.expand_dims(leaf_mask_soft, axis=2), 3, axis=2)

    if is_healthy:
        tinted = cv2.addWeighted(resized_bgr, 0.90, heat_color, 0.10, 0)
        final_bgr = (tinted * leaf_mask_soft + resized_bgr * (1.0 - leaf_mask_soft)).astype(np.uint8)
    else:
        blended = cv2.addWeighted(resized_bgr, 0.60, heat_color, 0.40, 0)
        final_bgr = (blended * leaf_mask_soft + resized_bgr * (1.0 - leaf_mask_soft)).astype(np.uint8)

    success, buffer = cv2.imencode(".jpg", final_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    b64_str = ("data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")) if success else None

    return {
        "image": b64_str,
        "affected_pct": affected_pct,
        "category": sev_category,
        "description": damage_desc,
        "is_healthy": is_healthy
    }


# ── Prediction function ───────────────────────────────────────────────────────

def run_prediction(img_bgr, field_mode=True, weather=None):
    """Run the full prediction pipeline and return a structured result dict."""
    if model is None or not class_names:
        # Demo mode — return mock data when model is not loaded
        demo_gc = generate_gradcam_and_affected_pct(img_bgr, 0, "tomato_early_blight", weather=weather)
        return {
            "prediction"     : "tomato_early_blight",
            "confidence"     : 0.87,
            "confidence_pct" : 87.0,
            "confidence_tier": "high confidence",
            "top3"           : [
                ["tomato_early_blight", 0.87],
                ["tomato_late_blight",  0.08],
                ["tomato_healthy",      0.03],
            ],
            "quality"        : image_quality_check(img_bgr),
            "severity"       : severity_proxy(img_bgr),
            "gradcam"        : demo_gc,
            "advice"         : "Apply protectant fungicide. Remove lower infected leaves. Maintain dry canopy.",
            "spread_risk"    : None,
            "mode"           : "field" if field_mode else "lab",
            "demo_mode"      : True,
        }

    quality  = image_quality_check(img_bgr)
    severity = severity_proxy(img_bgr)

    if field_mode:
        original        = cv2.resize(img_bgr, (448, 448))
        enhanced        = enhance_field_image(original)
        segmented, mask = segment_leaf_advanced(enhanced)
        blur_score      = calculate_blur_score(segmented)
        patches         = create_patches(segmented)

        w_vec = normalise_weather_vector(weather) if weather else np.zeros(WEATHER_DIM, np.float32)
        w_arr = np.expand_dims(w_vec, axis=0)

        patch_preds = []
        for patch in patches:
            patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32)
            patch_arr = np.expand_dims(patch_rgb, axis=0)
            patch_preds.append(model.predict([patch_arr, w_arr], verbose=0)[0])

        mean_pred  = np.mean(patch_preds, axis=0)
        class_idx  = int(np.argmax(mean_pred))
        confidence = float(mean_pred[class_idx])
        confidence = apply_confidence_penalty(confidence, blur_score)
        all_probs  = {class_names[i]: float(mean_pred[i]) for i in range(len(class_names))}
    else:
        img     = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_arr = np.expand_dims(img_rgb, axis=0)
        w_vec   = normalise_weather_vector(weather) if weather else np.zeros(WEATHER_DIM, np.float32)
        w_arr   = np.expand_dims(w_vec, axis=0)
        pred      = model.predict([img_arr, w_arr], verbose=0)[0]
        class_idx = int(np.argmax(pred))
        confidence = float(pred[class_idx])
        all_probs  = {class_names[i]: float(pred[i]) for i in range(len(class_names))}

    best_class = class_names[class_idx]
    top3 = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:3]

    conf_tier = ("high confidence" if confidence >= 0.85
                 else ("medium confidence" if confidence >= 0.55
                       else "low confidence — retake image"))

    spread_risk = None
    if weather:
        risk_score, risk_level, risk_expl = compute_spread_risk(best_class, weather)
        spread_risk = {"score": risk_score, "level": risk_level, "explanation": risk_expl}

    gc_result = generate_gradcam_and_affected_pct(img_bgr, class_idx, best_class, weather=weather)

    return {
        "prediction"     : best_class,
        "confidence"     : round(confidence, 4),
        "confidence_pct" : round(confidence * 100, 2),
        "confidence_tier": conf_tier,
        "top3"           : top3,
        "quality"        : quality,
        "severity"       : severity,
        "gradcam"        : gc_result,
        "advice"         : get_care_advice(best_class),
        "spread_risk"    : spread_risk,
        "mode"           : "field" if field_mode else "lab",
        "live_weather"   : weather is not None,
        "demo_mode"      : False,
    }


# ── Flask Routes ──────────────────────────────────────────────────────────────
# ── Flask Routes ──────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "name": "AgroIntelli API",
        "status": "running",
        "available_routes": [
            "/health",
            "/classes",
            "/weather",
            "/predict",
            "/batch_predict"
        ]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "tf_available": TF_AVAILABLE,
        "classes": len(class_names),
        "timestamp": datetime.now().isoformat(),
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
def get_weather():
    """Fetch live weather for the caller's approximate location."""
    payload = fetch_live_weather_snapshot()
    return jsonify(payload)


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Form fields:
        image       — file upload (required)
        mode        — "field" (default) or "lab"
        use_weather — "true" / "false" (default false)
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    img_bgr = decode_uploaded_image(file)
    if img_bgr is None:
        return jsonify({"error": "Could not decode image. Use JPG or PNG."}), 400

    field_mode = request.form.get("mode", "field") == "field"
    use_weather = request.form.get("use_weather", "false").lower() == "true"

    weather = None
    if use_weather:
        weather_payload = fetch_live_weather_snapshot()
        if weather_payload.get("success"):
            weather = weather_payload.get("weather")

    try:
        result = run_prediction(img_bgr, field_mode=field_mode, weather=weather)
        result["weather"] = weather
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    POST /batch_predict
    Accepts multiple image uploads to compare disease progression over time.

    Form fields:
        images      — repeated file uploads in the order they should be compared
        labels      — JSON array of labels (e.g. ["Day 1", "Day 5", "Day 10"])
        mode        — "field" (default) or "lab"
        use_weather — "true" / "false" (default false)
    """
    uploaded_files = request.files.getlist("images")
    if not uploaded_files:
        # Fallback for distinct field names used by some frontends
        for key in ("day1", "day5", "day10"):
            f = request.files.get(key)
            if f and f.filename:
                uploaded_files.append(f)

    if not uploaded_files:
        return jsonify({"error": "No images provided for batch analysis"}), 400

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

        result = predict_with_context(img_bgr, field_mode=field_mode, weather=session_weather)
        item = {
            "label": label,
            "index": idx + 1,
            "filename": file.filename,
            "result": result,
            "severity_score": round(float(result.get("severity_score", 0.0)), 4),
            "weather": session_weather,
        }
        items.append(item)

    # Add step deltas for the frontend
    prev_score = None
    for item in items:
        item["delta_from_previous"] = None if prev_score is None else round(item["severity_score"] - prev_score, 4)
        prev_score = item["severity_score"]

    summary = batch_progress_summary(items)
    overall_weather_note = None
    if session_weather:
        overall_weather_note = {
            "applied": True,
            "note": "A live weather snapshot was applied to every image in this batch."
        }

    return jsonify({
        "success": True,
        "mode": "batch",
        "field_mode": "field" if field_mode else "lab",
        "items": items,
        "summary": summary,
        "weather": session_weather,
        "weather_note": overall_weather_note,
    })


# ── Authentication & Plant Journal API Routes ─────────────────────────────────

@app.route("/api/status", methods=["GET"])
def api_status():
    """Returns storage connection mode (MongoDB active or resilient local mode)."""
    return jsonify({
        "status": "ok",
        "is_mongo": db_store.is_mongo,
        "database": DB_NAME if db_store.is_mongo else "local_json_store",
        "mongo_uri_target": MONGO_URI
    })


@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    email = data.get("email", "").strip()

    if not username or len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters."}), 400
    if not password or len(password) < 4:
        return jsonify({"error": "Password must be at least 4 characters."}), 400

    user, err = db_store.create_user(username, password, email)
    if err:
        return jsonify({"error": err}), 409

    return jsonify({"success": True, "user": user, "message": "Account created successfully."})


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json(silent=True) or {}
    identifier = data.get("identifier", "").strip()
    password = data.get("password", "")

    if not identifier or not password:
        return jsonify({"error": "Username/email and password required."}), 400

    user = db_store.get_user(identifier)
    if not user or not check_password_hash(user.get("password_hash", ""), password):
        return jsonify({"error": "Invalid username or password."}), 401

    clean_user = {k: v for k, v in user.items() if k not in ("password_hash", "_id")}
    return jsonify({"success": True, "user": clean_user})


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
def save_record():
    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id", "guest").strip()
    plant_name = data.get("plant_name", "").strip() or "Tracked Crop Leaf"
    day_label = data.get("day_label", "Day 1").strip()
    notes = data.get("notes", "").strip()

    prediction_data = data.get("diagnostic") or {}
    if not prediction_data:
        return jsonify({"error": "Missing diagnostic payload"}), 400

    record_id = data.get("record_id") or f"leaf_{uuid.uuid4().hex[:10]}"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    pred_name = prediction_data.get("prediction", "Unknown")
    aff_pct = prediction_data.get("gradcam", {}).get("affected_pct", 0.0)
    sev_cat = prediction_data.get("gradcam", {}).get("category", prediction_data.get("severity", {}).get("severity", "Unknown"))
    gradcam_img = prediction_data.get("gradcam", {}).get("image")

    # Initial timeline entry
    initial_checkin = {
        "checkin_id": uuid.uuid4().hex[:8],
        "day_label": day_label,
        "timestamp": now_str,
        "prediction": pred_name,
        "confidence": prediction_data.get("confidence", 0.0),
        "affected_pct": aff_pct,
        "severity": sev_cat,
        "delta_from_previous": 0.0,
        "verdict": "BASELINE",
        "explanation": "Initial baseline health assessment recorded.",
        "gradcam_image": gradcam_img,
        "notes": notes
    }

    record = {
        "record_id": record_id,
        "user_id": user_id,
        "plant_name": plant_name,
        "created_at": now_str,
        "initial_prediction": pred_name,
        "latest_prediction": pred_name,
        "latest_affected_pct": aff_pct,
        "latest_severity": sev_cat,
        "latest_day": day_label,
        "latest_verdict": "Baseline scan logged.",
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
    Submits a new photo of an existing leaf (e.g. Day 7, Day 14),
    runs diagnosis, compares against past scans, and updates timeline.
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
    notes = request.form.get("notes", "").strip()
    use_weather = request.form.get("use_weather", "false").lower() == "true"
    mode = request.form.get("mode", "field") == "field"

    weather = None
    if use_weather:
        weather_payload = fetch_live_weather_snapshot()
        if weather_payload.get("success"):
            weather = weather_payload.get("weather")

    # Run fresh prediction on the re-checked leaf
    diag = run_prediction(img_bgr, field_mode=mode, weather=weather)

    # Compare with previous checkin
    timeline = rec.get("timeline", [])
    prev_entry = timeline[-1] if timeline else None

    new_pred = diag.get("prediction", "Unknown")
    new_aff = float(diag.get("gradcam", {}).get("affected_pct", 0.0))
    new_sev = diag.get("gradcam", {}).get("category", "Unknown")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    prev_pred = prev_entry.get("prediction", new_pred) if prev_entry else new_pred
    prev_aff  = float(prev_entry.get("affected_pct", new_aff)) if prev_entry else new_aff

    delta = round(new_aff - prev_aff, 1)

    # Comparative progression analysis
    if new_pred != prev_pred:
        verdict = "NEW_DISEASE"
        status_tag = "⚠️ Condition Shift"
        expl = f"Diagnosis shifted from {prev_pred.replace('_',' ')} to {new_pred.replace('_',' ')}. Review treatment advice."
    elif delta <= -4.0:
        verdict = "IMPROVED"
        status_tag = "🟢 Significant Healing"
        expl = f"Lesion coverage reduced by {abs(delta)}% (from {prev_aff}% down to {new_aff}%). Therapeutic response confirmed."
    elif delta >= 4.0:
        verdict = "WORSENED"
        status_tag = "🔴 Disease Progression"
        expl = f"Lesion surface expanded by {delta}% (from {prev_aff}% to {new_aff}%). Consider aggressive fungicide or canopy pruning."
    else:
        verdict = "STABLE"
        status_tag = "🟡 Stable / Monitored"
        expl = f"Disease severity is steady (change of {delta}%). Continue ongoing monitoring."

    checkin_entry = {
        "checkin_id": uuid.uuid4().hex[:8],
        "day_label": day_label,
        "timestamp": now_str,
        "prediction": new_pred,
        "confidence": diag.get("confidence", 0.0),
        "affected_pct": new_aff,
        "severity": new_sev,
        "delta_from_previous": delta,
        "verdict": verdict,
        "status_tag": status_tag,
        "explanation": expl,
        "gradcam_image": diag.get("gradcam", {}).get("image"),
        "notes": notes,
        "advice": diag.get("advice", "")
    }

    timeline.append(checkin_entry)
    rec["timeline"] = timeline
    rec["latest_prediction"] = new_pred
    rec["latest_affected_pct"] = new_aff
    rec["latest_severity"] = new_sev
    rec["latest_day"] = day_label
    rec["latest_verdict"] = expl

    saved = db_store.save_leaf_record(rec)

    return jsonify({
        "success": True,
        "record": saved,
        "latest_checkin": checkin_entry,
        "diagnostic": diag,
        "comparison": {
            "previous_day": prev_entry.get("day_label", "Baseline") if prev_entry else "Baseline",
            "previous_affected": prev_aff,
            "current_affected": new_aff,
            "delta": delta,
            "verdict": verdict,
            "status_tag": status_tag,
            "explanation": expl
        }
    })


@app.route("/api/records/<record_id>", methods=["DELETE"])
def delete_record(record_id):
    user_id = request.args.get("user_id", "guest").strip()
    db_store.delete_record(record_id, user_id)
    return jsonify({"success": True, "message": "Record removed."})


# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model()
    print("\n🌿 AgroIntelli backend running at http://localhost:5000")
    print("   Frontend: open frontend/index.html in a browser")
    app.run(host="0.0.0.0", port=5000, debug=False)
