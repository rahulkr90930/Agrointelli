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


# ── Prediction function ───────────────────────────────────────────────────────

def run_prediction(img_bgr, field_mode=True, weather=None):
    """Run the full prediction pipeline and return a structured result dict."""
    if model is None or not class_names:
        # Demo mode — return mock data when model is not loaded
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

    return {
        "prediction"     : best_class,
        "confidence"     : round(confidence, 4),
        "confidence_pct" : round(confidence * 100, 2),
        "confidence_tier": conf_tier,
        "top3"           : top3,
        "quality"        : quality,
        "severity"       : severity,
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


# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_model()
    print("\n🌿 AgroIntelli backend running at http://localhost:5000")
    print("   Frontend: open frontend/index.html in a browser")
    app.run(host="0.0.0.0", port=5000, debug=False)
