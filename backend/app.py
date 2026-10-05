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
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

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
        available_models, class_names, TF_AVAILABLE, model
    )
    from .batch import batch_progress_summary, normalize_labels
except (ImportError, ValueError):
    from database import db_store
    from weather import fetch_live_weather_snapshot, evaluate_weather_progression, parse_day_num
    from inference import (
        load_model, run_prediction, predict_with_context, decode_uploaded_image,
        available_models, class_names, TF_AVAILABLE, model
    )
    from batch import batch_progress_summary, normalize_labels

app = Flask(__name__)
CORS(app)

# Initialize neural models and label mappings on startup
load_model()


# ── Root & System Endpoints ───────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def home():
    # If opened by a web browser, serve the interactive AgroIntelli web application
    if "text/html" in request.headers.get("Accept", ""):
        frontend_file = Path(__file__).parent.parent / "frontend" / "index.html"
        if frontend_file.exists():
            from flask import send_file
            return send_file(str(frontend_file))
    return jsonify({
        "name": "AgroIntelli Modular API",
        "status": "online",
        "modules": ["database", "weather", "gradcam", "inference", "batch"],
        "models_available": list(available_models.keys()),
        "mongo_connected": db_store.is_mongo
    })


@app.route("/app", methods=["GET"])
def web_app():
    frontend_file = Path(__file__).parent.parent / "frontend" / "index.html"
    from flask import send_file
    return send_file(str(frontend_file))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": len(available_models) > 0 or model is not None,
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
def get_weather():
    """Fetch live weather for caller location via weather module."""
    payload = fetch_live_weather_snapshot()
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
        if not user_records and user_id != "guest":
            user_records = db_store.get_user_records("guest")
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
    use_weather = request.form.get("weather", "false").lower() == "true"
    arch = request.form.get("architecture") or request.args.get("arch")

    weather = None
    if use_weather:
        weather_payload = fetch_live_weather_snapshot()
        if weather_payload.get("success"):
            weather = weather_payload.get("weather")

    try:
        result = run_prediction(img_bgr, field_mode=field_mode, weather=weather, model_choice=arch)
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

        result = predict_with_context(img_bgr, field_mode=field_mode, weather=session_weather, model_choice=arch)
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
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    from werkzeug.security import check_password_hash
    user = db_store.get_user(username)
    if not user or not check_password_hash(user.get("password_hash", ""), password):
        return jsonify({"error": "Invalid username or password."}), 401

    clean = {k: v for k, v in user.items() if k not in ("password_hash", "_id")}
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

    prediction_data = data.get("diagnostic") or {}
    if not prediction_data:
        return jsonify({"error": "Missing diagnostic payload"}), 400

    record_id = data.get("record_id") or f"leaf_{uuid.uuid4().hex[:10]}"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    day_num = parse_day_num(day_label) or 1

    pred_name = prediction_data.get("prediction", "Unknown")
    aff_pct = prediction_data.get("gradcam", {}).get("affected_pct", 0.0)
    sev_cat = prediction_data.get("gradcam", {}).get("category", prediction_data.get("severity", {}).get("severity", "Unknown"))
    gradcam_img = prediction_data.get("gradcam", {}).get("image")
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
        "created_at": now_str,
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

    diag = run_prediction(img_bgr, field_mode=mode, weather=weather, model_choice=arch)

    timeline = rec.get("timeline", [])
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
    rec["latest_affected_pct"] = new_aff
    rec["latest_severity"] = new_sev
    rec["latest_day"] = day_label
    rec["latest_day_number"] = curr_day_num
    rec["latest_verdict"] = expl

    saved = db_store.save_leaf_record(rec)
    prev_img = (prev_entry.get("gradcam_image") or rec.get("thumbnail")) if prev_entry else rec.get("thumbnail")

    return jsonify({
        "success": True,
        "record": saved,
        "latest_checkin": checkin_entry,
        "diagnostic": diag,
        "comparison": {
            "previous_day": prev_entry.get("day_label", "Baseline") if prev_entry else "Baseline",
            "previous_date": prev_entry.get("date") or (prev_entry.get("timestamp", "Initial Scan") if prev_entry else "Initial Scan"),
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


@app.route("/api/records/<record_id>", methods=["DELETE"])
def delete_record(record_id):
    user_id = request.args.get("user_id", "guest").strip()
    db_store.delete_record(record_id, user_id)
    return jsonify({"success": True, "message": "Record removed."})


# ── Startup ───────────────────────────────────────────────────────────────────

def start_server():
    load_model()
    print("\n🌿 AgroIntelli backend running at http://localhost:5000")
    print("   Frontend: open frontend/index.html in a browser")
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    start_server()
