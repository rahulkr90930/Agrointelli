"""Streamlit entry point for AgroIntelli's diagnosis and plant journal."""

from __future__ import annotations

import base64
import os
import uuid
from datetime import date, datetime

import cv2
import numpy as np
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from werkzeug.security import check_password_hash

st.set_page_config(
    page_title="AgroIntelli | Plant Health",
    page_icon="🌿",
    layout="wide",
)


def configure_streamlit_secrets() -> None:
    try:
        secrets = st.secrets
        configured = {name: secrets.get(name) for name in ("MONGO_URI", "MONGO_DB_NAME", "OWM_API_KEY")}
    except StreamlitSecretNotFoundError:
        return
    for name, value in configured.items():
        if value:
            os.environ[name] = str(value)


configure_streamlit_secrets()


@st.cache_resource(show_spinner="Loading AgroIntelli's plant disease model...")
def load_services():
    from backend.database import db_store
    from backend import inference
    from backend.weather import evaluate_weather_progression, fetch_live_weather_snapshot

    inference.load_model()
    return inference, db_store, evaluate_weather_progression, fetch_live_weather_snapshot


inference, db_store, evaluate_weather_progression, fetch_live_weather_snapshot = load_services()

if "streamlit_user_id" not in st.session_state:
    st.session_state.streamlit_user_id = f"streamlit_{uuid.uuid4().hex}"


def active_user_id() -> str:
    user = st.session_state.get("streamlit_user")
    return user["user_id"] if user else st.session_state.streamlit_user_id


def render_account_controls() -> None:
    with st.sidebar.expander("Journal account", expanded=False):
        user = st.session_state.get("streamlit_user")
        if user:
            st.write(f"Signed in as **{user['username']}**")
            if st.button("Sign out", key="streamlit-sign-out"):
                del st.session_state.streamlit_user
                st.rerun()
        else:
            mode = st.radio(
                "Journal access",
                ["Guest session", "Sign in", "Create account"],
                key="streamlit-account-mode",
            )
            if mode == "Sign in":
                with st.form("streamlit-sign-in"):
                    username = st.text_input("Username or email")
                    password = st.text_input("Password", type="password")
                    submitted = st.form_submit_button("Sign in")
                if submitted:
                    profile = db_store.get_user(username)
                    password_hash = profile.get("password_hash") if profile else None
                    if profile and password_hash and check_password_hash(password_hash, password):
                        st.session_state.streamlit_user = {
                            "user_id": profile["user_id"],
                            "username": profile["username"],
                        }
                        st.rerun()
                    st.error("The username/email or password was not recognized.")
            elif mode == "Create account":
                with st.form("streamlit-create-account"):
                    username = st.text_input("Username")
                    email = st.text_input("Email (optional)")
                    password = st.text_input("Password (at least 8 characters)", type="password")
                    submitted = st.form_submit_button("Create account")
                if submitted:
                    if not username.strip() or len(password) < 8:
                        st.error("Enter a username and a password of at least 8 characters.")
                    else:
                        profile, error = db_store.create_user(username.strip(), password, email.strip())
                        if error:
                            st.error(error)
                        elif not profile:
                            st.error("The account could not be created.")
                        else:
                            st.session_state.streamlit_user = {
                                "user_id": profile["user_id"],
                                "username": profile["username"],
                            }
                            st.rerun()
        if not db_store.is_mongo:
            st.warning("MongoDB is unavailable; account and journal data use local storage and may not persist on Streamlit Cloud.")


def gradcam_bytes(image_data: str | None) -> bytes | None:
    if not image_data or "," not in image_data:
        return None
    return base64.b64decode(image_data.split(",", 1)[1])


def decode_image(image_bytes: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)


def display_diagnostic(result: dict, original_image: bytes) -> None:
    gradcam = result.get("gradcam") or {}
    columns = st.columns([1.2, 1, 1])
    with columns[0]:
        st.subheader(str(result.get("prediction", "Diagnosis")).replace("_", " ").title())
        st.metric("Crop", result.get("plant", "Not identified"))
        st.metric("Confidence", f"{float(result.get('confidence_pct') or 0):.1f}%")
    with columns[1]:
        st.metric("Affected leaf area", f"{float(gradcam.get('affected_pct') or 0):.1f}%")
        st.caption(gradcam.get("category") or "Damage classification unavailable")
        st.image(original_image, caption="Uploaded leaf", use_container_width=True)
    with columns[2]:
        overlay = gradcam_bytes(gradcam.get("image"))
        if overlay:
            st.image(overlay, caption="Grad-CAM attention heatmap", use_container_width=True)
        else:
            st.info("Grad-CAM image is unavailable for this diagnosis.")

    top_classes = result.get("top3") or []
    if top_classes:
        st.markdown("**Other likely diagnoses**")
        for label, probability in top_classes:
            st.write(f"{str(label).replace('_', ' ').title()} — {float(probability) * 100:.1f}%")

    quality = result.get("quality") or {}
    if quality:
        with st.expander("Image quality details"):
            st.json(quality)

    risk = result.get("spread_risk") or {}
    if risk:
        st.markdown(f"**Weather-related spread risk:** {risk.get('level', 'Unknown')}")
        if risk.get("explanation"):
            st.write(risk["explanation"])
    if result.get("advice"):
        st.markdown("**Care guidance**")
        st.write(result["advice"])
    if result.get("weather"):
        st.markdown("**Weather used for this diagnosis**")
        st.json(result["weather"])


def build_initial_record(result: dict, user_id: str, notes: str = "") -> dict:
    now = datetime.now()
    gradcam = result.get("gradcam") or {}
    diagnosis = result.get("prediction", "Unknown")
    timestamp = now.strftime("%Y-%m-%d %H:%M")
    entry = {
        "checkin_id": uuid.uuid4().hex[:8],
        "day_number": 1,
        "day_label": "Day 1",
        "date": now.strftime("%Y-%m-%d"),
        "timestamp": timestamp,
        "prediction": diagnosis,
        "confidence": result.get("confidence", 0.0),
        "confidence_pct": result.get("confidence_pct", 0.0),
        "affected_pct": gradcam.get("affected_pct", 0.0),
        "severity": gradcam.get("category", "Unknown"),
        "delta_from_previous": 0.0,
        "rate_per_day": 0.0,
        "verdict": "BASELINE",
        "status_tag": "Baseline diagnosis",
        "explanation": "Initial leaf health assessment.",
        "gradcam_image": gradcam.get("image"),
        "weather": result.get("weather"),
        "notes": notes,
        "advice": result.get("advice", ""),
    }
    return {
        "record_id": f"leaf_{uuid.uuid4().hex[:10]}",
        "user_id": user_id,
        "plant_name": result.get("plant", "Tracked Crop"),
        "variety": "",
        "created_at": timestamp,
        "status": diagnosis,
        "initial_prediction": diagnosis,
        "latest_prediction": diagnosis,
        "latest_affected_pct": entry["affected_pct"],
        "latest_severity": entry["severity"],
        "latest_day": "Day 1",
        "latest_day_number": 1,
        "latest_verdict": entry["explanation"],
        "notes": notes,
        "thumbnail": entry["gradcam_image"],
        "timeline": [entry],
    }


def parse_entry_date(entry: dict) -> date | None:
    value = entry.get("date") or str(entry.get("timestamp") or "")[:10]
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


st.title("🌿 AgroIntelli")
st.caption("Crop diagnosis, explainable Grad-CAM, weather context, and a private plant scan journal.")

with st.sidebar:
    st.header("Diagnosis settings")
    crops = sorted({
        class_name.split("_", 1)[0]
        for class_name in inference.class_names
        if class_name != "background_without_leaves"
    })
    crop_choice = st.selectbox("Crop", ["Auto-detect"] + [crop.title() for crop in crops])
    use_weather = st.checkbox("Include current weather", value=True)
    weather_city = st.text_input("City (optional)", placeholder="Auto-detect if blank")
    page = st.radio("Open", ["Diagnose a leaf", "My plant journal"])
    if inference.model is None:
        st.warning("The trained model is unavailable; the backend may be using its demo fallback.")

render_account_controls()

if page == "Diagnose a leaf":
    st.subheader("Leaf diagnosis")
    uploaded = st.file_uploader("Upload or capture a leaf photo", type=["jpg", "jpeg", "png", "webp"])
    notes = st.text_area("Field notes (optional)", placeholder="Symptoms, treatment, or garden conditions")

    if st.button("Run diagnosis", type="primary", disabled=uploaded is None):
        st.session_state.pop("latest_diagnosis", None)
        st.session_state.pop("latest_diagnosis_image", None)
        if uploaded is None:
            st.error("Choose a leaf photo before running the diagnosis.")
        else:
            image_bytes = uploaded.getvalue()
            image = decode_image(image_bytes)
            if image is None:
                st.error("The uploaded file could not be decoded as an image.")
            else:
                weather = None
                if use_weather:
                    weather_result = fetch_live_weather_snapshot(city=weather_city.strip() or None)
                    if weather_result.get("success"):
                        weather = weather_result.get("weather")
                    else:
                        st.warning("Live weather could not be retrieved; the diagnosis will use default weather inputs.")
                try:
                    diagnosis = inference.run_prediction(
                        image,
                        field_mode=True,
                        weather=weather,
                        plant=None if crop_choice == "Auto-detect" else crop_choice.lower(),
                    )
                    diagnosis["weather"] = weather
                    st.session_state.latest_diagnosis = diagnosis
                    st.session_state.latest_diagnosis_image = image_bytes
                    st.session_state.latest_diagnosis_notes = notes
                except (ValueError, RuntimeError) as error:
                    st.error(f"Diagnosis could not be completed: {error}")

    diagnosis = st.session_state.get("latest_diagnosis")
    if diagnosis:
        st.divider()
        display_diagnostic(diagnosis, st.session_state.latest_diagnosis_image)
        if st.button("Save diagnosis to my journal", type="secondary"):
            record = build_initial_record(
                diagnosis,
                active_user_id(),
                st.session_state.get("latest_diagnosis_notes", ""),
            )
            db_store.save_leaf_record(record)
            st.session_state.latest_saved_record_id = record["record_id"]
            st.success("Diagnosis saved to your plant journal.")

elif page == "My plant journal":
    st.subheader("My plant journal")
    user_id = active_user_id()
    records = db_store.get_user_records(user_id)
    if not records:
        st.info("No saved scans for this account yet. Run a diagnosis and save it to start a journal.")

    for record in records:
        timeline = record.get("timeline") or []
        with st.expander(f"{record.get('plant_name', 'Plant')} — {record.get('status', 'Unknown').replace('_', ' ')}"):
            st.write(record.get("variety") or "Tracked plant")
            st.caption(f"Scans recorded: {len(timeline)}")
            for entry in timeline:
                st.markdown(
                    f"**{entry.get('day_label', 'Scan')} · {entry.get('date', '')}**  \n"
                    f"{str(entry.get('prediction', 'Unknown')).replace('_', ' ').title()} · "
                    f"{float(entry.get('confidence_pct') or 0):.1f}% confidence · "
                    f"{float(entry.get('affected_pct') or 0):.1f}% affected"
                )
                if entry.get("explanation"):
                    st.caption(entry["explanation"])
                overlay = gradcam_bytes(entry.get("gradcam_image"))
                if overlay:
                    st.image(overlay, caption=f"{entry.get('day_label', 'Scan')} Grad-CAM", width=420)

            st.markdown("**Add a follow-up scan**")
            followup_image = st.file_uploader(
                "Follow-up leaf photo",
                type=["jpg", "jpeg", "png", "webp"],
                key=f"followup-file-{record.get('record_id')}",
            )
            baseline_date = parse_entry_date(timeline[0]) if timeline else None
            latest_date = parse_entry_date(timeline[-1]) if timeline else baseline_date
            scan_date = st.date_input(
                "Scan date",
                value=date.today(),
                key=f"followup-date-{record.get('record_id')}",
            )
            day_number = max(1, (scan_date - baseline_date).days + 1) if baseline_date else len(timeline) + 1
            st.caption(f"Timeline day: Day {day_number}")
            followup_notes = st.text_input(
                "Follow-up notes (optional)",
                key=f"followup-notes-{record.get('record_id')}",
            )
            if st.button(
                "Analyze and add to timeline",
                key=f"followup-submit-{record.get('record_id')}",
                disabled=followup_image is None,
            ):
                if followup_image is None:
                    st.error("Choose a leaf photo before adding a follow-up scan.")
                elif latest_date and scan_date < latest_date:
                    st.error("Choose a scan date on or after the most recent timeline entry.")
                else:
                    image_bytes = followup_image.getvalue()
                    image = decode_image(image_bytes)
                    if image is None:
                        st.error("The uploaded file could not be decoded as an image.")
                    else:
                        weather = None
                        if use_weather:
                            weather_result = fetch_live_weather_snapshot(city=weather_city.strip() or None)
                            if weather_result.get("success"):
                                weather = weather_result.get("weather")
                        baseline_prediction = record.get("initial_prediction") or (
                            timeline[0].get("prediction") if timeline else ""
                        )
                        crop_filter = str(baseline_prediction).split("_", 1)[0].lower()
                        try:
                            diagnosis = inference.run_prediction(
                                image,
                                field_mode=True,
                                weather=weather,
                                plant=crop_filter,
                            )
                            previous = timeline[-1] if timeline else {}
                            current_area = float((diagnosis.get("gradcam") or {}).get("affected_pct") or 0.0)
                            previous_area = float(previous.get("affected_pct") or 0.0)
                            previous_date = latest_date or scan_date
                            days_elapsed = max((scan_date - previous_date).days, 1)
                            progression = evaluate_weather_progression(
                                prev_weather=previous.get("weather"),
                                curr_weather=weather,
                                days_elapsed=days_elapsed,
                                prev_pred=previous.get("prediction", diagnosis["prediction"]),
                                curr_pred=diagnosis["prediction"],
                                prev_aff=previous_area,
                                curr_aff=current_area,
                                curr_day_label=f"Day {day_number}",
                            )
                            gradcam = diagnosis.get("gradcam") or {}
                            entry = {
                                "checkin_id": uuid.uuid4().hex[:8],
                                "day_number": day_number,
                                "day_label": f"Day {day_number}",
                                "days_elapsed": days_elapsed,
                                "date": scan_date.isoformat(),
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                "prediction": diagnosis["prediction"],
                                "confidence": diagnosis.get("confidence", 0.0),
                                "confidence_pct": diagnosis.get("confidence_pct", 0.0),
                                "affected_pct": current_area,
                                "severity": gradcam.get("category", "Unknown"),
                                "delta_from_previous": progression["delta"],
                                "rate_per_day": progression["rate_per_day"],
                                "verdict": progression["verdict"],
                                "status_tag": progression["status_tag"],
                                "explanation": progression["explanation"],
                                "gradcam_image": gradcam.get("image"),
                                "weather": weather,
                                "notes": followup_notes,
                                "advice": diagnosis.get("advice", ""),
                            }
                            record["timeline"].append(entry)
                            record["status"] = diagnosis["prediction"]
                            record["latest_prediction"] = diagnosis["prediction"]
                            record["latest_affected_pct"] = current_area
                            record["latest_severity"] = entry["severity"]
                            record["latest_day"] = entry["day_label"]
                            record["latest_day_number"] = day_number
                            record["latest_verdict"] = entry["explanation"]
                            db_store.save_leaf_record(record)
                            st.success(
                                f"{entry['day_label']}: {diagnosis['prediction'].replace('_', ' ').title()} "
                                f"({entry['confidence_pct']:.1f}% confidence)."
                            )
                            overlay = gradcam_bytes(entry.get("gradcam_image"))
                            if overlay:
                                st.image(overlay, caption="Follow-up Grad-CAM heatmap", width=520)
                        except (ValueError, RuntimeError, KeyError) as error:
                            st.error(f"Follow-up diagnosis could not be completed: {error}")

            record_id = record.get("record_id")
            if record_id:
                confirm_key = f"confirm-delete-{record_id}"
                if st.button("Delete plant record", key=f"delete-record-{record_id}"):
                    st.session_state[confirm_key] = True

                if st.session_state.get(confirm_key):
                    st.warning("Delete this plant and its complete scan timeline? This cannot be undone.")
                    confirm_col, cancel_col = st.columns(2)
                    with confirm_col:
                        if st.button("Yes, delete record", type="primary", key=f"confirm-delete-button-{record_id}"):
                            if db_store.delete_record(record_id, user_id=user_id):
                                st.session_state.pop(confirm_key, None)
                                st.success("Plant record and its scan timeline were deleted.")
                                st.rerun()
                            else:
                                st.error("The record could not be deleted. It may already have been removed.")
                    with cancel_col:
                        if st.button("Cancel", key=f"cancel-delete-{record_id}"):
                            st.session_state.pop(confirm_key, None)
                            st.rerun()

st.caption(
    "Streamlit edition: diagnosis, Grad-CAM, weather context, and session-scoped journals. "
    "The existing Flask web app remains unchanged and provides community, account, chatbot, and voice features."
)
