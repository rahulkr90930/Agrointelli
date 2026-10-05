"""
AgroIntelli Agronomic Chatbot Engine
Multimodal agricultural intelligence powered by Google Gemini and grounded in
the AgroIntelli disease knowledge base, live weather, and Grad-CAM spatial metrics.
"""

import os
import sys
from pathlib import Path

# Ensure UTF-8 console output on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Load GEMINI_API_KEY from .env if not already in os.environ
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if ENV_FILE.exists():
    with open(ENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k not in os.environ:
                    os.environ[k] = v

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

# Gemini SDK
_gemini_client = None
_model = None

try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_client = genai
        # Initialize with modern Gemini Flash
        _model = genai.GenerativeModel("models/gemini-flash-latest")
        print("🌿 AgroIntelli Chatbot initialized with Gemini Flash API.")
    else:
        print("ℹ️ AgroIntelli Chatbot running in localized rule-grounded mode (GEMINI_API_KEY not configured).")
except Exception as e:
    print(f"⚠ Chatbot initialization notice: {e}")


SYSTEM_PROMPT = """You are AgroIntelli AI (AgroBot), a specialized, scientific, yet practical agricultural pathologist and precision crop advisor.
Your primary role is to assist farmers, agronomists, and greenhouse managers in diagnosing, understanding, and managing plant diseases.

CRITICAL RULES:
1. DOMAIN SPECIFICITY: You are strictly an agronomy and plant pathology specialist. Stay focused on crops, plant diseases, pathogens, chemical fungicides/bactericides, biological/organic solutions, weather-driven spore progression, irrigation, and field management. Do NOT act as a general conversational bot for non-agricultural topics.
2. SYNTHESIZE SCAN TELEMETRY: Always integrate the user's active leaf scan data provided in the prompt:
   - Current Leaf Prediction & Diagnostic Confidence
   - Grad-CAM affected lesion area percentage (% of leaf blade covered)
   - Live Weather Telemetry (temperature, relative humidity %, rainfall mm/h, wind)
   - Canonical Knowledge Profile (pathogen taxonomy, chemical treatments, organic remedies, prevention)
3. GROUNDED AGRONOMIC REASONING:
   - When humidity is high (>=70%) or rain is present, explain how free moisture promotes spore germination and bacterial splash dispersal.
   - For chemical treatments, recommend standard agricultural concentrations (e.g., Mancozeb 2-2.5 g/L, Chlorothalonil 2 ml/L, Copper Oxychloride 3 g/L) and emphasize proper application timing (early morning or late afternoon to prevent phytotoxicity/sun scorch).
   - Offer organic / biological alternatives (e.g., cold-pressed neem oil 5 ml/L with mild surfactant, Trichoderma viride, dilute potassium bicarbonate).
   - If progression data is provided (e.g., Day 1 to Day 5 with spread rate), explicitly evaluate whether the infection is expanding or under control.
4. TONE & STRUCTURE:
   - Structured, concise, and clear with bullet points and bold highlights.
   - Action-oriented: Provide immediate containment steps, spray schedule, and future preventative practices.
"""


def generate_chat_response(user_message, scan_context=None, history=None):
    """
    Generate an agronomic response grounded in scan telemetry and disease knowledge.
    
    Args:
        user_message (str): Farmer's query.
        scan_context (dict): Telemetry from the current diagnosis:
            - prediction: str (e.g. 'tomato_early_blight')
            - confidence_pct: float
            - affected_pct: float (from Grad-CAM)
            - weather: dict (temp_c, humidity_pct, rain_1h_mm, city)
            - timeline_notes: str (e.g. 'Day 1: 12% -> Day 5: 18%')
            - knowledge_record: dict (from disease_knowledge.csv)
        history (list): Optional previous conversation turns [{role: 'user'|'model', text: str}].
    
    Returns:
        dict: {'reply': str, 'grounded': bool, 'model_used': str}
    """
    if not user_message or not user_message.strip():
        return {
            "reply": "Hello! I am AgroBot, your plant health assistant. Scan a leaf or ask me anything about crop diseases, treatment sprays, or weather risks.",
            "grounded": False,
            "model_used": "system"
        }

    scan_context = scan_context or {}
    pred = scan_context.get("prediction", "Unknown / Not Scanned")
    conf = scan_context.get("confidence_pct", 0)
    aff_pct = scan_context.get("affected_pct", None)
    weather = scan_context.get("weather", {})
    knowledge = scan_context.get("knowledge_record", {})
    timeline = scan_context.get("timeline_notes", "")

    # Format telemetry block
    weather_desc = "Not provided"
    if weather and isinstance(weather, dict):
        temp = weather.get("temp_c", "N/A")
        hum = weather.get("humidity_pct", "N/A")
        rain = weather.get("rain_1h_mm", 0)
        city = weather.get("city", "Local Field")
        weather_desc = f"{city} | {temp}°C | {hum}% Humidity | Rain: {rain} mm/h"

    knowledge_summary = "General crop consultation"
    if knowledge and isinstance(knowledge, dict):
        knowledge_summary = (
            f"Pathogen: {knowledge.get('pathogen', 'N/A')}\n"
            f"Favorable Conditions: {knowledge.get('temp_min', '15')}-{knowledge.get('temp_max', '30')}°C, "
            f"Min Humidity: {knowledge.get('humidity_min', '70')}%, Rain Sensitive: {knowledge.get('rain_sensitive', 'False')}\n"
            f"Chemical Treatment: {knowledge.get('treatment_protocol', 'N/A')}\n"
            f"Organic Options: {knowledge.get('organic_remedies', 'N/A')}\n"
            f"Prevention: {knowledge.get('prevention', 'N/A')}"
        )

    context_prompt = f"""
[CURRENT SCAN TELEMETRY & CONTEXT]
- Plant / Leaf Diagnosis: {pred} ({conf}% confidence)
- Grad-CAM Affected Lesion Area: {f"{aff_pct:.1f}%" if aff_pct is not None else "Not calculated"}
- Microclimate Weather: {weather_desc}
- Progression Timeline: {timeline if timeline else "Single scan baseline"}
- Scientific Knowledge Grounding:
{knowledge_summary}

[FARMER QUESTION]
"{user_message}"

Answer as the AgroIntelli plant pathologist using the scan telemetry and scientific grounding above.
"""

    # If Gemini is available, query model
    if _model is not None:
        try:
            full_prompt = f"{SYSTEM_PROMPT}\n\n{context_prompt}"
            response = _model.generate_content(full_prompt)
            if response and response.text:
                return {
                    "reply": response.text.strip(),
                    "grounded": True,
                    "model_used": "Gemini Flash"
                }
        except Exception as e:
            print(f"⚠ Gemini API error, falling back to rule-grounded response: {e}")

    # Fallback to local rule-grounded reasoning if offline / API error
    fallback_reply = _build_local_grounded_reply(user_message, pred, aff_pct, weather, knowledge)
    return {
        "reply": fallback_reply,
        "grounded": True,
        "model_used": "AgroIntelli Local Expert"
    }


def _build_local_grounded_reply(query, pred, aff_pct, weather, knowledge):
    """Generates an immediate, high-quality agronomic response using local knowledge."""
    q_lower = query.lower()
    crop = knowledge.get("crop", "your crop")
    disease = knowledge.get("common_name", pred.replace("_", " ").title())
    pathogen = knowledge.get("pathogen", "fungal/bacterial pathogen")
    chem = knowledge.get("treatment_protocol", "Apply a broad-spectrum protective fungicide (e.g., Mancozeb or Chlorothalonil).")
    organic = knowledge.get("organic_remedies", "Neem oil spray (5ml/L) and copper-based organic formulations.")
    prevention = knowledge.get("prevention", "Avoid overhead watering; prune infected leaves to enhance air circulation.")

    temp = weather.get("temp_c", 25) if weather else 25
    hum = weather.get("humidity_pct", 65) if weather else 65
    rain = weather.get("rain_1h_mm", 0) if weather else 0

    aff_str = f"covering approximately **{aff_pct:.1f}%** of the leaf surface" if aff_pct is not None else ""

    if any(k in q_lower for k in ["spray", "chemical", "medicine", "treatment", "cure", "fungicide"]):
        return (
            f"### 🛡️ Recommended Treatment Protocol for **{disease}** ({pathogen})\n\n"
            f"Based on your scan {aff_str}:\n\n"
            f"1. **Chemical Treatment**:\n   - {chem}\n"
            f"   - *Application Tip*: Spray early in the morning (before 9 AM) or late afternoon. Ensure full coverage on both upper and lower leaf surfaces.\n\n"
            f"2. **Weather Considerations**:\n   - Current conditions: **{temp}°C, {hum}% humidity, {rain} mm/h rain**.\n"
            f"   - If rainfall is expected, use a non-ionic spreader/sticker so the spray does not wash off.\n\n"
            f"3. **Follow-up**:\n   - Re-inspect in 4-6 days to monitor if lesion borders have arrested."
        )

    if any(k in q_lower for k in ["organic", "natural", "home", "bio", "neem"]):
        return (
            f"### 🌿 Organic & Biological Solutions for **{disease}**\n\n"
            f"For sustainable, chemical-free management:\n\n"
            f"1. **Organic Formulations**:\n   - {organic}\n"
            f"   - *Neem Oil Recipe*: Mix 5 ml pure cold-pressed neem oil + 1 ml liquid soap in 1 liter of warm water. Spray every 5 days.\n\n"
            f"2. **Cultural Controls**:\n   - Immediately excise lower leaves that touch the soil.\n   - Mulch around the base to prevent fungal spores from splashing up from the soil.\n\n"
            f"3. **Prevention**:\n   - {prevention}"
        )

    if any(k in q_lower for k in ["rain", "weather", "humidity", "temperature", "climate"]):
        risk = "HIGH" if (hum >= 75 or rain > 0) else "MODERATE"
        return (
            f"### 🌦️ Microclimate Disease Risk Analysis\n\n"
            f"Current field readings: **{temp}°C | {hum}% Humidity | {rain} mm/h Rain**.\n\n"
            f"- **Spore Germination Risk**: **{risk}**\n"
            f"- Pathogen: *{pathogen}*.\n"
            f"- **Why it matters**: {'Persistent leaf wetness from rain accelerates spore release and germination.' if rain > 0 or hum >= 75 else 'Moderate humidity slows down rapid sporulation.'}\n\n"
            f"**Action Required**: Do not irrigate using sprinklers or overhead hoses. Water exclusively at the root base."
        )

    # General diagnosis breakdown
    return (
        f"### 🌾 AgroIntelli Agronomic Consultation\n\n"
        f"**Diagnosed Condition**: **{disease}** (*{pathogen}*)\n"
        f"{f'- **Lesion Severity**: {aff_pct:.1f}% affected leaf tissue.' if aff_pct is not None else ''}\n"
        f"- **Field Conditions**: {temp}°C, {hum}% humidity.\n\n"
        f"**Immediate Actions**:\n"
        f"1. **Curative Spray**: {chem}\n"
        f"2. **Organic Alternative**: {organic}\n"
        f"3. **Preventive Sanitation**: {prevention}\n\n"
        f"Feel free to ask for specific spray dosage, rain safeguards, or organic recipes!"
    )
