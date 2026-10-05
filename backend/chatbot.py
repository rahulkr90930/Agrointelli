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


SYSTEM_PROMPT = """You are AgroIntelli AI (AgroBot), an intelligent, friendly, and practical agricultural pathologist and precision crop consultant.

BEHAVIOR GUIDELINES:
1. ADDRESS THE USER'S SPECIFIC QUESTION DIRECTLY:
   - Always answer what the user actually asked first and foremost.
   - For simple greetings ("hi", "hello", "hey", "who are you"), reply with a warm, natural 1-2 sentence welcome. Do NOT dump unsolicited chemical spray dosages or disclaimers.
   - For general agricultural questions (irrigation/watering, soil, fertilization, sunlight, pruning), provide clear, practical horticultural advice directly relevant to the question.
   - For non-agricultural topics (general chit-chat, math, programming, unrelated queries), provide a polite, concise answer and gently offer assistance with crop health and agronomy.

2. CONTEXT AWARENESS (WHEN RELEVANT):
   - When the user asks about diagnosing a leaf, treating an infection, or understanding symptoms, leverage the provided telemetry:
     * Diagnostic prediction & confidence
     * Grad-CAM lesion affected percentage (% of leaf blade)
     * Microclimate weather (temperature, humidity %, rainfall)
     * Scientific disease profile (pathogen, chemical dosage, organic remedies)
   - If the user asks about their previous field records or past scans, consult the provided MongoDB history.
   - If NO leaf scan is active and the user asks a general question, answer their question directly without unprompted chemical treatments.

3. PRACTICAL DOSAGES & PREVENTATIVE CARE (WHEN TREATMENT IS ASKED):
   - Provide standard concentrations (e.g., Mancozeb 2-2.5 g/L, Chlorothalonil 2 ml/L, Copper Oxychloride 3 g/L, cold-pressed neem oil 5 ml/L).
   - Highlight weather-sensitive application tips (spray in early morning/late afternoon, avoid spraying right before rain).

4. TONE & STRUCTURE:
   - Natural, conversational, and direct. Use bullet points and bold highlights for readability when helpful.
"""


FAST_MODELS = [
    "models/gemini-2.5-flash",
    "models/gemini-2.0-flash",
    "models/gemini-1.5-flash",
    "models/gemini-flash-latest",
    "models/gemini-3.5-flash-lite",
    "models/gemini-3.1-flash-lite"
]

FAST_GEN_CONFIG = {
    "max_output_tokens": 1024,
    "temperature": 0.35,
}


def generate_chat_response(user_message, scan_context=None, history=None, language="English"):
    """
    Generate an intelligent, conversational agronomic response grounded in live or MongoDB historical telemetry,
    disease knowledge, and requested language (English, Hindi, Spanish, etc.).
    """
    if not user_message or not user_message.strip():
        welcome_msgs = {
            "Hindi": "नमस्ते! मैं एग्रोबॉट (AgroBot) हूँ, आपका कृषि सहायक। पत्ती स्कैन करें, पिछले रिकॉर्ड देखें या फसल, सिंचाई और रोग उपचार के बारे में कुछ भी पूछें।",
            "Spanish": "¡Hola! Soy AgroBot, su asistente agrícola y de salud vegetal. Escanee una hoja, revise sus registros de MongoDB o pregúnteme sobre cultivos y tratamientos.",
        }
        return {
            "reply": welcome_msgs.get(language, "Hello! I am AgroBot, your crop health and agronomy assistant. Scan a leaf, check your past field records, or ask me anything about crops, diseases, irrigation, and care."),
            "grounded": False,
            "model_used": "system"
        }

    scan_context = scan_context or {}
    has_active_scan = bool(scan_context.get("prediction") and scan_context.get("prediction") != "Unknown / Not Scanned")
    pred = scan_context.get("prediction", "None active")
    conf = scan_context.get("confidence_pct", 0)
    aff_pct = scan_context.get("affected_pct", None)
    weather = scan_context.get("weather", {})
    knowledge = scan_context.get("knowledge_record", {})
    timeline = scan_context.get("timeline_notes", "")
    mongo_summary = scan_context.get("mongo_history_summary", "")
    is_historical = scan_context.get("is_historical", False)
    plant_name = scan_context.get("plant_name", "")

    # Format telemetry block
    weather_desc = "Not provided"
    if weather and isinstance(weather, dict):
        temp = weather.get("temp_c", "N/A")
        hum = weather.get("humidity_pct", "N/A")
        rain = weather.get("rain_1h_mm", 0)
        city = weather.get("city", "Local Field")
        weather_desc = f"{city} | {temp}°C | {hum}% Humidity | Rain: {rain} mm/h"

    knowledge_summary = "General crop consultation"
    if knowledge and isinstance(knowledge, dict) and has_active_scan:
        knowledge_summary = (
            f"Pathogen: {knowledge.get('pathogen', 'N/A')}\n"
            f"Favorable Conditions: {knowledge.get('temp_min', '15')}-{knowledge.get('temp_max', '30')}°C, "
            f"Min Humidity: {knowledge.get('humidity_min', '70')}%, Rain Sensitive: {knowledge.get('rain_sensitive', 'False')}\n"
            f"Chemical Treatment: {knowledge.get('treatment_protocol', 'N/A')}\n"
            f"Organic Options: {knowledge.get('organic_remedies', 'N/A')}\n"
            f"Prevention: {knowledge.get('prevention', 'N/A')}"
        )

    scan_status_line = "No active leaf scan uploaded in current session"
    if has_active_scan:
        scan_status_line = f"Active Leaf Scan: {pred} ({conf}% confidence), Lesion Area: {f'{aff_pct:.1f}%' if aff_pct is not None else 'N/A'}"
    elif is_historical:
        scan_status_line = f"Saved MongoDB Record for Crop: {plant_name or 'Past Leaf'}"

    mongo_block = ""
    if mongo_summary:
        mongo_block = f"""
[FARMER MONGODB CROP JOURNAL]
{mongo_summary}
"""

    context_prompt = f"""
[FIELD & TELEMETRY CONTEXT]
- Scan Status: {scan_status_line}
- Microclimate Weather: {weather_desc}
- Progression Timeline: {timeline if timeline else "None"}
- Agronomic Knowledge:
{knowledge_summary}
{mongo_block}
[USER QUESTION]
"{user_message}"

[INSTRUCTIONS]
1. Target Language: {language} (respond fluently in {language}).
2. Answer the user's question directly, naturally, and completely.
3. If this is a greeting or general question, respond conversationally without dumping unprompted chemical treatments or disclaimers.
4. If the question asks for disease diagnosis, remedies, or past records, use the provided context accurately.
"""

    # Try fast model candidates in sequence
    if _gemini_client is not None and GEMINI_API_KEY:
        for model_name in FAST_MODELS:
            try:
                candidate_model = _gemini_client.GenerativeModel(model_name, generation_config=FAST_GEN_CONFIG)
                full_prompt = f"{SYSTEM_PROMPT}\n\n{context_prompt}"
                response = candidate_model.generate_content(full_prompt)
                if response and response.text:
                    display_name = model_name.split("/")[-1].replace("gemini-", "Gemini ")
                    return {
                        "reply": response.text.strip(),
                        "grounded": True,
                        "model_used": display_name
                    }
            except Exception as e:
                # Log and fallback to next candidate
                continue

    # Fallback to local rule-grounded reasoning if offline / API error
    fallback_reply = _build_local_grounded_reply(
        user_message, pred, aff_pct, weather, knowledge,
        language=language, is_historical=is_historical, plant_name=plant_name, mongo_summary=mongo_summary
    )
    return {
        "reply": fallback_reply,
        "grounded": True,
        "model_used": "AgroIntelli Local Expert"
    }


def _build_local_grounded_reply(query, pred, aff_pct, weather, knowledge, language="English", is_historical=False, plant_name="", mongo_summary=""):
    """Generates an immediate, natural agronomic response using local knowledge and MongoDB history."""
    q_lower = query.lower().strip()

    # 1. Greetings & Identity
    if any(q_lower.startswith(g) or q_lower == g for g in ["hi", "hello", "hey", "namaste", "hola", "who are you", "what is your name", "who r u"]):
        return (
            "Hello! I am **AgroIntelli AI (AgroBot)**, your intelligent agricultural and crop health assistant.\n\n"
            "You can upload a leaf photo to diagnose diseases, check your past field records from MongoDB, "
            "or ask me any questions about watering, soil, crop pests, and treatment sprays. How can I assist you today?"
        )

    # 2. Watering & Irrigation
    if any(k in q_lower for k in ["water", "watering", "irrigation", "sinchai", "paani", "how often to water"]):
        return (
            "### 💧 Practical Irrigation & Moisture Management\n\n"
            "Here are recommended watering guidelines for optimal crop vigor and disease prevention:\n\n"
            "1. **Method**: Always use **drip irrigation or ground-level soakers** at the root zone rather than overhead sprinklers. Wet foliage is the #1 trigger for fungal spore germination and bacterial leaf spots.\n"
            "2. **Timing**: Irrigate **early in the morning (6:00 AM – 9:00 AM)**. Any incidental moisture on the leaves evaporates quickly with morning warmth, minimizing leaf-wetness duration.\n"
            "3. **Frequency**: Deep, less frequent watering (2–3 times per week, 1–1.5 inches total) promotes deeper root systems compared to shallow daily wetting.\n"
            "4. **Moisture Check**: Insert a finger or moisture meter 2 inches into the soil. If it feels cool and damp, delay watering to prevent root hypoxia and root rots (*Pythium/Phytophthora*)."
        )

    # 3. Soil & Fertilization
    if any(k in q_lower for k in ["soil", "fertilizer", "fertiliser", "khad", "npk", "compost", "manure"]):
        return (
            "### 🌱 Soil Health & Balanced Crop Nutrition\n\n"
            "Healthy soil is the first line of defense against crop stress and disease:\n\n"
            "1. **Organic Matter**: Incorporate well-aged compost or vermicompost (2–3 inches worked into the topsoil) to improve soil aeration, drainage, and beneficial microbial activity.\n"
            "2. **Balanced N-P-K**: Avoid excessive nitrogen (N) fertilization, which produces lush, succulent vegetative growth that is highly vulnerable to fungal blight and aphid infestations.\n"
            "3. **Soil pH**: Most vegetable crops (tomatoes, potatoes, corn) thrive in slightly acidic to neutral soil (pH 6.0–6.8). Test pH annually.\n"
            "4. **Mulching**: Apply 2 inches of organic straw or wood chips around the crop base to retain soil moisture, suppress weeds, and prevent soil-borne pathogens from splashing onto lower leaves."
        )

    has_active_disease = bool(pred and pred != "None active" and pred != "Unknown / Not Scanned")
    crop = plant_name or knowledge.get("crop", "your crop")
    disease = knowledge.get("common_name", pred.replace("_", " ").title()) if has_active_disease else "Crop Health Consultation"
    pathogen = knowledge.get("pathogen", "fungal/bacterial pathogen")
    chem = knowledge.get("treatment_protocol", "Apply a broad-spectrum protective fungicide (e.g., Mancozeb 2.5 g/L or Chlorothalonil 2 ml/L).")
    organic = knowledge.get("organic_remedies", "Cold-pressed neem oil spray (5 ml/L with mild surfactant) or dilute copper hydroxide.")
    prevention = knowledge.get("prevention", "Ensure good plant spacing, sanitize pruning tools, and water at the root base.")

    temp = weather.get("temp_c", 25) if weather else 25
    hum = weather.get("humidity_pct", 65) if weather else 65
    rain = weather.get("rain_1h_mm", 0) if weather else 0

    aff_str = f"covering approximately **{aff_pct:.1f}%** of the leaf surface" if (aff_pct is not None and has_active_disease) else ""
    source_prefix = "### 🍃 Historical Field Record (From MongoDB)\n\n" if is_historical else ""

    # 4. Sprays & Chemical Fungicides
    if any(k in q_lower for k in ["spray", "chemical", "medicine", "treatment", "cure", "fungicide", "dawa"]):
        return (
            f"{source_prefix}### 🛡️ Recommended Spray Protocol for **{disease}**\n\n"
            f"1. **Chemical Treatment**:\n   - {chem}\n"
            f"   - *Application Timing*: Spray in the early morning (before 9 AM) or late afternoon. Coat both upper and lower leaf surfaces.\n\n"
            f"2. **Weather Safeguards**:\n   - Current conditions: **{temp}°C, {hum}% humidity, {rain} mm/h rain**.\n"
            f"   - If rain is expected within 4–6 hours, delay spraying or add a sticker/adjuvant so the solution is not washed off.\n\n"
            f"3. **Follow-up**:\n   - Re-check after 5–7 days to ensure lesion expansion has halted."
        )

    # 5. Organic Remedies
    if any(k in q_lower for k in ["organic", "natural", "home", "bio", "neem", "jaivik"]):
        return (
            f"{source_prefix}### 🌿 Organic & Biological Solutions for **{disease}**\n\n"
            f"1. **Organic Formulation**:\n   - {organic}\n"
            f"   - *Neem Oil Recipe*: Mix 5 ml pure cold-pressed neem oil + 1 ml mild liquid soap into 1 liter of warm water. Spray every 5–7 days.\n\n"
            f"2. **Cultural Controls**:\n   - Prune and safely destroy lower infected leaves.\n   - Ensure adequate spacing between plants to maximize airflow.\n\n"
            f"3. **Prevention**:\n   - {prevention}"
        )

    # 6. MongoDB History / Past Records
    if any(k in q_lower for k in ["past", "history", "mongo", "previous", "record", "purani", "pichla"]):
        history_info = f"\n\n**Your Saved Field Records**:\n{mongo_summary}" if mongo_summary else ""
        return (
            f"### 📜 AgroIntelli Field Journal (MongoDB Records)\n\n"
            f"Here is your historical crop health journal:\n"
            f"- **Latest Saved Crop**: **{crop}**\n"
            f"- **Diagnosis**: **{disease}**\n"
            f"{f'- **Lesion Area**: {aff_pct:.1f}%' if (aff_pct is not None and has_active_disease) else ''}"
            f"{history_info}\n\n"
            f"**Next Step**: Upload a new leaf photo to compare disease progression over time."
        )

    # 7. Weather / Climate Risk
    if any(k in q_lower for k in ["rain", "weather", "humidity", "temperature", "climate", "mausam"]):
        risk = "HIGH" if (hum >= 75 or rain > 0) else "MODERATE"
        return (
            f"{source_prefix}### 🌦️ Microclimate Disease Risk Telemetry\n\n"
            f"Field conditions: **{temp}°C | {hum}% Humidity | {rain} mm/h Rain**.\n\n"
            f"- **Spore Germination Risk**: **{risk}**\n"
            f"- **Analysis**: {'Prolonged humidity and rain create optimal conditions for fungal sporulation and bacterial splash dispersal.' if risk == 'HIGH' else 'Moderate conditions slow pathogen spread. Maintain baseline preventative care.'}\n"
            f"- **Action**: Keep leaf canopies well-ventilated and avoid working in the field while plants are wet."
        )

    # 8. Active Disease Overview (if a scan was performed)
    if has_active_disease:
        return (
            f"{source_prefix}### 🌾 Diagnostic Summary: **{disease}** ({pathogen})\n\n"
            f"{f'- **Lesion Severity**: {aff_pct:.1f}% affected leaf tissue.' if aff_pct is not None else ''}\n"
            f"- **Field Conditions**: {temp}°C, {hum}% humidity.\n\n"
            f"**Recommended Steps**:\n"
            f"1. **Curative Spray**: {chem}\n"
            f"2. **Organic Alternative**: {organic}\n"
            f"3. **Prevention**: {prevention}\n\n"
            f"Ask me for specific application dosages, rainfall timing, or organic preparation recipes!"
        )

    # 9. General Agronomic Response
    return (
        f"Thank you for reaching out. As your agricultural pathologist, I am here to assist with all aspects of crop health, "
        f"plant nutrition, irrigation, and pest or disease control.\n\n"
        f"Feel free to ask a specific question (e.g., *'How to manage early blight in potatoes?'*, *'How often should I water corn?'*), "
        f"or upload a leaf scan to receive an instant diagnostic breakdown and Grad-CAM lesion analysis."
    )
