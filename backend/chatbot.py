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
            "Bengali": "নমস্কার! আমি এগ্রোবট (AgroBot), আপনার কৃষি সহকারী। পাতার ফটো স্ক্যান করুন, আগের ফিল্ড রেকর্ড দেখুন অথবা ফসল সেচ এবং রোগ নিরাময় সম্পর্কে প্রশ্ন করুন।",
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
    """Generates an immediate, natural agronomic response using local knowledge and MongoDB history in requested language."""
    q_lower = query.lower().strip()

    # 1. Greetings & Identity
    if any(q_lower.startswith(g) or q_lower == g for g in ["hi", "hello", "hey", "namaste", "hola", "namoshkar", "who are you", "what is your name"]):
        if language == "Hindi":
            return (
                "नमस्ते! मैं **एग्रोबॉट (AgroBot)** हूँ, आपका डिजिटल कृषि सहायक और फसल रोग विशेषज्ञ।\n\n"
                "आप पत्ती की फोटो अपलोड करके बीमारी का पता लगा सकते हैं, अपने पिछले रिकॉर्ड देख सकते हैं, "
                "या सिंचाई, खाद और फसल सुरक्षा से जुड़ा कोई भी सवाल पूछ सकते हैं।"
            )
        elif language == "Bengali":
            return (
                "নমস্কার! আমি **এগ্রোবট (AgroBot)**, আপনার ডিজিটাল কৃষি সহকারী এবং ফসল বিশেষজ্ঞ।\n\n"
                "আপনি পাতার ছবি আপলোড করে রোগ নির্ণয় করতে পারেন, আপনার অতীতের রেকর্ড পরীক্ষা করতে পারেন "
                "অথবা সেচ, সার এবং কীটনাশক সংক্রান্ত যেকোনো প্রশ্ন জিজ্ঞাসা করতে পারেন।"
            )
        return (
            "Hello! I am **AgroIntelli AI (AgroBot)**, your intelligent agricultural and crop health assistant.\n\n"
            "You can upload a leaf photo to diagnose diseases, check your past field records from MongoDB, "
            "or ask me any questions about watering, soil, crop pests, and treatment sprays. How can I assist you today?"
        )

    # 2. Watering & Irrigation
    if any(k in q_lower for k in ["water", "watering", "irrigation", "sinchai", "paani", "jol", "how often to water"]):
        if language == "Hindi":
            return (
                "### 💧 सिंचाई और नमी प्रबंधन सलाह\n\n"
                "1. **तरीका**: हमेशा **ड्रिप सिंचाई (Drip Irrigation)** का उपयोग करें। पत्तियों के ऊपर पानी छिड़कने से फफूंद जनित रोग तेजी से फैलते हैं।\n"
                "2. **समय**: सुबह जल्दी (6:00 AM - 9:00 AM) पानी दें ताकि पत्तियों पर पड़ा पानी सुबह की धूप में सूख जाए।\n"
                "3. **नमी जांच**: मिट्टी में 2 इंच गहराई पर नमी जांचें। यदि मिट्टी गीली महसूस हो तो सिंचाई रोक दें।"
            )
        elif language == "Bengali":
            return (
                "### 💧 সেচ এবং আর্দ্রতা ব্যবস্থাপনা পরামর্শ\n\n"
                "১. **পদ্ধতি**: সর্বদা **ড্রিপ সেচ (Drip Irrigation)** ব্যবহার করুন। পাতার ওপর জল ছেটালে ছত্রাকজনিত রোগ দ্রুত ছড়ায়।\n"
                "২. **সময়**: সকালের দিকে (৬:০০ AM - ৯:০০ AM) সেচ দিন যাতে পাতার ওপর জমাপড়া জল দ্রুত শুকিয়ে যায়।\n"
                "৩. **আর্দ্রতা পরীক্ষা**: মাটিতে ২ ইঞ্চি গভীরে আর্দ্রতা পরীক্ষা করুন। মাটি ভেজা থাকলে সেচ স্থগিত রাখুন।"
            )
        return (
            "### 💧 Practical Irrigation & Moisture Management\n\n"
            "Here are recommended watering guidelines for optimal crop vigor and disease prevention:\n\n"
            "1. **Method**: Always use **drip irrigation or ground-level soakers** at the root zone rather than overhead sprinklers.\n"
            "2. **Timing**: Irrigate **early in the morning (6:00 AM – 9:00 AM)** so leaf wetness evaporates quickly.\n"
            "3. **Frequency**: Deep, less frequent watering (2–3 times per week) promotes deeper root growth.\n"
            "4. **Moisture Check**: Check soil 2 inches deep before watering to avoid root rot."
        )

    # 3. Soil & Fertilization
    if any(k in q_lower for k in ["soil", "fertilizer", "fertiliser", "khad", "npk", "compost", "manure", "mati"]):
        if language == "Hindi":
            return (
                "### 🌱 मिट्टी का स्वास्थ्य और संतुलित पोषण\n\n"
                "1. **जैविक खाद**: मिट्टी में अच्छी तरह से सड़ी वर्मीकंपोस्ट या गोबर की खाद मिलाएं।\n"
                "2. **संतुलित NPK**: अत्यधिक नाइट्रोजन (N) के उपयोग से बचें, इससे पौधे कमजोर होते हैं और कीट-रोगों का हमला बढ़ता है।\n"
                "3. **मल्चिंग**: पौधों की जड़ों के पास सूखी घास या मल्च लगाएं ताकि मिट्टी की नमी बरकरार रहे।"
            )
        elif language == "Bengali":
            return (
                "### 🌱 মাটির স্বাস্থ্য এবং সুষম পুষ্টি\n\n"
                "১. **জৈব সার**: মাটিতে ভার্মিকম্পোস্ট বা পচা গোবর সার ভালো করে মিশিয়ে দিন।\n"
                "২. **সুষম NPK**: অতিরিক্ত নাইট্রোজেন ব্যবহার এড়িয়ে চলুন, এতে রোগ ও পোকার আক্রমণ বাড়ে।\n"
                "৩. **মালচিং**: গাছের গোড়ায় খড় বা কুপন ব্যবহার করে মালচিং করুন যাতে মাটির আর্দ্রতা বজায় থাকে।"
            )
        return (
            "### 🌱 Soil Health & Balanced Crop Nutrition\n\n"
            "1. **Organic Matter**: Incorporate well-aged compost or vermicompost to improve soil aeration and microbial activity.\n"
            "2. **Balanced N-P-K**: Avoid excessive nitrogen, which produces soft growth vulnerable to fungal blights.\n"
            "3. **Mulching**: Apply 2 inches of organic straw around the crop base to retain moisture and suppress weeds."
        )

    has_active_disease = bool(pred and pred != "None active" and pred != "Unknown / Not Scanned")
    crop = plant_name or knowledge.get("crop", "your crop")
    disease = knowledge.get("common_name", pred.replace("_", " ").title()) if has_active_disease else "Crop Health Consultation"
    pathogen = knowledge.get("pathogen", "fungal/bacterial pathogen")
    chem = knowledge.get("treatment_protocol", "Apply Mancozeb 2.5 g/L or Chlorothalonil 2 ml/L.")
    organic = knowledge.get("organic_remedies", "Neem oil spray (5 ml/L with mild soap).")
    prevention = knowledge.get("prevention", "Ensure good plant spacing and sanitize pruning tools.")

    temp = weather.get("temp_c", 25) if weather else 25
    hum = weather.get("humidity_pct", 65) if weather else 65
    rain = weather.get("rain_1h_mm", 0) if weather else 0

    # 4. Sprays & Chemical Fungicides
    if any(k in q_lower for k in ["spray", "chemical", "medicine", "treatment", "cure", "fungicide", "dawa", "aushadh"]):
        if language == "Hindi":
            return (
                f"### 🛡️ **{disease}** के लिए संस्तुत छिड़काव (Treatment Protocol)\n\n"
                f"1. **रासायनिक उपचार**: {chem}\n"
                f"2. **मौसम सावधानी**: वर्तमान तापमान **{temp}°C, नमी {hum}%** है। यदि बारिश की संभावना हो तो स्टिकर मिलाएं।\n"
                f"3. **समय**: सुबह 9 बजे से पहले या शाम को छिड़काव करें।"
            )
        elif language == "Bengali":
            return (
                f"### 🛡️ **{disease}** এর জন্য স্প্রে নির্দেশিকা\n\n"
                f"১. **রাসায়নিক প্রতিকার**: {chem}\n"
                f"২. **আবহাওয়া সর্তকতা**: বর্তমান তাপমাত্রা **{temp}°C, আর্দ্রতা {hum}%**। বৃষ্টিপাতের সম্ভাবনা থাকলে স্টিকার মেশান।\n"
                f"৩. **সময়**: সকাল ৯টার আগে বা বৈকালে স্প্রে করুন।"
            )
        return (
            f"### 🛡️ Recommended Spray Protocol for **{disease}**\n\n"
            f"1. **Chemical Treatment**: {chem}\n"
            f"2. **Weather Conditions**: **{temp}°C, {hum}% humidity, {rain} mm/h rain**.\n"
            f"3. **Application**: Spray early morning or late afternoon."
        )

    # 5. Organic Remedies
    if any(k in q_lower for k in ["organic", "natural", "home", "bio", "neem", "jaivik"]):
        if language == "Hindi":
            return (
                f"### 🌿 **{disease}** का जैविक उपचार\n\n"
                f"1. **नीम तेल गोल**: 5 मिली शुद्ध नीम तेल + 1 मिली तरल साबुन को 1 लीटर गुनगुने पानी में घोलकर 5-7 दिनों पर स्प्रे करें।\n"
                f"2. **जैविक विकल्प**: {organic}\n"
                f"3. **रोकथाम**: {prevention}"
            )
        elif language == "Bengali":
            return (
                f"### 🌿 **{disease}** এর জৈব সমাধান\n\n"
                f"১. **নিম তেলের মিশ্রণ**: ৫ মিলি নিম তেল + ১ মিলি তরল সাবান ১ লিটার উষ্ণ জলে মিশিয়ে স্প্রে করুন।\n"
                f"২. **জৈব বিকল্প**: {organic}\n"
                f"৩. **প্রতিরোধ**: {prevention}"
            )
        return (
            f"### 🌿 Organic & Biological Solutions for **{disease}**\n\n"
            f"1. **Neem Oil Recipe**: Mix 5 ml cold-pressed neem oil + 1 ml liquid soap per liter of water.\n"
            f"2. **Biological Remedy**: {organic}\n"
            f"3. **Prevention**: {prevention}"
        )

    # General Agronomic Response
    if language == "Hindi":
        return (
            f"धन्यवाद! मैं आपका फसल रोग और कृषि सलाहकार हूँ। "
            f"आप बीमारी के इलाज (*'आलू अगेती झुलसा का इलाज क्या है?'*), सिंचाई, या जैविक स्प्रे के बारे में सवाल पूछ सकते हैं "
            f"या पत्ती की फोटो अपलोड करके विस्तृत जांच पा सकते हैं।"
        )
    elif language == "Bengali":
        return (
            f"ধন্যবাদ! আমি আপনার শস্য স্বাস্থ্য ও কৃষি পরামর্শদাতা। "
            f"আপনি রোগ নিরাময় (*'আলুর আর্লি ব্লাইট রোগের প্রতিকার কী?'*), সেচ, বা জৈব স্প্রে সম্পর্কে প্রশ্ন জিজ্ঞাসা করতে পারেন "
            f"অথবা পাতার ছবি আপলোড করে বিস্তারিত রিপোর্ট পেতে পারেন।"
        )

    return (
        f"Thank you for reaching out. As your agricultural pathologist, I am here to assist with crop health, "
        f"plant nutrition, irrigation, and pest or disease control.\n\n"
        f"Feel free to ask a specific question (e.g., *'How to manage early blight in potatoes?'*), "
        f"or upload a leaf scan to receive an instant diagnostic breakdown and Grad-CAM lesion analysis."
    )

