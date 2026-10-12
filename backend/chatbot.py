"""
AgroIntelli Precision Agronomic Chatbot Engine
Multimodal agricultural intelligence powered by Google Gemini (gemini-3.8-flash)
with instant, deeply-grounded local agronomic fallback covering 39 PlantVillage pathologies,
live OpenWeather telemetry, and MongoDB journal progression histories.
"""

import os
import sys
import re
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
IS_VALID_GEMINI_KEY = bool(GEMINI_API_KEY and len(GEMINI_API_KEY) > 20)

# Disease Knowledge Store import for grounding
try:
    from .inference import KNOWLEDGE_STORE
except (ImportError, ValueError):
    try:
        from inference import KNOWLEDGE_STORE
    except Exception:
        KNOWLEDGE_STORE = {}

# Gemini SDK setup
_gemini_client = None
_active_gemini_model = None

if IS_VALID_GEMINI_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_client = genai
        # Candidate models in preference order
        candidate_models = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-1.5-flash"]
        for m_name in candidate_models:
            try:
                m = genai.GenerativeModel(m_name)
                _active_gemini_model = m_name
                break
            except Exception:
                continue
        if _active_gemini_model:
            print(f"🌿 AgroIntelli Chatbot initialized with Google Gemini API ({_active_gemini_model}).")
        else:
            print("🌿 AgroIntelli Chatbot initialized with Gemini API.")
    except Exception as e:
        print(f"ℹ️ Google Gemini SDK notice: {e}. Local agronomic expert active.")
else:
    print("🌿 AgroIntelli Chatbot initialized in ultra-fast local agronomic expert mode.")

SYSTEM_PROMPT = """You are AgroIntelli AI (AgroBot), an expert agricultural pathologist, precision agronomist, and crop advisor.

BEHAVIOR GUIDELINES:
1. ADDRESS THE USER'S SPECIFIC QUESTION DIRECTLY:
   - Always answer what the user actually asked first and foremost.
   - GIBBERISH / NONSENSE / UNRECOGNIZABLE TEXT (e.g., 'tbrehh', 'asdfgh', random letters):
     * DO NOT fabricate or assume a leaf scan!
     * Politely inform the user that the input was not understood, and ask them to specify their crop, disease, or farming question.
   - DIFFERENT CROPS:
     * If the user asks about a specific crop or disease (e.g. orange citrus greening, apple scab, corn rust), answer specifically about THAT crop and disease, even if a different leaf (like tomato) was previously scanned or is present in telemetry!
   - WEATHER / TEMPERATURE:
     * If the user asks about the weather, temperature, or humidity, provide the current telemetry (temperature in °C, humidity %, rain) and discuss how these conditions affect plant disease risks (fungal infection windows, irrigation timing).
   - SAVED PLANTS / JOURNAL / MONGO HISTORY:
     * If the user asks about their saved plants, past scans, or journal history, use the provided MongoDB history summary to answer them accurately.
   - CURRENT LEAF SCAN:
     * When the user asks about their currently uploaded/scanned leaf, leverage the provided telemetry:
       * Diagnostic prediction & confidence %
       * Grad-CAM lesion affected percentage (% of leaf blade)
       * Microclimate weather and disease risk
       * Scientific pathogen, chemical dosages (e.g. Mancozeb 2-2.5 g/L, Chlorothalonil 2 ml/L, Copper Oxychloride 3 g/L, cold-pressed neem oil 5 ml/L), organic remedies, and prevention.
2. TONE & STRUCTURE:
   - Professional, encouraging, clear, structured with bold highlights and bullet points.
   - Do not invent values or provide generic non-answers.
"""

FAST_GEN_CONFIG = {
    "max_output_tokens": 800,
    "temperature": 0.3,
}

# Supported crops for entity matching
CROPS_LIST = [
    "tomato", "potato", "corn", "apple", "grape", "orange",
    "cherry", "peach", "pepper", "strawberry", "squash",
    "soybean", "blueberry", "raspberry"
]

# Disease keyword lookup map
DISEASE_KEYWORDS = {
    "citrus greening": "orange_haunglongbing_citrus_greening",
    "huanglongbing": "orange_haunglongbing_citrus_greening",
    "early blight": "tomato_early_blight",
    "late blight": "tomato_late_blight",
    "apple scab": "apple_apple_scab",
    "black rot": "apple_black_rot",
    "cedar apple rust": "apple_cedar_apple_rust",
    "cedar rust": "apple_cedar_apple_rust",
    "common rust": "corn_common_rust",
    "gray leaf spot": "corn_gray_leaf_spot",
    "northern leaf blight": "corn_northern_leaf_blight",
    "powdery mildew": "squash_powdery_mildew",
    "leaf mold": "tomato_leaf_mold",
    "septoria": "tomato_septoria_leaf_spot",
    "spider mite": "tomato_spider_mites_two_spotted_spider_mite",
    "spider mites": "tomato_spider_mites_two_spotted_spider_mite",
    "yellow leaf curl": "tomato_yellow_leaf_curl_virus",
    "mosaic virus": "tomato_mosaic_virus",
    "bacterial spot": "pepper_bacterial_spot",
    "leaf scorch": "strawberry_leaf_scorch",
    "esca": "grape_esca_black_measles",
    "black measles": "grape_esca_black_measles",
}


def is_gibberish(text):
    """
    Detects random keystrokes, keyboard mashing, or unpronounceable non-words
    such as 'tbrehh', 'asdfgh', 'qwerty', 'zzzz', etc.
    """
    if not text:
        return True
    t = text.strip().lower()
    words = re.findall(r'[a-zA-Z]+', t)
    if not words:
        return True

    # Established English words, conversational phrases, and agricultural vocabulary
    meaningful_words = {
        'hi', 'hey', 'hello', 'who', 'what', 'when', 'where', 'why', 'how', 'is', 'are', 'am',
        'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'can', 'you',
        'i', 'my', 'me', 'we', 'our', 'it', 'its', 'do', 'does', 'did', 'have', 'has', 'had',
        'tell', 'show', 'give', 'help', 'cure', 'treat', 'spray', 'disease', 'leaf', 'plant',
        'crop', 'water', 'soil', 'weather', 'temp', 'rain', 'temperature', 'humidity', 'wind',
        'good', 'morning', 'night', 'name', 'bot', 'agrobot', 'please', 'thanks', 'thank',
        'tomato', 'potato', 'corn', 'apple', 'grape', 'orange', 'citrus', 'peach', 'cherry',
        'pepper', 'strawberry', 'squash', 'soybean', 'blueberry', 'raspberry', 'blight',
        'scab', 'rot', 'rust', 'mildew', 'spot', 'mold', 'virus', 'canker', 'neem', 'fertilizer',
        'compost', 'npk', 'urea', 'pruning', 'drip', 'irrigation', 'greening', 'scorch',
        'healthy', 'bacteria', 'fungus', 'fungal', 'damage', 'sunlight', 'garden', 'farm',
        'namaste', 'hola', 'namoshkar', 'karan', 'upchar', 'dawa', 'sinchai', 'paani', 'mati'
    }

    meaningful_count = 0
    for w in words:
        if w in meaningful_words:
            meaningful_count += 1
            continue
        # Keyboard mashing row substrings
        if any(mash in w for mash in ["asdf", "qwerty", "zxcv", "hjkl", "ghjk", "dfgh"]):
            continue
        # 4 or more consonants in a row
        if re.search(r'[^aeiouy\s]{4,}', w):
            continue
        # Repeated character 3+ times
        if re.search(r'(.)\1{2,}', w):
            continue
        # Vowel to length ratio check for words >= 4 letters
        if len(w) >= 4:
            v_count = sum(1 for c in w if c in 'aeiouy')
            if v_count == 0 or (v_count / len(w)) < 0.2:  # e.g., 'tbrehh' has 1/6 = 0.166
                continue
        meaningful_count += 1

    return meaningful_count == 0


def match_disease_or_crop_from_query(query):
    """
    Extracts an explicitly mentioned disease or crop from the user query.
    Returns (matched_key, matched_record, matched_crop_name).
    """
    if not query:
        return None, None, None
    q = query.lower().replace("-", " ").replace("_", " ")

    # 1. Exact disease keywords
    for kw, key in DISEASE_KEYWORDS.items():
        if kw in q and key in KNOWLEDGE_STORE:
            rec = KNOWLEDGE_STORE[key]
            return key, rec, rec.get("crop", key.split("_")[0])

    # 2. Knowledge store direct matches
    for k, v in KNOWLEDGE_STORE.items():
        name_clean = k.replace("_", " ").lower()
        common = v.get("common_name", "").lower()
        if name_clean in q or (common and common in q):
            return k, v, v.get("crop", k.split("_")[0])

    # 3. Explicit crop mentions
    for crop in CROPS_LIST:
        # Match crop as standalone word or token
        if re.search(r'\b' + re.escape(crop) + r'\b', q):
            # Find representative disease or healthy profile for this crop
            for k, v in KNOWLEDGE_STORE.items():
                if k.startswith(crop) and "healthy" not in k:
                    return k, v, crop
            # Fallback to healthy if no infected found
            for k, v in KNOWLEDGE_STORE.items():
                if k.startswith(crop):
                    return k, v, crop
            return None, None, crop

    return None, None, None


def is_weather_inquiry(query):
    """Detects inquiries about weather, temperature, humidity, rain, etc."""
    q = query.lower()
    weather_words = [
        "weather", "temperature", "temp", "humidity", "rain", "rainfall",
        "hot", "cold", "forecast", "climate", "mausam", "tapman", "barish",
        "hawa", "degrees", "celsius"
    ]
    return any(re.search(r'\b' + re.escape(w) + r'\b', q) for w in weather_words)


def is_journal_inquiry(query):
    """Detects inquiries about saved plants, personal journal, or past scan history."""
    q = query.lower()
    journal_words = [
        "my plant", "my plants", "my journal", "saved plant", "saved plants",
        "tracked crop", "tracked crops", "history", "records", "timeline",
        "checkin", "check in", "previous scan", "meri fasal", "mera record"
    ]
    return any(w in q for w in journal_words)


def is_greeting(query):
    """Detects simple conversational greetings."""
    q = query.lower().strip()
    greetings = ["hi", "hello", "hey", "namaste", "hola", "namoshkar", "good morning", "good evening", "who are you", "what is your name"]
    return any(q == g or q.startswith(g + " ") or q.startswith(g + "!") for g in greetings)


def generate_chat_response(user_message, scan_context=None, history=None, language="English"):
    """
    Generates a conversational agronomic response grounded in scan telemetry,
    pathology knowledge base, live weather, and MongoDB history.
    Responds via Google Gemini (gemini-3.8-flash) or ultra-fast local expert.
    """
    msg = (user_message or "").strip()

    # 1. Empty message fallback
    if not msg:
        welcome_msgs = {
            "Hindi": "नमस्ते! मैं एग्रोबॉट (AgroBot) हूँ, आपका कृषि सहायक। पत्ती स्कैन करें या फसल, सिंचाई और रोग उपचार के बारे में कुछ भी पूछें।",
            "Bengali": "নমস্কার! আমি এগ্রোবট (AgroBot), আপনার ডিজিটাল কৃষি সহকারী। পাতার ছবি স্ক্যান করুন বা ফসল ও রোগ সম্পর্কে জিজ্ঞাসা করুন।",
            "Spanish": "¡Hola! Soy AgroBot, su consultor agronómico. Suba una foto de hoja o pregunte sobre cualquier cultivo o enfermedad.",
        }
        return {
            "reply": welcome_msgs.get(language, "Hello! I am AgroIntelli AI (AgroBot), your precision crop health consultant. Upload a leaf scan or ask me anything about crop diseases, watering, and treatments!"),
            "grounded": True,
            "model_used": "AgroIntelli Local Expert"
        }

    # 2. Gibberish / Meaningless text check
    if is_gibberish(msg):
        gibberish_replies = {
            "Hindi": "क्षमा करें, मुझे आपका सवाल समझ नहीं आया। 🌾 कृपया अपना प्रश्न स्पष्ट शब्दों में लिखें—जैसे *'संतरा सिट्रस ग्रीनिंग का इलाज क्या है?'*, *'टमाटर अगेती झुलसा के लक्षण'*, या *'आज का तापमान और मौसम'*।",
            "Bengali": "দুঃখিত, আপনার বার্তাটি বোঝা যায়নি। 🌾 অনুগ্রহ করে ফসল, রোগ বা চাষাবাদ সংক্রান্ত প্রশ্ন স্পষ্টভাবে জিজ্ঞাসা করুন—যেমন *'টমেটোর ব্লাইটের প্রতিকার কী?'* বা *'আজকের আবহাওয়া ও তাপমাত্রা'*।",
            "Spanish": "Disculpe, no pude entender su consulta. 🌾 Por favor, escriba su pregunta con claridad sobre cultivos (ej. tomate, naranja, manzana), enfermedades, riego o clima actual.",
        }
        return {
            "reply": gibberish_replies.get(language, "I couldn't quite understand that query. 🌾 Could you please rephrase or specify which crop, disease (e.g., Tomato Early Blight, Orange Citrus Greening, Apple Scab), weather condition, or farming practice you'd like advice on?"),
            "grounded": False,
            "model_used": "AgroIntelli Guard"
        }

    # 3. Extract scan telemetry
    scan_ctx = scan_context or {}
    scanned_pred = scan_ctx.get("prediction")
    conf = scan_ctx.get("confidence_pct", 0)
    weather = scan_ctx.get("weather", {})
    knowledge = scan_ctx.get("knowledge_record")
    gradcam_data = scan_ctx.get("gradcam", {})
    aff_pct = (
        gradcam_data.get("affected_pct", scan_ctx.get("affected_pct"))
        if isinstance(gradcam_data, dict)
        else scan_ctx.get("affected_pct")
    )
    plant_name = scan_ctx.get("plant_name", "")
    mongo_summary = scan_ctx.get("mongo_history_summary", "")
    is_historical = scan_ctx.get("is_historical", False)

    # 4. Entity Extraction & Conflict Resolution
    # Does user query explicitly mention another crop or disease?
    query_key, query_rec, query_crop = match_disease_or_crop_from_query(msg)
    user_asked_different_crop = False

    active_knowledge = knowledge
    active_pred = scanned_pred

    if query_rec:
        # If user explicitly asks about a disease or crop, that OVERRIDES any loaded leaf!
        if not scanned_pred or (query_crop and query_crop.lower() not in (scanned_pred or "").lower()):
            user_asked_different_crop = True
            active_knowledge = query_rec
            active_pred = query_key
    elif query_crop:
        # User mentioned a crop name without specific disease
        user_asked_different_crop = True
        active_knowledge = None
        active_pred = query_crop

    # Weather & Journal intent flags
    is_weather = is_weather_inquiry(msg)
    is_journal = is_journal_inquiry(msg)

    # 5. Try Google Gemini with active model (e.g. gemini-3.8-flash)
    if IS_VALID_GEMINI_KEY and _gemini_client is not None:
        try:
            target_model = _active_gemini_model or "gemini-3.8-flash"
            weather_desc = f"{weather.get('temp_c', '26')}°C, {weather.get('humidity_pct', '65')}% humidity, {weather.get('condition', 'Optimal')}, Rain: {weather.get('rain_1h_mm', '0')} mm/h"
            
            k_sum = ""
            if active_knowledge:
                k_sum = (
                    f"Selected Topic/Disease: {active_knowledge.get('common_name', active_pred)}\n"
                    f"Crop: {active_knowledge.get('crop', query_crop or 'Crop')}\n"
                    f"Pathogen: {active_knowledge.get('pathogen', 'N/A')}\n"
                    f"Chemical Treatment: {active_knowledge.get('treatment_protocol', 'N/A')}\n"
                    f"Organic Remedies: {active_knowledge.get('organic_remedies', 'N/A')}\n"
                    f"Prevention: {active_knowledge.get('prevention', 'N/A')}\n"
                    f"Temp Range: {active_knowledge.get('temp_min', 18)}–{active_knowledge.get('temp_max', 30)}°C, Humidity: >{active_knowledge.get('humidity_min', 60)}%"
                )

            # Build rich prompt for Gemini
            prompt_parts = [
                f"{SYSTEM_PROMPT}\n",
                f"[TELEMETRY & CONTEXT]",
                f"- Active leaf scan loaded: {scanned_pred or 'None'}",
                f"- User query mentions specific topic: {active_pred if user_asked_different_crop else 'No override'}",
                f"- Current Microclimate Weather: {weather_desc}",
            ]
            if not user_asked_different_crop and scanned_pred and aff_pct is not None:
                prompt_parts.append(f"- Scanned Leaf Lesion Area (Grad-CAM): {aff_pct}% blade affected (Confidence: {conf}%)")
            if mongo_summary:
                prompt_parts.append(f"- User MongoDB Journal History:\n{mongo_summary}")
            if k_sum:
                prompt_parts.append(f"- Agronomic Grounding Knowledge:\n{k_sum}")

            prompt_parts.extend([
                f"\n[USER QUESTION]",
                f"{msg}\n",
                f"[INSTRUCTIONS]",
                f"1. Answer what the user specifically asked in a helpful, structured response (2-5 concise paragraphs or bullet points).",
                f"2. If the user asked about weather/temperature, focus on the weather telemetry and how it impacts plants.",
                f"3. If the user asked about a different crop/disease (e.g. orange vs tomato), answer specifically about that crop/disease.",
                f"4. If the user asked about their journal/saved plants, answer based on the MongoDB records.",
                f"5. Target Language: {language}",
                f"6. Do not output raw markdown heading markers (#), bold text is acceptable."
            ])

            full_prompt = "\n".join(prompt_parts)
            cand = _gemini_client.GenerativeModel(target_model, generation_config=FAST_GEN_CONFIG)
            res = cand.generate_content(full_prompt, request_options={"timeout": 6.0})
            if res and res.text and len(res.text.strip()) > 10:
                return {
                    "reply": res.text.strip(),
                    "grounded": True,
                    "model_used": f"Gemini ({target_model})"
                }
        except Exception as e:
            # Fallback to local expert engine smoothly
            pass

    # 6. Ultra-fast, deeply-grounded localized agronomic expert
    reply = _build_local_grounded_reply(
        msg,
        scanned_pred=scanned_pred,
        active_pred=active_pred,
        aff_pct=aff_pct,
        weather=weather,
        knowledge=active_knowledge,
        language=language,
        is_historical=is_historical,
        plant_name=plant_name,
        mongo_summary=mongo_summary,
        confidence_pct=conf,
        is_weather=is_weather,
        is_journal=is_journal,
        user_asked_different_crop=user_asked_different_crop,
        query_crop=query_crop
    )
    return {
        "reply": reply,
        "grounded": True,
        "model_used": "AgroIntelli Local Expert"
    }


def _build_local_grounded_reply(
    query, scanned_pred=None, active_pred=None, aff_pct=None, weather=None,
    knowledge=None, language="English", is_historical=False, plant_name="",
    mongo_summary="", confidence_pct=0, is_weather=False, is_journal=False,
    user_asked_different_crop=False, query_crop=None
):
    """
    Deterministic, highly specific localized agronomic expert handler.
    Accurately branches on user intent: Greetings, Weather, Saved Journal,
    Specific Crop/Disease Overrides, Irrigation, Fertilizer, or Active Scan.
    """
    q_low = query.lower().strip()
    wx = weather or {}
    temp_c = wx.get("temp_c", 26)
    hum_pct = wx.get("humidity_pct", 65)
    rain_mm = wx.get("rain_1h_mm", 0)
    condition = wx.get("condition", "Partly Cloudy")

    # 1. Greetings / Chit-chat
    if is_greeting(query):
        if language == "Hindi":
            return (
                "नमस्ते! मैं **एग्रोबॉट (AgroBot)** हूँ, आपका डिजिटल फसल सलाहकार और पौध रोग विशेषज्ञ। 🌾\n\n"
                "आप किसी भी फसल (टमाटर, संतरा, आलू, सेब, मक्का), रोग उपचार (कवकनाशी और जैविक नीम तेल), "
                "वर्तमान मौसम व सिंचाई, या अपने सहेजे गए पौधों के इतिहास के बारे में पूछ सकते हैं। मैं आपकी क्या मदद करूँ?"
            )
        elif language == "Bengali":
            return (
                "নমস্কার! আমি **এগ্রোবট (AgroBot)**, আপনার ডিজিটাল কৃষি বিশেষজ্ঞ ও ফসল সহকারী। 🌾\n\n"
                "আপনি যেকোনো ফসল (টমেটো, কমলা, আপেল, ভুট্টা), রোগের সঠিক প্রতিকার, জৈব কীটনাশক, "
                "বর্তমান তাপমাত্রা ও সেচ, অথবা জার্নাল হিস্ট্রি সম্পর্কে জিজ্ঞাসা করতে পারেন। আজ কীভাবে সাহায্য করতে পারি?"
            )
        return (
            "Hello! I am **AgroIntelli AI (AgroBot)**, your precision agricultural pathologist and crop health consultant. 🌾\n\n"
            "You can ask me about any crop disease (e.g. Orange Citrus Greening, Tomato Blights, Apple Scab), "
            "chemical fungicide dosages, organic neem sprays, live microclimate weather risk, or your saved plant records. How can I help you today?"
        )

    # 2. Weather & Temperature Inquiries
    if is_weather:
        disease_risk = "Elevated risk for fungal blights and foliar sporulation" if hum_pct >= 75 else "Moderate disease pressure; favorable for standard foliage development"
        if language == "Hindi":
            return (
                f"### ⛅ वर्तमान मौसम और कृषि मौसम जोखिम रिपोर्ट\n\n"
                f"- **तापमान**: **{temp_c}°C**\n"
                f"- **सापेक्ष आर्द्रता (Humidity)**: **{hum_pct}%**\n"
                f"- **वर्षा (Rainfall)**: **{rain_mm} मिमी/घंटा** ({condition})\n\n"
                f"**फसल स्वास्थ्य प्रभाव**: {disease_risk}। "
                f"यदि आर्द्रता 75% से अधिक है, तो शाम के समय पत्तियों पर पानी छिड़कने से बचें और निचली पत्तियों पर फफूंद के लक्षणों की जांच करें।"
            )
        elif language == "Bengali":
            return (
                f"### ⛅ বর্তমান আবহাওয়া ও ফসল সুরক্ষা রিপোর্ট\n\n"
                f"- **তাপমাত্রা**: **{temp_c}°C**\n"
                f"- **বাতাসের আর্দ্রতা**: **{hum_pct}%**\n"
                f"- **বৃষ্টিপাত**: **{rain_mm} মিমি/ঘণ্টা** ({condition})\n\n"
                f"**কৃষি পরামর্শ**: {disease_risk}। আর্দ্রতা বেশি থাকলে সন্ধ্যায় সেচ পরিহার করুন যাতে পাতায় দীর্ঘক্ষণ জল জমে না থাকে।"
            )
        return (
            f"### ⛅ Current Microclimate Weather & Agronomic Risk\n\n"
            f"- **Ambient Temperature**: **{temp_c}°C**\n"
            f"- **Relative Humidity**: **{hum_pct}%**\n"
            f"- **Precipitation**: **{rain_mm} mm/h** ({condition})\n\n"
            f"**Pathogen Risk Assessment**: {disease_risk}.\n"
            f"When relative humidity exceeds 70–75%, fungal spores (such as *Alternaria* and *Phytophthora*) germinate rapidly on wet foliage. "
            f"Ensure good row ventilation and switch to morning ground-level drip irrigation."
        )

    # 3. Saved Plants / MongoDB Journal Inquiries
    if is_journal:
        if mongo_summary:
            if language == "Hindi":
                return (
                    f"### 📖 आपके सहेजे गए पौधों का जर्नल इतिहास (MongoDB)\n\n"
                    f"{mongo_summary}\n\n"
                    f"**सुझाव**: किसी भी पौधे पर फॉलो-अप स्कैन जोड़ने के लिए 'My Plants' टैब पर जाएं और 'Add Follow-up Scan' पर टैप करें।"
                )
            return (
                f"### 📖 Your Tracked Crops & Leaf Journal Records\n\n"
                f"Here are the records currently stored in your personal MongoDB journal:\n\n"
                f"{mongo_summary}\n\n"
                f"**Action Advice**: You can track progressive lesion recovery over time by opening the **My Plants** tab and adding follow-up scans as your treatments take effect."
            )
        else:
            return (
                "### 📖 Personal Plant Journal\n\n"
                "You do not have any saved plant records in your journal yet. "
                "After running a leaf scan on the **Scan** page, tap **'Save to My Plants Journal'** to begin tracking day-by-day healing and Grad-CAM lesion recovery!"
            )

    # 4. Irrigation & Watering
    if any(k in q_low for k in ["water", "watering", "irrigation", "sinchai", "paani", "jol", "how often to water"]):
        if language == "Hindi":
            return (
                "### 💧 सिंचाई और जल प्रबंधन के महत्वपूर्ण नियम\n\n"
                "1. **सिंचाई विधि**: पत्तियों पर ऊपर से पानी छिड़कने से बचें। हमेशा **ड्रिप सिंचाई (Drip)** या जड़ों के पास पानी दें, ताकि फफूंद न फैले।\n"
                "2. **सही समय**: सिंचाई हमेशा **सुबह 6:00 से 9:00 बजे** के बीच करें ताकि पत्तियों पर पड़ा पानी धूप में जल्दी सूख जाए।\n"
                "3. **नमी जांच**: पानी देने से पहले मिट्टी में 2 इंच गहराई पर नमी जांचें। यदि मिट्टी नम है, तो सिंचाई रोक दें।"
            )
        return (
            "### 💧 Practical Irrigation & Moisture Management Guidelines\n\n"
            "1. **Root-Zone Delivery**: Always water at the base of the plant using **drip irrigation or ground soaker hoses**. Avoid overhead sprinklers, as wet foliage triggers fungal spore germination.\n"
            "2. **Timing**: Water **early in the morning (6:00 AM – 9:00 AM)** so that any moisture dries rapidly under morning sunlight.\n"
            "3. **Frequency**: Deep watering 2–3 times per week is far superior to shallow daily watering, promoting resilient deep root growth."
        )

    # 5. Soil, Compost & Fertilizers
    if any(k in q_low for k in ["soil", "fertilizer", "fertiliser", "khad", "npk", "compost", "manure", "mati", "potash"]):
        if language == "Hindi":
            return (
                "### 🌱 मिट्टी का स्वास्थ्य और संतुलित पोषक तत्व\n\n"
                "1. **जैविक खाद**: बुवाई से पहले 2-3 टन प्रति एकड़ अच्छी सड़ी हुई गोबर की खाद या वर्मीकंपोस्ट मिट्टी में मिलाएं।\n"
                "2. **संतुलित NPK**: अत्यधिक नाइट्रोजन (यूरिया) देने से पौधे कोमल हो जाते हैं और फफूंद रोग तेजी से पकड़ते हैं। पोटाश (K) और फास्फोरस (P) का संतुलन बनाए रखें।\n"
                "3. **मल्चिंग**: पौधों की जड़ों के पास 2-3 इंच सूखी घास की परत (मल्च) लगाएं ताकि नमी बनी रहे और मिट्टी का कटाव रुके।"
            )
        return (
            "### 🌱 Soil Health & Balanced Crop Nutrition\n\n"
            "1. **Organic Enrichment**: Incorporate well-decomposed vermicompost or farmyard manure before planting to enhance beneficial soil microbes.\n"
            "2. **Balanced N-P-K Ratio**: Avoid excessive nitrogen (Urea), which causes lush, soft tissue highly vulnerable to fungal blight. Ensure adequate Potassium (K) to reinforce plant cell walls.\n"
            "3. **Soil Mulching**: Apply a 2-inch organic mulch layer around the root zone to conserve moisture and prevent soil-splash pathogen dispersal."
        )

    # 6. Specific Crop or Disease Knowledge (from User Query or Active Scan)
    # If the user asked about a different crop, knowledge is already set to that crop!
    if knowledge:
        crop = knowledge.get("crop", query_crop or "Crop").title()
        d_name = knowledge.get("common_name", (active_pred or "Plant Disease").replace("_", " ").title())
        pathogen = knowledge.get("pathogen", "Pathogenic microorganism")
        chem = knowledge.get("treatment_protocol", "Apply standard fungicide spray as per label recommendations.")
        org = knowledge.get("organic_remedies", "Spray cold-pressed neem oil (5 ml/L with 1 ml mild liquid soap).")
        prev = knowledge.get("prevention", "Maintain adequate plant spacing, sanitize pruning shears, and remove fallen infected leaves.")
        temp_min = knowledge.get("temp_min", "18")
        temp_max = knowledge.get("temp_max", "30")
        hum_min = knowledge.get("humidity_min", "70")

        # Specific: Treatment & Chemical Sprays
        if any(k in q_low for k in ["treatment", "cure", "spray", "chemical", "medicine", "dawa", "fungicide", "pesticide", "how to treat", "chikitsa"]):
            if language == "Hindi":
                return (
                    f"### 💊 **{d_name} ({crop})** का प्रभावी उपचार प्रोटोकॉल\n\n"
                    f"**रोगकारक (Pathogen)**: *{pathogen}*\n\n"
                    f"1. **रासायनिक कवकनाशी (Chemical Protocol)**:\n"
                    f"   - {chem}\n"
                    f"   - **स्प्रे का समय**: सुबह 9 बजे से पहले या शाम को तेज धूप ढलने के बाद स्प्रे करें।\n"
                    f"2. **जैविक विकल्प (Organic Remedy)**:\n"
                    f"   - {org}\n"
                    f"3. **रोकथाम और प्रबंधन**:\n"
                    f"   - {prev}"
                )
            elif language == "Bengali":
                return (
                    f"### 💊 **{d_name} ({crop})** এর সঠিক প্রতিকার নির্দেশিকা\n\n"
                    f"**জীবাণু (Pathogen)**: *{pathogen}*\n\n"
                    f"1. **রাসায়নিক স্প্রে (Chemical Spray)**:\n"
                    f"   - {chem}\n"
                    f"2. **জৈব সমাধান (Organic Remedy)**:\n"
                    f"   - {org}\n"
                    f"3. **প্রতিরোধ ও পরিচর্যা**:\n"
                    f"   - {prev}"
                )
            return (
                f"### 💊 Recommended Treatment Protocol for **{d_name}** ({crop})\n\n"
                f"- **Causal Pathogen**: *{pathogen}*\n"
                f"- **Optimal Spore Germination Window**: Temperature {temp_min}–{temp_max}°C, Humidity > {hum_min}%\n\n"
                f"#### 1. Targeted Fungicidal Spray Protocol:\n"
                f"{chem}\n\n"
                f"#### 2. Organic & Biological Alternatives:\n"
                f"{org}\n\n"
                f"#### 3. Cultural Sanitation & Prevention:\n"
                f"{prev}"
            )

        # Specific: Organic remedies
        if any(k in q_low for k in ["organic", "natural", "home", "bio", "neem", "jaivik"]):
            if language == "Hindi":
                return (
                    f"### 🌿 **{d_name} ({crop})** के लिए प्राकृतिक व जैविक समाधान\n\n"
                    f"1. **नीम का तेल (Neem Oil Spray)**: 5 मिली शुद्ध कोल्ड-प्रेस्ड नीम तेल + 1 मिली तरल साबुन को 1 लीटर गुनगुने पानी में घोलकर हर 5-7 दिन पर पत्तियों के दोनों तरफ स्प्रे करें।\n"
                    f"2. **जैविक उपचार**: {org}\n"
                    f"3. **स्वच्छता**: {prev}"
                )
            return (
                f"### 🌿 Organic & Biological Solutions for **{d_name}** ({crop})\n\n"
                f"1. **Cold-Pressed Neem Oil Spray**: Mix 5 ml pure neem oil (10,000 ppm Azadirachtin) + 1 ml mild liquid soap in 1 liter of lukewarm water. Spray thoroughly over both leaf surfaces every 5–7 days.\n"
                f"2. **Biological Remedy**: {org}\n"
                f"3. **Cultural Sanitation**: {prev}"
            )

        # Specific: Causes / Symptoms
        if any(k in q_low for k in ["cause", "symptom", "identify", "why", "karan", "lakshan", "what is"]):
            return (
                f"### 🔍 Pathology & Symptom Breakdown for **{d_name}** ({crop})\n\n"
                f"- **Causal Pathogen**: *{pathogen}*\n"
                f"- **Environmental Trigger**: Spreads rapidly when temperatures range between **{temp_min}–{temp_max}°C** with relative humidity above **{hum_min}%**.\n\n"
                f"#### Management Strategy:\n"
                f"- **Fungicide Option**: {chem}\n"
                f"- **Organic Remedy**: {org}\n"
                f"- **Field Sanitation**: {prev}"
            )

        # Default comprehensive disease profile
        # Only attach scan measurements if this profile corresponds to the active scan!
        scan_metric = ""
        if not user_asked_different_crop and aff_pct is not None:
            scan_metric = f"\n- **Estimated Affected Area (Grad-CAM)**: **{aff_pct}%** (Confidence: {confidence_pct}%)"

        if language == "Hindi":
            return (
                f"### 📋 **{d_name}** ({crop}) की विस्तृत रिपोर्ट\n"
                f"{scan_metric}\n\n"
                f"- **रोगकारक**: *{pathogen}*\n"
                f"- **रासायनिक छिड़काव**: {chem}\n"
                f"- **जैविक उपचार**: {org}\n"
                f"- **बचाव**: {prev}"
            )
        elif language == "Bengali":
            return (
                f"### 📋 **{d_name}** ({crop}) এর বিস্তারিত রিপোর্ট\n"
                f"{scan_metric}\n\n"
                f"- **জীবাণু**: *{pathogen}*\n"
                f"- **প্রতিকার ও স্প্রে**: {chem}\n"
                f"- **জৈব সমাধান**: {org}\n"
                f"- **প্রতিরোধ**: {prev}"
            )
        return (
            f"### 🌿 **{d_name}** ({crop})\n"
            f"- **Causal Pathogen**: *{pathogen}*{scan_metric}\n\n"
            f"#### 1. Primary Action & Spray Protocol:\n"
            f"{chem}\n\n"
            f"#### 2. Organic Care:\n"
            f"{org}\n\n"
            f"#### 3. Prevention & Hygiene:\n"
            f"{prev}"
        )

    # 7. Query was about a crop without specific disease (e.g. "tell me about orange crop")
    if query_crop:
        c_title = query_crop.title()
        return (
            f"### 🍊 Overview of **{c_title}** Crop Pathology & Care\n\n"
            f"Common challenges in {c_title} cultivation include foliar infections and nutrient imbalances. "
            f"For optimal {c_title} vigor:\n"
            f"1. **Monitoring**: Inspect leaf undersides and new flush growth every 3–5 days.\n"
            f"2. **Disease Prevention**: Apply protective copper-based sprays (Copper Oxychloride 3 g/L) before prolonged rain cycles.\n"
            f"3. **Irrigation**: Ensure excellent root drainage and avoid standing water at the tree collar.\n\n"
            f"You can upload a photo of any {c_title} leaf in the **Scan** tab to run deep learning disease diagnosis!"
        )

    # 8. General fallback
    if language == "Hindi":
        return (
            "धन्यवाद! मैं आपका डिजिटल कृषि सलाहकार हूँ। "
            "आप किसी भी फसल (टमाटर, संतरा, सेब, आलू, मक्का), कवकनाशी स्प्रे, जैविक नीम तेल उपचार, या मौसम के बारे में पूछ सकते हैं, "
            "या 'Scan' टैब में जाकर किसी भी पत्ती का फोटो स्कैन करवा सकते हैं।"
        )
    return (
        "Thank you for reaching out! As your precision agricultural pathologist, I can assist with any crop disease, "
        "fungicide spray dosage, organic remedies, irrigation, and field management.\n\n"
        "Feel free to ask a specific question (e.g. *'How to treat orange citrus greening?'* or *'What is the current temperature?'*), "
        "or upload a leaf scan in the Scan tab to get an instant neural diagnosis and Grad-CAM heatmap analysis!"
    )
