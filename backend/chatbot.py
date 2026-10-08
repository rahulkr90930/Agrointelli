"""
AgroIntelli Agronomic Chatbot Engine
Multimodal agricultural intelligence powered by Google Gemini (when configured with an AIzaSy key)
and an instant, highly detailed localized agronomic expert engine grounded in the 39-disease
pathology knowledge base, live weather telemetry, and Grad-CAM spatial metrics.
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
IS_VALID_GEMINI_KEY = bool(GEMINI_API_KEY and GEMINI_API_KEY.startswith("AIzaSy"))

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
if IS_VALID_GEMINI_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_client = genai
        print("🌿 AgroIntelli Chatbot initialized with Google Gemini API.")
    except Exception as e:
        print(f"ℹ️ Google Gemini SDK notice: {e}. Local agronomic expert active.")
else:
    print("🌿 AgroIntelli Chatbot initialized in ultra-fast local agronomic expert mode.")

SYSTEM_PROMPT = """You are AgroIntelli AI (AgroBot), an intelligent, friendly, and practical agricultural pathologist and precision crop consultant.

BEHAVIOR GUIDELINES:
1. ADDRESS THE USER'S SPECIFIC QUESTION DIRECTLY:
   - Always answer what the user actually asked first and foremost.
   - For simple greetings ("hi", "hello", "hey", "who are you"), reply with a warm, natural 1-2 sentence welcome.
   - For general agricultural questions (irrigation, soil, fertilization, sunlight, pruning), provide clear, practical horticultural advice.
2. CONTEXT AWARENESS:
   - When the user asks about diagnosing a leaf or treating an infection, leverage the provided telemetry:
     * Diagnostic prediction & confidence
     * Grad-CAM lesion affected percentage (% of leaf blade)
     * Microclimate weather (temperature, humidity %, rainfall)
     * Scientific disease profile (pathogen, chemical dosage, organic remedies)
3. PRACTICAL DOSAGES:
   - Provide standard concentrations (e.g., Mancozeb 2-2.5 g/L, Chlorothalonil 2 ml/L, Copper Oxychloride 3 g/L, cold-pressed neem oil 5 ml/L).
4. TONE & STRUCTURE:
   - Natural, conversational, structured with bullet points and bold highlights.
"""

FAST_GEN_CONFIG = {
    "max_output_tokens": 800,
    "temperature": 0.3,
}


def _match_disease_from_query(query):
    """Identifies mentioned crop diseases from user text if no active scan is attached."""
    if not query or not KNOWLEDGE_STORE:
        return None, None
    q = query.lower().replace("-", " ").replace("_", " ")

    # Direct profile matches
    for k, v in KNOWLEDGE_STORE.items():
        name_clean = k.replace("_", " ")
        common = v.get("common_name", "").lower()
        if name_clean in q or (common and common in q):
            return k, v

    # Symptom / disease keywords
    keyword_map = {
        "early blight": "tomato_early_blight",
        "late blight": "tomato_late_blight",
        "bacterial spot": "pepper_bacterial_spot",
        "apple scab": "apple_apple_scab",
        "black rot": "apple_black_rot",
        "cedar apple rust": "apple_cedar_apple_rust",
        "common rust": "corn_common_rust",
        "gray leaf spot": "corn_gray_leaf_spot",
        "powdery mildew": "squash_powdery_mildew",
        "leaf mold": "tomato_leaf_mold",
        "septoria": "tomato_septoria_leaf_spot",
        "spider mite": "tomato_spider_mites_two_spotted_spider_mite",
        "yellow leaf curl": "tomato_yellow_leaf_curl_virus",
        "mosaic virus": "tomato_mosaic_virus",
        "citrus greening": "orange_haunglongbing_citrus_greening",
        "scorch": "strawberry_leaf_scorch",
    }
    for kw, key in keyword_map.items():
        if kw in q and key in KNOWLEDGE_STORE:
            return key, KNOWLEDGE_STORE[key]

    # Crop mentions
    crops = ["tomato", "apple", "corn", "potato", "grape", "peach", "pepper", "strawberry", "squash", "cherry"]
    for crop in crops:
        if crop in q:
            for k, v in KNOWLEDGE_STORE.items():
                if k.startswith(crop) and "healthy" not in k:
                    return k, v
    return None, None


def generate_chat_response(user_message, scan_context=None, history=None, language="English"):
    """
    Generates an intelligent, conversational agronomic response grounded in scan telemetry,
    pathology knowledge base, and live weather. Responds in < 50ms locally or via Gemini if valid key exists.
    """
    if not user_message or not user_message.strip():
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

    # Extract context
    scan_ctx = scan_context or {}
    pred = scan_ctx.get("prediction")
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

    # If no scan context is provided, check if the question mentions a disease
    if not knowledge and not pred:
        matched_key, matched_record = _match_disease_from_query(user_message)
        if matched_record:
            pred = matched_key
            knowledge = matched_record

    # If valid Gemini key exists, try Gemini Flash with rapid 4-second timeout
    if IS_VALID_GEMINI_KEY and _gemini_client is not None:
        try:
            weather_desc = f"{weather.get('temp_c', 'N/A')}°C, {weather.get('humidity_pct', 'N/A')}% humidity, {weather.get('condition', 'N/A')}"
            has_scan = bool(pred and pred != "None active")
            k_sum = ""
            if knowledge:
                k_sum = f"Pathogen: {knowledge.get('pathogen', 'N/A')}\nChemical: {knowledge.get('treatment_protocol', 'N/A')}\nOrganic: {knowledge.get('organic_remedies', 'N/A')}\nPrevention: {knowledge.get('prevention', 'N/A')}"

            prompt = (
                f"{SYSTEM_PROMPT}\n\n[CONTEXT]\n"
                f"- Scan: {pred if has_scan else 'None'}\n"
                f"- Crop: {plant_name or 'Not identified'}\n"
                f"- Confidence: {scan_ctx.get('confidence_pct', 'N/A')}%\n"
                f"- Estimated affected leaf area: {aff_pct if aff_pct is not None else 'N/A'}%\n"
                f"- Weather: {weather_desc}\n{k_sum}\n\n"
                f"[USER QUESTION]\n{user_message}\n\n"
                f"Answer this question directly in 2-5 concise sentences. Mention scan measurements only when relevant. "
                f"Do not use markdown heading markers (#). Avoid generic advice and do not invent values.\n"
                f"[TARGET LANGUAGE]: {language}"
            )
            cand = _gemini_client.GenerativeModel("models/gemini-2.0-flash", generation_config=FAST_GEN_CONFIG)
            res = cand.generate_content(prompt, request_options={"timeout": 4.0})
            if res and res.text:
                return {"reply": res.text.strip(), "grounded": True, "model_used": "Gemini 2.0 Flash"}
        except Exception:
            pass

    # Instant, deeply-grounded localized agronomic expert
    reply = _build_local_grounded_reply(
        user_message, pred, aff_pct, weather, knowledge,
        language=language, is_historical=is_historical, plant_name=plant_name,
        mongo_summary=mongo_summary, confidence_pct=conf
    )
    return {
        "reply": reply,
        "grounded": True,
        "model_used": "AgroIntelli Local Expert"
    }


def _build_local_grounded_reply(query, pred, aff_pct, weather, knowledge, language="English", is_historical=False, plant_name="", mongo_summary="", confidence_pct=0):
    """
    Instant, conversational, comprehensive agronomic response engine.
    Extracts disease information from knowledge base, provides exact dosages,
    and tailors answers to the user's specific intent.
    """
    q_low = query.lower().strip()

    # 1. Greetings
    if any(q_low.startswith(g) or q_low == g for g in ["hi", "hello", "hey", "namaste", "hola", "namoshkar", "who are you", "what is your name"]):
        if language == "Hindi":
            return (
                "नमस्ते! मैं **एग्रोबॉट (AgroBot)** हूँ, आपका डिजिटल कृषि सहायक और फसल रोग विशेषज्ञ। 🌾\n\n"
                "आप पत्ती की फोटो अपलोड करके बीमारी की सटीक जांच कर सकते हैं, या मुझसे किसी भी फसल (टमाटर, आलू, सेब, मक्का), "
                "रोग उपचार, जैविक कीटनाशक (नीम तेल) और सिंचाई के बारे में पूछ सकते हैं। आज मैं आपकी क्या मदद कर सकता हूँ?"
            )
        elif language == "Bengali":
            return (
                "নমস্কার! আমি **এগ্রোবট (AgroBot)**, আপনার ডিজিটাল কৃষি সহকারী এবং ফসল বিশেষজ্ঞ। 🌾\n\n"
                "আপনি পাতার ছবি আপলোড করে রোগ নির্ণয় করতে পারেন, অথবা ফসল, সার, সেচ ও কীটনাশক সংক্রান্ত যেকোনো প্রশ্ন করতে পারেন। আমি কীভাবে আপনাকে সাহায্য করতে পারি?"
            )
        return (
            "Hello! I am **AgroIntelli AI (AgroBot)**, your intelligent precision agricultural pathologist and crop health consultant. 🌾\n\n"
            "You can upload a crop leaf photo for instant disease diagnosis and Grad-CAM lesion analysis, "
            "or ask me any questions about crop treatments, organic sprays (like neem oil), irrigation, and soil management. How can I help you today?"
        )

    # 2. Irrigation / Watering
    if any(k in q_low for k in ["water", "watering", "irrigation", "sinchai", "paani", "jol", "how often to water"]):
        if language == "Hindi":
            return (
                "### 💧 सिंचाई और जल प्रबंधन के महत्वपूर्ण नियम\n\n"
                "1. **सिंचाई विधि**: पत्तियों पर ऊपर से पानी छिड़कने से बचें। हमेशा **ड्रिप सिंचाई (Drip)** या जड़ों के पास पानी दें, ताकि फफूंद न फैले।\n"
                "2. **सही समय**: सिंचाई हमेशा **सुबह 6:00 से 9:00 बजे** के बीच करें ताकि पत्तियों पर पड़ी ओस और पानी धूप में जल्दी सूख जाए।\n"
                "3. **नमी जांच**: पानी देने से पहले मिट्टी में 2 इंच गहराई पर उंगली डालकर नमी जांचें। यदि मिट्टी नम है, तो सिंचाई रोक दें।"
            )
        return (
            "### 💧 Practical Irrigation & Moisture Management Guidelines\n\n"
            "1. **Root-Zone Delivery**: Always water at the base of the plant using **drip irrigation or ground soaker hoses**. Avoid overhead sprinklers, as wet foliage triggers fungal spore germination.\n"
            "2. **Timing**: Water **early in the morning (6:00 AM – 9:00 AM)** so that any accidental splash on lower foliage dries rapidly under morning sunlight.\n"
            "3. **Frequency**: Provide deep watering 2–3 times per week rather than shallow daily watering to encourage deep, drought-resilient root architecture.\n"
            "4. **Moisture Check**: Verify soil moisture 2 inches beneath the surface before irrigating."
        )

    # 3. Soil, Compost & Fertilizer
    if any(k in q_low for k in ["soil", "fertilizer", "fertiliser", "khad", "npk", "compost", "manure", "mati"]):
        if language == "Hindi":
            return (
                "### 🌱 मिट्टी का स्वास्थ्य और संतुलित पोषक तत्व\n\n"
                "1. **जैविक खाद**: बुवाई से पहले 2-3 टन प्रति एकड़ अच्छी सड़ी हुई गोबर की खाद या वर्मीकंपोस्ट मिट्टी में मिलाएं।\n"
                "2. **संतुलित NPK**: अत्यधिक नाइट्रोजन (यूरिया) देने से पौधे कोमल हो जाते हैं और फफूंद रोग तेजी से पकड़ते हैं। पोटाश (K) और फास्फोरस (P) का संतुलन बनाए रखें।\n"
                "3. **मल्चिंग**: पौधों की जड़ों के पास 2-3 इंच सूखी घास की परत (मल्च) लगाएं ताकि नमी बनी रहे।"
            )
        return (
            "### 🌱 Soil Health & Balanced Crop Nutrition\n\n"
            "1. **Organic Enrichment**: Incorporate well-decomposed vermicompost or farmyard manure before planting to enhance beneficial soil microbes.\n"
            "2. **Balanced N-P-K Ratio**: Avoid excessive nitrogen (Urea), which causes lush, soft tissue highly vulnerable to fungal blight. Ensure adequate Potassium (K) to reinforce plant cell walls.\n"
            "3. **Soil Aeration & Mulch**: Apply a 2-inch organic mulch layer around the root zone to prevent soil splash and conserve moisture."
        )

    # 4. Check if a disease is available (either from active scan or query match)
    if not knowledge:
        k_key, k_val = _match_disease_from_query(query)
        if k_val:
            knowledge = k_val
            pred = k_key

    # If we have disease knowledge (from scan OR query), provide rich specific answer!
    if knowledge:
        crop = knowledge.get("crop", "Crop")
        d_name = knowledge.get("common_name", pred.replace("_", " ").title() if pred else "Plant Disease")
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
                    f"1. **रासायनिक कवकनाशी (Chemical Treatment)**:\n"
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
                    f"1. **রাসায়নিক ছত্রাকনাশক (Chemical Spray)**:\n"
                    f"   - {chem}\n"
                    f"   - **স্প্রে করার সময়**: সকালে কড়া রোদ ওঠার আগে অথবা বিকেলে স্প্রে করুন।\n"
                    f"2. **জৈব সমাধান (Organic Remedy)**:\n"
                    f"   - {org}\n"
                    f"3. **প্রতিরোধ ও পরিচর্যা**:\n"
                    f"   - {prev}"
                )
            return (
                f"### 💊 Recommended Treatment Protocol for **{d_name}** ({crop})\n\n"
                f"- **Causal Pathogen**: *{pathogen}*\n"
                f"- **Optimal Environmental Window**: Temperature {temp_min}–{temp_max}°C, Humidity > {hum_min}%\n\n"
                f"#### 1. Chemical Fungicide / Bactericide Protocol:\n"
                f"{chem}\n\n"
                f"#### 2. Organic & Biological Alternatives:\n"
                f"{org}\n\n"
                f"#### 3. Cultural & Preventative Sanitation:\n"
                f"{prev}"
            )

        # Specific: Organic remedies
        if any(k in q_low for k in ["organic", "natural", "home", "bio", "neem", "jaivik"]):
            if language == "Hindi":
                return (
                    f"### 🌿 **{d_name}** के लिए प्राकृतिक व जैविक समाधान\n\n"
                    f"1. **नीम का तेल (Neem Oil Spray)**: 5 मिली शुद्ध कोल्ड-प्रेस्ड नीम तेल + 1 मिली तरल साबुन को 1 लीटर गुनगुने पानी में अच्छी तरह मिलाकर हर 5-7 दिन पर पत्तियों के दोनों तरफ स्प्रे करें।\n"
                    f"2. **जैविक उपचार**: {org}\n"
                    f"3. **स्वच्छता**: संक्रमित पत्तियों को तोड़कर खेत से दूर नष्ट कर दें।"
                )
            elif language == "Bengali":
                return (
                    f"### 🌿 **{d_name}** এর প্রাকৃতিক ও জৈব প্রতিকার\n\n"
                    f"1. **নিম তেলের স্প্রে**: ১ লিটার জলে ৫ মিলি খাঁটি কোল্ড-প্রেসড নিম তেল ও ১ মিলি হালকা তরল সাবান মিশিয়ে প্রতি ৫-৭ দিন অন্তর পাতার উভয় পিঠে স্প্রে করুন।\n"
                    f"2. **জৈব নিয়ন্ত্রণ**: {org}\n"
                    f"3. **পরিচ্ছন্নতা**: আক্রান্ত পাতা তুলে পুড়িয়ে বা মাটি চাপা দিয়ে ধ্বংস করুন।"
                )
            return (
                f"### 🌿 Organic & Biological Solutions for **{d_name}**\n\n"
                f"1. **Cold-Pressed Neem Oil Spray**: Mix 5 ml pure neem oil (10,000 ppm Azadirachtin) + 1 ml mild dish soap in 1 liter of lukewarm water. Spray thoroughly over upper and lower leaf surfaces every 5–7 days.\n"
                f"2. **Bio-Control Remedy**: {org}\n"
                f"3. **Cultural Sanitation**: {prev}"
            )

        # Specific: Causes / Symptoms / Identification
        if any(k in q_low for k in ["cause", "symptom", "identify", "why", "karan", "lakshan", "what is"]):
            if language == "Hindi":
                return (
                    f"### 🔍 **{d_name}** के लक्षण और कारण\n\n"
                    f"- **रोगकारक**: *{pathogen}*\n"
                    f"- **अनुकूल मौसम**: यह रोग **{temp_min}°C से {temp_max}°C** तापमान और **{hum_min}% से अधिक नमी** में सबसे तेजी से फैलता है।\n"
                    f"- **संकेत**: पत्तियों पर भूरे/काले धब्बे, पीलापन और ऊतकों का सूखना।\n\n"
                    f"**उपचार**: {chem}\n"
                    f"**जैविक उपाय**: {org}"
                )
            elif language == "Bengali":
                return (
                    f"### 🔍 **{d_name}** এর লক্ষণ ও কারণ\n\n"
                    f"- **জীবাণু**: *{pathogen}*\n"
                    f"- **অনুকূল আবহাওয়া**: তাপমাত্রা **{temp_min}°C থেকে {temp_max}°C** এবং **{hum_min}% এর বেশি আর্দ্রতায়** রোগ দ্রুত ছড়ায়।\n"
                    f"- **লক্ষণ**: পাতায় বাদামী বা কালো দাগ, হলদে ভাব ও শুকিয়ে যাওয়া।\n\n"
                    f"**রাসায়নিক ব্যবস্থা**: {chem}\n"
                    f"**জৈব ব্যবস্থা**: {org}"
                )
            return (
                f"### 🔍 Pathology & Symptom Breakdown for **{d_name}**\n\n"
                f"- **Causal Pathogen**: *{pathogen}*\n"
                f"- **Environmental Triggers**: Spreads rapidly when temperatures range between **{temp_min}–{temp_max}°C** with relative humidity above **{hum_min}%**.\n"
                f"- **Visual Symptoms**: Typical presentation includes circular to irregular necrotic brown lesions, concentric rings, marginal chlorosis, or powdery fungal sporulation on leaf surfaces.\n\n"
                f"#### Immediate Action Plan:\n"
                f"- **Target Treatment**: {chem}\n"
                f"- **Organic Option**: {org}\n"
                f"- **Sanitation**: {prev}"
            )

        # Default comprehensive disease profile
        if language == "Hindi":
            return (
                f"### 📋 **{d_name}** ({crop}) की पूरी रिपोर्ट\n\n"
                f"- **रोगकारक**: *{pathogen}*\n"
                f"- **रासायनिक छिड़काव**: {chem}\n"
                f"- **जैविक उपचार**: {org}\n"
                f"- **बचाव**: {prev}"
            )
        elif language == "Bengali":
            return (
                f"### 📋 **{d_name}** ({crop}) এর বিস্তারিত রিপোর্ট\n\n"
                f"- **জীবাণু**: *{pathogen}*\n"
                f"- **প্রতিকার ও স্প্রে**: {chem}\n"
                f"- **জৈব সমাধান**: {org}\n"
                f"- **প্রতিরোধ**: {prev}"
            )
        return (
            f"{crop} scan: {d_name} is associated with {pathogen}. "
            f"The affected area is {aff_pct}%"
            f"{f' at {confidence_pct}% model confidence' if confidence_pct else ''}.\n\n"
            f"First action: {chem}\n"
            f"Prevention: {prev}\n"
            f"Organic option: {org}"
            if aff_pct is not None
            else (
                f"{crop}: {d_name} is associated with {pathogen}. "
                f"Treatment: {chem} Prevention: {prev} Organic option: {org}"
            )
        )

    # General agronomy inquiry fallback
    if language == "Hindi":
        return (
            "धन्यवाद! मैं आपका डिजिटल फसल सलाहकार हूँ। "
            "आप किसी भी फसल रोग (जैसे *'टमाटर अगेती झुलसा का इलाज क्या है?'*), कवकनाशी स्प्रे, या जैविक नीम तेल उपचार के बारे में सवाल पूछ सकते हैं "
            "या पत्ती की फोटो अपलोड करके तुरंत जांच करवा सकते हैं।"
        )
    elif language == "Bengali":
        return (
            "ধন্যবাদ! আমি আপনার কৃষি সহায়ক এআই। "
            "আপনি যেকোনো ফসলের রোগ (যেমন *'টমেটোর আর্লি ব্লাইটের চিকিৎসা কী?'*), ছত্রাকনাশক স্প্রে বা জৈব নিম তেলের প্রতিকার সম্পর্কে প্রশ্ন করতে পারেন "
            "অথবা পাতার ছবি আপলোড করে তাৎক্ষণিক রোগ নির্ণয় করতে পারেন।"
        )
    return (
        "Thank you for reaching out! As your precision agricultural pathologist, I can assist with any crop disease, "
        "fungicide spray dosage, organic remedies, irrigation, and field management.\n\n"
        "Feel free to ask a specific question (e.g. *'How to treat tomato early blight?'* or *'What fungicide cures apple scab?'*), "
        "or upload a leaf scan to get an instant neural diagnosis and Grad-CAM lesion analysis!"
    )
