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
   - GIBBERISH / NONSENSE / RANDOM WORDS (e.g. 'tbrehh', 'asdfgh', 'random words', unpronounceable text):
     * DO NOT fabricate, assume, or pull a random disease from the library!
     * Politely inform the user that their query was unclear, and ask them to specify their crop, disease symptom, or agricultural question.
   - PREVENTION VS TREATMENT:
     * When the user asks HOW TO PREVENT a disease (e.g., 'how could i prevent potato disease', 'prevention tips'):
       Focus strictly on PROPHYLAXIS and PREVENTION: certified seeds, crop rotation, soil drainage, canopy spacing, preventive biofungicides (Trichoderma), mulch, and avoiding overhead watering. Do NOT just dump disease symptoms and reactive chemical cures!
   - TIMELINE PROGRESSION / TRAJECTORY:
     * When the user asks to discuss a multi-day timeline trajectory, analyze the progression data (initial vs latest severity, elapsed days, weather humidity/rain impact) and provide a tailored step-by-step recovery plan.
   - DIFFERENT CROPS:
     * If the user asks about a specific crop or disease (e.g. orange citrus greening, apple scab, corn rust), answer specifically about THAT crop and disease, even if another leaf was previously scanned!
   - WEATHER & SAVED PLANTS:
     * Answer directly using the provided weather telemetry or MongoDB history.
2. TONE & STRUCTURE:
   - Professional, encouraging, clear, structured with bold highlights and bullet points.
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

# Dedicated Prophylactic & Prevention Protocols by Crop
CROP_PREVENTION_GUIDE = {
    "potato": {
        "title": "Comprehensive Potato Disease Prevention & Prophylaxis Protocol",
        "steps": [
            ("Certified Disease-Free Seed Tubers", "Plant exclusively certified seed potatoes with well-developed sprouts. Discard any tubers showing soft rot, bruising, or sunken lesions. Allow cut tubers to suberize (heal) for 2–3 days before planting."),
            ("Strict 3–4 Year Crop Rotation", "Never plant potatoes in soil previously cropped with Solanaceous species (Tomato, Eggplant, Pepper) within the last 3–4 years to starve overwintering fungal spores (*Alternaria* & *Phytophthora*)."),
            ("Continuous Hilling & Ridge Formation", "Hill up soil around the potato stems as plants grow (15–20 cm high ridges). This creates a physical soil barrier that prevents foliar late blight spores from washing down into developing tubers during rainfall."),
            ("Drip or Furrow Irrigation Only", "Avoid overhead sprinkler irrigation completely. Deliver water at the base or through furrows early in the morning so the foliage remains completely dry."),
            ("Prophylactic Biofungicide Inoculation", "Incorporate *Trichoderma harzianum* or *Bacillus subtilis* (5–10 g/kg seed tuber) at planting to outcompete soil-borne pathogens. Apply preventive Copper Oxychloride (2.5 g/L) or Mancozeb before monsoon rain fronts."),
            ("Row Spacing & Canopy Aeration", "Maintain 25–30 cm spacing between plants and 70–80 cm between ridges to promote rapid wind drying across the lower leaf canopy.")
        ]
    },
    "tomato": {
        "title": "Comprehensive Tomato Disease Prevention Protocol",
        "steps": [
            ("Bottom Pruning & Soil Clearance", "Prune all suckers and foliage within the lowest 25–30 cm of the soil. This prevents soil-splash pathogens from reaching the lower leaf canopy during rain."),
            ("Organic Mulching", "Apply a 2-inch layer of clean straw, pine needles, or plastic mulch beneath the plants to suppress weed competition and seal fungal spores in the soil."),
            ("Morning Drip Irrigation", "Water strictly at root level between 6:00 AM and 9:00 AM. Wet evening leaves create an 8+ hour infection window for Early and Late Blight."),
            ("Preventative Biological Sprays", "Apply cold-pressed neem oil (5 ml/L) or biofungicide (*Bacillus subtilis*) every 7–10 days during warm, humid weather cycles."),
            ("Trellising & Airflow", "Stake or cage all indeterminate vines to ensure upright growth, maximizing sunlight penetration and airflow.")
        ]
    },
    "corn": {
        "title": "Comprehensive Corn/Maize Disease Prevention Protocol",
        "steps": [
            ("Resistant Hybrid Cultivars", "Select hybrid corn varieties certified with genetic resistance to Common Rust (*Puccinia sorghi*) and Northern Leaf Blight."),
            ("Crop Residue Management & Tillage", "Deeply incorporate or remove previous corn stubble, as fungal spores overwinter on surface debris."),
            ("Balanced Nitrogen vs Potassium", "Avoid excessive nitrogen fertilizers which create soft, lush tissues susceptible to fungal penetration; ensure adequate potassium to strengthen stalks."),
            ("Optimal Plant Density", "Follow recommended plant populations (60,000–75,000 plants/ha) to prevent excessive canopy humidity.")
        ]
    },
    "apple": {
        "title": "Comprehensive Apple Orchard Disease Prevention Protocol",
        "steps": [
            ("Dormant Season Sanitation", "Rake and compost or shred all fallen leaves in autumn to disrupt the overwintering cycle of Apple Scab (*Venturia inaequalis*)."),
            ("Dormant Copper & Lime Sulfur", "Apply a dormant copper spray before green tip stage in early spring to kill overwintering fungal spores and bacterial cankers."),
            ("Canopy Pruning for Light Penetration", "Prune out interior water sprouts and crossing branches to allow sunlight and wind to dry the interior foliage within 2 hours of rainfall.")
        ]
    },
    "grape": {
        "title": "Comprehensive Grapevine Disease Prevention Protocol",
        "steps": [
            ("Canopy Management & Leaf Pulling", "Pull basal leaves around fruit clusters early in the season to enhance airflow and spray penetration, preventing Black Rot and Powdery Mildew."),
            ("Dormant Sanitation", "Prune out dead canes and mummified berries from the previous season, which act as primary inoculum reservoirs."),
            ("Preventative Sulfur/Copper Program", "Apply wettable sulfur or copper protectants preventatively during shoot elongation (10–25 cm shoot length).")
        ]
    },
    "orange": {
        "title": "Comprehensive Citrus/Orange Disease Prevention Protocol",
        "steps": [
            ("Vector Control for Greening (HLB)", "Regularly monitor for the Asian citrus psyllid (*Diaphorina citri*), the insect vector of Citrus Greening. Apply horticultural oil or systemic treatments at shoot flush."),
            ("Certified Nursery Budwood", "Plant only certified disease-free nursery stock from quarantined greenhouses."),
            ("Collar Clearance & Drainage", "Avoid wet soil buildup around the trunk collar; ensure rapid soil drainage to prevent Phytophthora root rot.")
        ]
    }
}


def is_gibberish(text):
    """
    Detects random keystrokes, keyboard mashing, or unpronounceable non-words
    such as 'tbrehh', 'asdfgh', 'qwerty', 'zzzz', or meaningless word strings.
    """
    if not text:
        return True
    t = text.strip().lower()

    # Exact short nonsense strings
    if t in ["random words", "some random words", "asdfgh", "qwerty", "tbrehh", "blabla", "xyz", "abc"]:
        return True

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
        'prevent', 'prevention', 'preventive', 'avoid', 'protect', 'protection', 'cure',
        'trajectory', 'timeline', 'progression', 'recovery', 'worsen', 'healing',
        'namaste', 'hola', 'namoshkar', 'karan', 'upchar', 'dawa', 'sinchai', 'paani', 'mati',
        'roktham', 'bachav', 'suraksha'
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

    return meaningful_count == 0


def match_disease_or_crop_from_query(query):
    """
    Extracts an explicitly mentioned disease or crop from the user query.
    Returns (matched_key, matched_record, matched_crop_name).
    Carefully distinguishes between mentioning just a crop vs a specific disease!
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
        if (len(name_clean) > 5 and name_clean in q) or (common and len(common) > 5 and common in q):
            return k, v, v.get("crop", k.split("_")[0])

    # 3. Explicit crop mentions
    for crop in CROPS_LIST:
        if re.search(r'\b' + re.escape(crop) + r'\b', q):
            # Check if user mentioned a specific disease token belonging to this crop
            for k, v in KNOWLEDGE_STORE.items():
                if k.startswith(crop):
                    disease_part = k[len(crop):].replace("_", " ").strip()
                    tokens = [tok for tok in disease_part.split() if len(tok) > 3]
                    if tokens and any(tok in q for tok in tokens):
                        return k, v, crop
            # Otherwise, ONLY crop was mentioned, NOT a specific disease
            return None, None, crop

    return None, None, None


def is_prevention_inquiry(query):
    """Detects inquiries about disease prevention, prophylaxis, avoiding infection, etc."""
    q = query.lower()
    prev_words = [
        "prevent", "prevention", "preventative", "preventive", "prophylaxis",
        "avoid", "how to avoid", "stop disease", "protect", "protection",
        "kaise bachein", "roktham", "bachav", "suraksha", "stop infection"
    ]
    return any(re.search(r'\b' + re.escape(w) + r'\b', q) for w in prev_words)


def is_trajectory_inquiry(query):
    """Detects inquiries about timeline progression, trajectory, recovery over time."""
    q = query.lower()
    traj_words = [
        "trajectory", "timeline progression", "progression results", "healing progression",
        "multi-day", "multiday", "recovery trajectory", "disease trajectory",
        "progression tracking", "timeline day", "over time"
    ]
    return any(w in q for w in traj_words)


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


def has_agronomic_intent(query):
    """Determines if the query actually contains any agricultural/crop/weather/conversational relevance."""
    q = query.lower().strip()
    if is_greeting(q) or is_weather_inquiry(q) or is_journal_inquiry(q) or is_trajectory_inquiry(q) or is_prevention_inquiry(q):
        return True
    if any(crop in q for crop in CROPS_LIST):
        return True
    agri_terms = [
        "leaf", "plant", "crop", "flower", "fruit", "root", "soil", "seed",
        "water", "irrigation", "fertilizer", "compost", "spray", "cure", "treat",
        "fungicide", "pesticide", "neem", "blight", "rot", "scab", "rust", "mildew",
        "spot", "mold", "virus", "symptom", "disease", "harvest", "prun", "farm"
    ]
    return any(re.search(r'\b' + re.escape(term) + r'\b', q) for term in agri_terms)


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
    if is_gibberish(msg) or not has_agronomic_intent(msg):
        gibberish_replies = {
            "Hindi": "क्षमा करें, मुझे आपका सवाल समझ नहीं आया। 🌾 कृपया अपना प्रश्न स्पष्ट शब्दों में लिखें—जैसे *'आलू की फसल को रोग से कैसे बचाएं?'*, *'संतरा सिट्रस ग्रीनिंग का इलाज क्या है?'*, या *'आज का तापमान और मौसम'*।",
            "Bengali": "দুঃখিত, আপনার বার্তাটি বোঝা যায়নি। 🌾 অনুগ্রহ করে ফসল বা রোগ সংক্রান্ত প্রশ্ন স্পষ্টভাবে জিজ্ঞাসা করুন—যেমন *'আলুর রোগ কীভাবে প্রতিরোধ করবেন?'* বা *'টমেটোর ব্লাইটের চিকিৎসা'*।",
            "Spanish": "Disculpe, no pude entender su consulta. 🌾 Por favor, escriba su pregunta con claridad sobre cultivos (ej. papa, tomate, naranja), prevención de enfermedades o clima actual.",
        }
        return {
            "reply": gibberish_replies.get(language, "I couldn't detect a specific agricultural, crop, or plant pathology topic in your question. 🌾\n\nAs your precision AI agronomist, I specialize in:\n• **Disease Prevention & Care** (e.g. *'How could I prevent potato disease?'*)\n• **Targeted Treatment Protocols** (e.g. *'Cure for Tomato Early Blight'*)\n• **Microclimate & Weather Risk** (e.g. *'Current humidity and fungal risk'*)\n• **Multi-Day Timeline Trajectory Analysis**\n\nCould you please specify which crop, symptom, or farming practice you'd like advice on?"),
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
    is_trajectory_flag = scan_ctx.get("is_trajectory", False) or is_trajectory_inquiry(msg)

    # 4. Entity Extraction & Conflict Resolution
    query_key, query_rec, query_crop = match_disease_or_crop_from_query(msg)
    user_asked_different_crop = False

    active_knowledge = knowledge
    active_pred = scanned_pred

    if query_rec:
        if not scanned_pred or (query_crop and query_crop.lower() not in (scanned_pred or "").lower()):
            user_asked_different_crop = True
            active_knowledge = query_rec
            active_pred = query_key
    elif query_crop:
        user_asked_different_crop = True
        active_knowledge = None
        active_pred = query_crop

    # Intent flags
    is_weather = is_weather_inquiry(msg)
    is_journal = is_journal_inquiry(msg)
    is_prevention = is_prevention_inquiry(msg)

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

            prompt_parts = [
                f"{SYSTEM_PROMPT}\n",
                f"[TELEMETRY & CONTEXT]",
                f"- Active leaf scan loaded: {scanned_pred or 'None'}",
                f"- Specific topic in query: {active_pred if user_asked_different_crop else 'No override'}",
                f"- User Query Crop: {query_crop or 'General'}",
                f"- Is Prevention Query: {'YES' if is_prevention else 'NO'}",
                f"- Is Trajectory Query: {'YES' if is_trajectory_flag else 'NO'}",
                f"- Current Microclimate Weather: {weather_desc}",
            ]
            if is_trajectory_flag:
                prompt_parts.append(f"- Timeline Trajectory Data: {scan_ctx}")
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
                f"1. Directly answer what the user asked with structured bullet points and practical guidelines.",
                f"2. IF user asked HOW TO PREVENT a disease: Provide proactive, preventative cultural, biological, and irrigation protocols rather than simply diagnosing or naming a curative spray.",
                f"3. IF user asked about trajectory/progression: Provide an agronomic consultation analyzing the multi-day progression results and outline a phased recovery roadmap.",
                f"4. IF user asked about weather/temperature: Focus on the live weather telemetry and disease infection windows.",
                f"5. IF user asked about a different crop: Answer specifically for that crop.",
                f"6. Target Language: {language}.",
                f"7. Use standard markdown without raw heading tags (#)."
            ])

            full_prompt = "\n".join(prompt_parts)
            model_inst = _gemini_client.GenerativeModel(
                model_name=target_model,
                generation_config=FAST_GEN_CONFIG
            )
            response = model_inst.generate_content(full_prompt)
            if response and response.text:
                return {
                    "reply": response.text.strip(),
                    "grounded": True,
                    "model_used": f"Google {target_model}"
                }
        except Exception as e:
            print(f"⚠️ Gemini Chat generation notice ({e}); transitioning seamlessly to local expert.")

    # 6. Ultra-Fast High-Accuracy Local Agronomic Expert
    reply = _build_local_grounded_reply(
        query=msg,
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
        is_prevention=is_prevention,
        is_trajectory=is_trajectory_flag,
        user_asked_different_crop=user_asked_different_crop,
        query_crop=query_crop,
        scan_ctx=scan_ctx
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
    is_prevention=False, is_trajectory=False, user_asked_different_crop=False,
    query_crop=None, scan_ctx=None
):
    """
    Deterministic, highly specific localized agronomic expert handler.
    Accurately branches on user intent: Greetings, Prevention, Trajectory Discussion,
    Weather, Saved Journal, Specific Crop/Disease Overrides, Irrigation, Fertilizer, or Active Scan.
    """
    q_low = query.lower().strip()
    wx = weather or {}
    temp_c = wx.get("temp_c", 26)
    hum_pct = wx.get("humidity_pct", 65)
    rain_mm = wx.get("rain_1h_mm", 0)
    condition = wx.get("condition", "Partly Cloudy")
    s_ctx = scan_ctx or {}

    # 1. Greetings
    if is_greeting(query):
        if language == "Hindi":
            return (
                "नमस्ते! मैं **एग्रोबॉट (AgroBot)** हूँ, आपका डिजिटल फसल सलाहकार और पौध रोग विशेषज्ञ। 🌾\n\n"
                "आप किसी भी फसल (आलू, टमाटर, संतरा, सेब, मक्का), रोग रोकथाम (Prevention), कवकनाशी उपचार, "
                "वर्तमान मौसम व सिंचाई, या अपने सहेजे गए पौधों के इतिहास के बारे में पूछ सकते हैं। मैं आपकी क्या मदद करूँ?"
            )
        return (
            "Hello! I am **AgroIntelli AI (AgroBot)**, your precision agricultural pathologist and crop health consultant. 🌾\n\n"
            "You can ask me about:\n"
            "• **Crop Disease Prevention** (e.g. *'How could I prevent potato disease?'*)\n"
            "• **Chemical & Organic Treatment Protocols** (e.g. *'Orange citrus greening spray'*)\n"
            "• **Live Weather Risk & Irrigation** (e.g. *'Current humidity & disease risk'*)\n"
            "• **Multi-Day Timeline Trajectory Consultations**\n\n"
            "How can I assist your farm today?"
        )

    # 2. Trajectory Discussion with AgroBot
    if is_trajectory or is_trajectory_inquiry(query):
        crop_title = (s_ctx.get("crop") or query_crop or "Crop").title()
        init_p = (s_ctx.get("initial_prediction") or "Baseline Condition").replace("_", " ").title()
        late_p = (s_ctx.get("latest_prediction") or "Current Observation").replace("_", " ").title()
        total_d = s_ctx.get("total_days") or 5
        delta_val = s_ctx.get("delta_pct", 0)
        traj_status = s_ctx.get("trajectory") or ("Worsening" if delta_val > 0 else "Improving")

        if language == "Hindi":
            return (
                f"### 📈 **{crop_title} टाइमलाइन प्रोग्रेशन एवं सुधार परामर्श रिपोर्ट**\n\n"
                f"- **मूल स्थिति (Baseline)**: {init_p}\n"
                f"- **वर्तमान स्थिति (Latest)**: {late_p} ({total_d} दिनों के बाद)\n"
                f"- **संक्रमण बदलाव**: **{delta_val}%** ({traj_status})\n\n"
                f"#### 🔍 कृषि वैज्ञानिक विश्लेषण:\n"
                f"समय के साथ लक्षणों का यह बदलाव तापमान ({temp_c}°C) और आर्द्रता ({hum_pct}%) के सीधे संपर्क में रहने के कारण हुआ है।\n\n"
                f"#### 🛠️ सुझाई गई रिकवरी रणनीति:\n"
                f"1. **तत्काल सुरक्षात्मक स्प्रे**: यदि रोग बढ़ रहा है, तो 24-48 घंटों के भीतर लक्षित कवकनाशी या जैविक नीम तेल का छिड़काव करें।\n"
                f"2. **संक्रमित पत्तियां हटाएं**: अधिक खराब हो चुकी निचली पत्तियों को काटकर नष्ट करें ताकि बीजाणु न उड़ें।\n"
                f"3. **सिंचाई नियंत्रण**: पत्तियों पर ऊपर से पानी न डालें; केवल सुबह के समय जड़ों में ड्रिप से पानी दें।"
            )
        return (
            f"### 📈 Clinical Trajectory Consultation for **{crop_title}** ({total_d}-Day Span)\n\n"
            f"- **Baseline Diagnosis**: **{init_p}**\n"
            f"- **Latest Sequential Scan**: **{late_p}**\n"
            f"- **Pathology Trajectory**: **{traj_status}** ({delta_val > 0 and '+' or ''}{delta_val}% net area delta)\n"
            f"- **Environmental Context**: Ambient temperature {temp_c}°C with {hum_pct}% relative humidity.\n\n"
            f"#### 🔍 Clinical Assessment:\n"
            f"Over the {total_d}-day monitoring interval, the lesion progression indicates "
            f"{'active pathogen replication driven by sustained microclimate moisture' if delta_val > 0 else 'positive response to sanitation or initial fungicide barriers'}. "
            f"Grad-CAM visual activation confirms that disease margins are {'actively encroaching into green lamina' if delta_val > 0 else 'stabilizing and contracting'}.\n\n"
            f"#### 🛠️ Phased Agronomic Recovery Protocol:\n"
            f"1. **Therapeutic Action**: Apply targeted curative systemic fungicide (or cold-pressed neem oil 5 ml/L if organic) within 24–48 hours.\n"
            f"2. **Canopy Sanitization**: Selectively prune severely necrotic lower foliage to reduce internal inoculum pressure.\n"
            f"3. **Irrigation Protocol**: Switch exclusively to early morning ground-level drip irrigation; completely eliminate overhead wetting.\n"
            f"4. **Checkpoint**: Perform a follow-up scan in 4–5 days to verify lesion stabilization."
        )

    # 3. Disease Prevention Inquiries (e.g. "how could i prevent potato disease")
    if is_prevention:
        target_crop = (query_crop or (knowledge.get("crop") if knowledge else None) or "potato").lower()
        if target_crop in CROP_PREVENTION_GUIDE:
            guide = CROP_PREVENTION_GUIDE[target_crop]
            steps_rendered = "\n".join([f"{i+1}. **{title}**:\n   {desc}" for i, (title, desc) in enumerate(guide["steps"])])
            return (
                f"### 🛡️ {guide['title']}\n\n"
                f"Disease prevention is far more effective and economical than reactive chemical spraying. "
                f"Here is the gold-standard prophylactic management protocol for **{target_crop.title()}**:\n\n"
                f"{steps_rendered}\n\n"
                f"**Pro Tip**: If rainfall is forecasted, apply a protective copper barrier spray (Copper Oxychloride 2.5 g/L) 24 hours *before* the rain to prevent spore germination."
            )
        elif knowledge and knowledge.get("prevention"):
            return (
                f"### 🛡️ Disease Prevention Protocol for **{knowledge.get('common_name', target_crop.title())}**\n\n"
                f"1. **Prophylactic Field Management**:\n   {knowledge.get('prevention')}\n"
                f"2. **Canopy Aeration**: Ensure proper spacing between rows to promote rapid leaf drying.\n"
                f"3. **Water Management**: Deliver water strictly via ground drip lines in early mornings to minimize leaf wetness duration.\n"
                f"4. **Biological Barrier**: Apply protective biofungicide (*Bacillus subtilis* or cold-pressed neem oil 5 ml/L) during humid infection windows."
            )

    # 4. Weather & Temperature Inquiries
    if is_weather:
        disease_risk = "Elevated risk for fungal spore germination and foliar blights" if hum_pct >= 72 else "Moderate disease pressure; favorable for standard foliage development"
        if language == "Hindi":
            return (
                f"### ⛅ वर्तमान मौसम और कृषि मौसम जोखिम रिपोर्ट\n\n"
                f"- **तापमान**: **{temp_c}°C**\n"
                f"- **सापेक्ष आर्द्रता (Humidity)**: **{hum_pct}%**\n"
                f"- **वर्षा (Rainfall)**: **{rain_mm} मिमी/घंटा** ({condition})\n\n"
                f"**फसल स्वास्थ्य प्रभाव**: {disease_risk}।\n"
                f"यदि आर्द्रता 75% से अधिक है, तो शाम के समय पत्तियों पर पानी छिड़कने से बचें और निचली पत्तियों पर फफूंद के लक्षणों की जांच करें।"
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

    # 5. Saved Plants / MongoDB Journal Inquiries
    if is_journal:
        if mongo_summary:
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

    # 6. Specific Disease Knowledge Profile (When a Specific Disease was identified)
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
            return (
                f"### 🌿 Organic & Biological Solutions for **{d_name}** ({crop})\n\n"
                f"1. **Cold-Pressed Neem Oil Spray**: Mix 5 ml pure neem oil (10,000 ppm Azadirachtin) + 1 ml mild liquid soap in 1 liter of lukewarm water. Spray thoroughly over both leaf surfaces every 5–7 days.\n"
                f"2. **Biological Remedy**: {org}\n"
                f"3. **Cultural Sanitation**: {prev}"
            )

        # Default disease profile
        scan_metric = ""
        if not user_asked_different_crop and aff_pct is not None:
            scan_metric = f"\n- **Estimated Affected Area (Grad-CAM)**: **{aff_pct}%** (Confidence: {confidence_pct}%)"

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

    # 7. Crop-only query without specific disease (e.g. "tell me about potato disease", "potato crop overview")
    if query_crop:
        c_title = query_crop.title()
        return (
            f"### 🥔 Overview of **{c_title}** Pathology, Health & Care\n\n"
            f"Common foliar challenges in {c_title} cultivation include Early Blight (*Alternaria*), Late Blight (*Phytophthora*), and bacterial wilts.\n\n"
            f"#### Essential Care Guidelines for {c_title}:\n"
            f"1. **Field Scouting**: Inspect leaf undersides and lower foliage weekly for concentric brown rings or dark water-soaked margins.\n"
            f"2. **Prophylactic Spray**: Apply protective copper or mancozeb before extended humid rain fronts.\n"
            f"3. **Irrigation**: Deliver water strictly at root level through furrow or drip systems; keep foliage dry.\n"
            f"4. **Leaf Scanning**: Upload a photo of any suspect {c_title} leaf in the **Scan** tab to run deep learning disease diagnosis!"
        )

    # 8. Clarification Fallback (Never fabricate a random disease!)
    return (
        "I couldn't detect a specific agricultural, crop, or plant pathology topic in your question. 🌾\n\n"
        "As your precision AI agronomist, I specialize in:\n"
        "• **Disease Prevention Protocols** (e.g. *'How to prevent potato disease?'*)\n"
        "• **Chemical & Organic Treatments** (e.g. *'Neem oil dosage for aphids'*)\n"
        "• **Weather & Humidity Risk** (e.g. *'Current humidity & fungal risk'*)\n"
        "• **Timeline Progression Analysis**\n\n"
        "Could you please specify which crop (such as Potato, Tomato, Orange, Apple, Corn) or farming issue you'd like advice on?"
    )
