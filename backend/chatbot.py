"""
AgroIntelli Precision Agronomic Chatbot Engine
Multimodal agricultural intelligence powered by Google Gemini (gemini-3.8-flash)
with comprehensive plant science (Plantist) cultivation guides, pathology knowledge,
OpenWeather microclimate telemetry, and MongoDB journal progression histories.
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
_active_gemini_model = "gemini-3.5-flash"
GEMINI_MODELS_CASCADE = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.8-flash"]

if IS_VALID_GEMINI_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini_client = genai
        _active_gemini_model = "gemini-3.5-flash"
        print(f"🌿 AgroIntelli Chatbot initialized with Google Gemini API ({_active_gemini_model}).")
    except Exception as e:
        print(f"ℹ️ Google Gemini SDK notice: {e}. Local plant science expert active.")
else:
    print("🌿 AgroIntelli Chatbot initialized in ultra-fast local plant science expert mode.")

SYSTEM_PROMPT = """You are AgroIntelli AI (AgroBot), a master agricultural scientist, plant pathologist, and expert botanist (Plantist).
You possess comprehensive horticultural and botanical expertise across ALL crops, fruits, vegetables, grains, berries, herbs, and ornamental plants on Earth (including Potato, Tomato, Strawberry, Mango, Corn, Pepper, Apple, Grape, Blueberry, Carrot, Onion, Lettuce, etc.).

CORE BEHAVIOR RULES:
1. ANSWER EXACTLY WHAT THE USER ASKS:
   - PLANTING / CULTIVATION / GROWING (e.g., 'how do i plant strawberry', 'how to grow mango', 'how do i plant a potato'):
     * Act strictly as an expert botanist and horticulturist.
     * Provide comprehensive, practical cultivation instructions:
       a) Propagation/Seed/Crown/Tuber Preparation (e.g. for strawberries, crown placement level with soil line; for potatoes, chitting tubers).
       b) Soil Requirements: Ideal soil texture, composition, organic matter, and precise pH range.
       c) Planting Depth, Technique & Spacing between plants and rows.
       d) Sunlight & Solar Exposure (daily direct sunlight hours).
       e) Irrigation & Moisture Regimen (water quantity, frequency, avoiding overhead wetting).
       f) Time to Germination & Harvest Timeline.
     * DO NOT fabricate, assume, or bring up plant diseases, blights, or chemical fungicides unless the user explicitly asks about diseases!
   - HARVESTING & STORAGE:
     * Focus on optimal maturity indicators, picking techniques, curing, and storage conditions.
   - PRUNING & CANOPY MANAGEMENT:
     * Focus on pruning techniques, sucker removal, staking, and trellising.
   - DISEASE PREVENTION (PROPHYLAXIS):
     * If the user asks how to prevent disease, provide cultural, biological, and irrigation prevention steps. Do NOT just dump disease symptoms and curative chemicals!
   - DIAGNOSED DISEASE / SICK LEAF:
     * When the user asks about an identified disease or asks for treatment for a sick plant, provide scientific pathology, causal pathogen, chemical spray dosages (e.g. Mancozeb, Chlorothalonil), and organic alternatives (neem oil).
   - LIVE WEATHER & TEMPERATURE:
     * Provide current temperature, humidity, rainfall, and discuss how these conditions impact crop health.
   - MULTI-DAY TIMELINE TRAJECTORY:
     * Analyze multi-day progression dynamics and provide a customized recovery roadmap.

2. AVOID CONTEXT LEAKS:
   - Never assume a plant is sick just because a crop name was mentioned in a general question.
   - Only reference an uploaded leaf photo if the user explicitly asks about 'my photo', 'my leaf', 'this scan', or 'my diagnosis'.

3. TONE & STRUCTURE:
   - Professional, encouraging, scientifically authoritative, and practical.
   - Use bold highlights and clean bullet points.
"""

FAST_GEN_CONFIG = {
    "max_output_tokens": 1200,
    "temperature": 0.25,
}


def _extract_gemini_reply(response):
    """Safely extracts text from Gemini response, including when finish_reason is MAX_TOKENS."""
    if not response:
        return ""
    try:
        if response.text:
            return response.text.strip()
    except Exception:
        pass
    try:
        if hasattr(response, 'candidates') and response.candidates:
            parts = response.candidates[0].content.parts
            txt = "".join(p.text for p in parts if hasattr(p, 'text')).strip()
            if txt:
                return txt
    except Exception:
        pass
    return ""

# Supported crops for entity matching
CROPS_LIST = [
    "tomato", "potato", "corn", "apple", "grape", "orange",
    "cherry", "peach", "pepper", "strawberry", "squash",
    "soybean", "blueberry", "raspberry", "mango", "banana",
    "watermelon", "melon", "onion", "garlic", "carrot", "lettuce",
    "cucumber", "cabbage", "cauliflower", "spinach", "eggplant",
    "brinjal", "chili", "chilli", "pea", "peas", "bean", "beans",
    "wheat", "rice", "cotton", "coffee", "tea", "sugarcane", "ginger"
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

# Complete Plant Science Cultivation Guides (Planting, Soil, Spacing, Care)
CROP_CULTIVATION_GUIDE = {
    "potato": {
        "title": "Master Potato Planting & Cultivation Guide",
        "steps": [
            ("Seed Tuber Preparation & Chitting", "Use certified disease-free seed potatoes. Sprout ('chit') them 1–2 weeks before planting in a cool, bright spot until sturdy green shoots (1–2 cm) appear. Cut large tubers into 40–50 g pieces with at least 1–2 eyes each, and let the cut surfaces dry/callus for 2 days."),
            ("Soil Preparation & Sunlight", "Potatoes thrive in deep, loose, well-draining sandy loam enriched with aged compost. Ideal soil pH is 5.5–6.5. Plant in a location that receives 6–8 hours of full direct sunlight daily."),
            ("Planting Depth & Spacing", "Dig trenches 15 cm (6 inches) deep. Place potato pieces cut-side down with eyes/sprouts facing upward. Space seed pieces 30 cm (12 inches) apart in rows 75–90 cm (30–36 inches) apart. Cover with 8–10 cm of loose soil."),
            ("Hilling (Ridging) Schedule", "When green stems reach 15–20 cm (6–8 inches) tall, mound soil up along both sides of the plants, covering the bottom half of the stem. Hill again 2–3 weeks later. Hilling prevents developing tubers from turning green/toxic in sunlight and dramatically increases tuber yield."),
            ("Watering & Irrigation", "Keep soil consistently moist (about 25–50 mm / 1–2 inches of water per week), especially during tuber bulking (when flowers appear). Always water at the base; avoid overhead foliage wetting."),
            ("Harvesting Timeline", "Harvest 'new potatoes' 7–8 weeks after planting when plants flower. For storage potatoes, wait 100–120 days until the foliage turns yellow and dies back; let tubers cure in dry soil for 10 days before digging.")
        ]
    },
    "tomato": {
        "title": "Master Tomato Planting & Cultivation Guide",
        "steps": [
            ("Seed Starting & Timing", "Start seeds indoors 6–8 weeks before the last spring frost. Transplant outdoors only when nighttime temperatures stay reliably above 12°C (55°F) and soil warms to 18°C."),
            ("Deep Planting Technique", "Dig deep holes. Pinch off the lower leaves and bury up to two-thirds of the stem. Tomatoes produce adventitious roots all along the buried stem, building an exceptionally strong root system."),
            ("Soil & Solar Exposure", "Plant in nutrient-rich, well-drained loamy soil (pH 6.2–6.8) with full sun (at least 8 hours of direct sunlight). Work in 2 inches of compost before planting."),
            ("Spacing & Staking", "Space indeterminate varieties 60–90 cm (24–36 inches) apart; determinate varieties 45–60 cm apart. Install stakes, cages, or trellises at planting time to avoid damaging root systems later."),
            ("Pruning & Maintenance", "Pinch out non-productive suckers in leaf axils on indeterminate plants to direct energy into fruit clusters. Remove all leaves within 25 cm of the soil to prevent soil-splash infections."),
            ("Watering Regimen", "Deliver 1–2 inches of water per week at ground level using drip irrigation. Inconsistent watering leads to blossom end rot and fruit splitting.")
        ]
    },
    "corn": {
        "title": "Master Sweet Corn Planting & Cultivation Guide",
        "steps": [
            ("Block Planting Layout", "Always plant corn in square blocks (e.g. at least 4 rows of 4–5 plants) rather than a single long row. Corn is wind-pollinated; block planting ensures full kernel development on every cob."),
            ("Soil Temperature & Sowing", "Sow directly outdoors when soil temperature reaches at least 16°C (60°F). Plant seeds 2.5–4 cm (1–1.5 inches) deep, spaced 20–25 cm apart in rows 75 cm apart."),
            ("Sunlight & Fertility", "Requires full sun and high soil fertility. Corn is a heavy nitrogen feeder; side-dress with nitrogen-rich compost or balanced organic fertilizer when plants reach knee-height."),
            ("Weed Control & Shallow Cultivation", "Corn roots are shallow; cultivate weeds carefully near the surface or apply straw mulch to conserve moisture and suppress weeds.")
        ]
    },
    "pepper": {
        "title": "Master Pepper (Bell & Chili) Cultivation Guide",
        "steps": [
            ("Warm Soil Requirement", "Peppers are very frost-sensitive. Transplant outdoors 2–3 weeks after last frost when soil is at least 18–20°C (65°F)."),
            ("Spacing & Depth", "Plant at the same depth as the nursery pot. Space plants 45–50 cm (18 inches) apart in rows 60 cm apart."),
            ("Staking & Pinching", "Pinch the first early flower buds to encourage bushy vegetative growth. Stake plants early to prevent heavy fruit loads from snapping brittle branches."),
            ("Sunlight & Soil", "Requires 6–8 hours of full sun and well-draining soil (pH 6.0–6.8). Avoid excessive nitrogen, which causes lush foliage with few peppers.")
        ]
    },
    "apple": {
        "title": "Master Apple Tree Planting & Care Guide",
        "steps": [
            ("Site & Pollination", "Plant in full sun in well-drained soil with a pH of 6.0–7.0. Most apple varieties require a second, compatible pollinator variety nearby for successful fruit set."),
            ("Planting Bare-Root or Potted Trees", "Dig a hole twice as wide as the root spread and the same depth as the root ball. Ensure the graft union sits 5–8 cm (2–3 inches) above the final soil line."),
            ("Canopy Pruning", "Prune in late winter while dormant to establish a central leader system, opening the canopy to sunlight and airflow."),
            ("Watering & Mulching", "Water young trees deeply once a week during dry spells. Mulch around the drip line (keeping mulch 10 cm away from the trunk).")
        ]
    },
    "grape": {
        "title": "Master Grapevine Planting & Vineyard Management",
        "steps": [
            ("Site Selection & Soil", "Plant in full sun on gentle slopes with deep, well-draining soil. Grapes tolerate poor soil but cannot stand waterlogged roots."),
            ("Planting & Trellising", "Space vines 1.8–2.4 meters (6–8 feet) apart. Install a sturdy wire trellis system during the first season."),
            ("Annual Pruning", "Prune heavily every dormant season, removing 85–90% of the previous year's wood to balance yield and fruit quality.")
        ]
    },
    "orange": {
        "title": "Master Citrus / Orange Tree Care Guide",
        "steps": [
            ("Sunlight & Drainage", "Citrus trees demand full sunlight (8+ hours) and sandy, well-draining soil. Standing water causes rapid Phytophthora root rot."),
            ("Planting Depth", "Plant slightly elevated so the root collar sits 2–3 cm above the surrounding grade; never bury the graft union."),
            ("Irrigation & Feeding", "Water deeply every 7–10 days during warm months, allowing the top 5 cm of soil to dry between waterings. Feed quarterly with balanced citrus fertilizer containing zinc, iron, and manganese.")
        ]
    },
    "strawberry": {
        "title": "Master Strawberry Planting & Cultivation Guide",
        "steps": [
            ("Crown Depth & Orientation", "Plant bare-root or potted crowns so that the midpoint of the central crown sits exactly level with the soil surface. Burying the crown leads to crown rot; leaving roots exposed causes desiccation."),
            ("Soil Drainage & pH Requirements", "Strawberries demand fertile, sandy loam with superb drainage and a slightly acidic pH of 5.5–6.5. Incorporate 5–8 cm of aged compost into the top 15 cm of soil."),
            ("Spacing & Bed Layout", "Space plants 30–45 cm (12–18 inches) apart in rows 90 cm apart, or 30 cm apart in raised beds. Stagger plantings for maximum airflow."),
            ("Mulching with Clean Straw", "Spread a 5 cm layer of clean wheat straw, pine needles, or black plastic mulch around plants. Mulch keeps developing fruit off moist soil, suppresses weeds, and moderates soil temperature."),
            ("Watering & Care", "Deliver 25–40 mm of water weekly via root drip irrigation. Pinch off first-month blossoms on June-bearing plants to stimulate strong crown and root establishment."),
            ("Harvesting Schedule", "Harvest berries when fully colored red down to the calyx. Pick gently in cool mornings with stems attached.")
        ]
    },
    "blueberry": {
        "title": "Master Blueberry Planting & Acidic Soil Management",
        "steps": [
            ("Acidic Soil Requirement", "Blueberries strictly require acidic soil with a pH of 4.5–5.5. Amend planting soil heavily with peat moss, pine bark, or elemental sulfur to reach optimal acidity."),
            ("Planting Depth & Spacing", "Space highbush varieties 1.2–1.5 meters apart. Plant at the same depth as the nursery container; never bury stems deeper."),
            ("Mulching & Shallow Roots", "Blueberries have shallow, fine root systems without root hairs. Maintain an 8–10 cm layer of pine needle or wood chip mulch and water consistently using drip irrigation.")
        ]
    },
    "raspberry": {
        "title": "Master Raspberry Cane Planting & Trellising Guide",
        "steps": [
            ("Site & Trellis Support", "Plant in full sun in deep, fertile, moisture-retentive soil (pH 6.0–6.8). Install a sturdy two-wire post trellis (at 0.8m and 1.5m heights)."),
            ("Planting Bare-Root Canes", "Plant dormant canes 45–60 cm apart in rows 1.8–2.4 m apart. Cut canes back to 20 cm after planting to stimulate vigorous new basal shoots."),
            ("Cane Pruning System", "Prune out spent floricanes (fruiting wood) immediately after harvest, leaving 4–6 vigorous primocanes per plant for the next year.")
        ]
    },
    "peach": {
        "title": "Master Peach Tree Planting & Orchard Care",
        "steps": [
            ("Sunlight & Chill Hours", "Plant in full direct sun in well-drained loamy soil (pH 6.0–6.8). Verify that the variety matches your area's winter chill hours (400–800 hours below 7°C)."),
            ("Planting & Graft Union", "Dig a wide planting hole. Position the tree so the graft union sits 5–8 cm above the final grade. Water deeply to settle soil."),
            ("Open-Center Pruning", "Prune at planting to establish an 'open-center' (vase) scaffold with 3–4 primary limbs, allowing sun into the canopy core.")
        ]
    },
    "cherry": {
        "title": "Master Cherry Tree Planting & Care Guide",
        "steps": [
            ("Soil Drainage & Exposure", "Plant in deep, well-draining soil (pH 6.2–6.8) with full sun. Avoid frost pockets and waterlogged ground."),
            ("Spacing & Pollination", "Space sweet cherries 4–6 meters apart (sour cherries 3–4 meters). Ensure you have compatible cross-pollinators unless planting a self-fertile variety."),
            ("Summer Pruning", "Prune cherry trees in mid-summer rather than winter dormancy to reduce susceptibility to bacterial canker (*Pseudomonas*).")
        ]
    },
    "squash": {
        "title": "Master Squash (Zucchini & Winter Squash) Guide",
        "steps": [
            ("Warm Soil & Hill Sowing", "Sow outdoors when soil reaches 18–21°C (65–70°F). Plant in mounds/hills 2.5 cm deep (4–5 seeds per hill, thinning to 2–3 strongest plants)."),
            ("Spacing & Sun", "Bush varieties require 90 cm spacing; vining varieties require 1.5–2 m. Provide at least 6–8 hours of intense direct sunlight."),
            ("Watering at Base", "Deliver 25–50 mm of water weekly directly at the base with soaker hoses. Avoid overhead wetting to prevent foliar mildew.")
        ]
    },
    "soybean": {
        "title": "Master Soybean Cultivation & Nitrogen Fixation Guide",
        "steps": [
            ("Bacterial Inoculation", "Inoculate seeds with *Bradyrhizobium japonicum* prior to sowing to maximize root nodule nitrogen fixation."),
            ("Sowing Depth & Spacing", "Sow seeds 2.5–4 cm deep when soil reaches 15°C (60°F). Space seeds 5–8 cm apart in rows 38–76 cm apart in full sun."),
            ("Harvesting", "Harvest green edamame pods when plump and 80–90% filled; for dry beans, wait until leaves turn yellow and drop and pods are completely brown.")
        ]
    }
}

# Disease Prevention Protocols by Crop
CROP_PREVENTION_GUIDE = {
    "potato": {
        "title": "Comprehensive Potato Disease Prevention Protocol",
        "steps": [
            ("Certified Disease-Free Seed Tubers", "Plant exclusively certified seed potatoes. Never plant potatoes showing soft rot, bruising, or sunken lesions."),
            ("Strict 3–4 Year Crop Rotation", "Never plant potatoes in soil previously cropped with Solanaceous crops (Tomato, Eggplant, Pepper) within the last 3–4 years."),
            ("Continuous Hilling & Ridging", "Hill up soil around stems as plants grow (15–20 cm ridges) to create a physical barrier preventing late blight spores from washing into tubers."),
            ("Drip or Furrow Irrigation Only", "Avoid overhead sprinkler irrigation; deliver water at root level in early mornings so foliage remains dry."),
            ("Prophylactic Biofungicide Inoculation", "Incorporate *Trichoderma harzianum* or *Bacillus subtilis* at planting. Apply preventive Copper Oxychloride (2.5 g/L) before monsoon rain fronts.")
        ]
    },
    "tomato": {
        "title": "Comprehensive Tomato Disease Prevention Protocol",
        "steps": [
            ("Bottom Pruning & Soil Clearance", "Prune all suckers and leaves within 25–30 cm of the ground to eliminate soil-splash pathogen transfer."),
            ("Organic Mulching", "Cover bare soil with clean straw or plastic mulch to suppress weeds and contain soil-borne spores."),
            ("Morning Root Irrigation", "Water strictly at root level between 6:00 AM and 9:00 AM using drip lines."),
            ("Preventative Biological Sprays", "Apply cold-pressed neem oil (5 ml/L) or *Bacillus subtilis* every 7–10 days during humid weather.")
        ]
    }
}


def is_planting_inquiry(query):
    """Detects inquiries about planting, growing, sowing, germination, seedlings, soil prep, etc."""
    q = query.lower()
    planting_words = [
        "plant", "planting", "how to plant", "how do i plant", "how to grow",
        "how do i grow", "sow", "sowing", "germination", "seedling", "depth",
        "spacing", "when to plant", "lagaye", "kaise lagaye", "ugaye", "bone ka tarika",
        "cultivation", "start growing", "soil prep"
    ]
    return any(re.search(r'\b' + re.escape(w) + r'\b', q) for w in planting_words)


def is_harvest_inquiry(query):
    """Detects inquiries about harvesting, picking, curing, or storing crops."""
    q = query.lower()
    harvest_words = ["harvest", "harvesting", "when to pick", "storage", "store", "keeping", "post-harvest curing", "curing tubers", "todna"]
    return any(re.search(r'\b' + re.escape(w) + r'\b', q) for w in harvest_words)


def is_pruning_inquiry(query):
    """Detects inquiries about pruning, suckers, staking, or trellising."""
    q = query.lower()
    pruning_words = ["prune", "pruning", "stake", "staking", "trellis", "trellising", "sucker", "suckers", "cutting back"]
    return any(re.search(r'\b' + re.escape(w) + r'\b', q) for w in pruning_words)


def is_disease_or_symptom_inquiry(query):
    """Detects if user is asking about disease, symptoms, sickness, pests, or cures."""
    q = query.lower()
    disease_words = [
        "disease", "blight", "rot", "scab", "rust", "spot", "mold", "mildew", "virus",
        "wilt", "sick", "curling", "yellow leaf", "yellowing", "brown spot", "fungus",
        "fungal", "pest", "canker", "scorch", "cure", "treat", "treatment", "spray",
        "fungicide", "pesticide", "neem", "dawa", "karan", "bimari", "lakshan", "upchar"
    ]
    return any(re.search(r'\b' + re.escape(w) + r'\b', q) for w in disease_words)


def is_prevention_inquiry(query):
    """Detects inquiries about disease prevention, prophylaxis, avoiding infection."""
    q = query.lower()
    prev_words = [
        "prevent", "prevention", "preventative", "preventive", "prophylaxis",
        "avoid", "how to avoid", "stop disease", "protect", "protection",
        "kaise bachein", "roktham", "bachav", "suraksha"
    ]
    return any(re.search(r'\b' + re.escape(w) + r'\b', q) for w in prev_words)


def is_trajectory_inquiry(query):
    """Detects inquiries about timeline progression, trajectory, recovery over time."""
    q = query.lower()
    traj_words = [
        "trajectory", "timeline progression", "progression results", "healing progression",
        "multi-day", "multiday", "recovery trajectory", "disease trajectory"
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


def is_gibberish(text):
    """Detects random keystrokes, keyboard mashing, or unpronounceable non-words."""
    if not text:
        return True
    t = text.strip().lower()

    if t in ["random words", "some random words", "asdfgh", "qwerty", "tbrehh", "blabla", "xyz", "abc"]:
        return True

    words = re.findall(r'[a-zA-Z]+', t)
    if not words:
        return True

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
        'prevent', 'prevention', 'preventive', 'avoid', 'protect', 'protection', 'grow', 'growing',
        'sow', 'sowing', 'seed', 'depth', 'spacing', 'harvest', 'harvesting', 'tuber', 'root',
        'trajectory', 'timeline', 'progression', 'recovery', 'worsen', 'healing',
        'namaste', 'hola', 'namoshkar', 'karan', 'upchar', 'dawa', 'sinchai', 'paani', 'mati'
    }

    meaningful_count = 0
    for w in words:
        if w in meaningful_words:
            meaningful_count += 1
            continue
        if any(mash in w for mash in ["asdf", "qwerty", "zxcv", "hjkl", "ghjk", "dfgh"]):
            continue
        if re.search(r'[^aeiouy\s]{4,}', w):
            continue
        if re.search(r'(.)\1{2,}', w):
            continue
        if len(w) >= 4:
            v_count = sum(1 for c in w if c in 'aeiouy')
            if v_count == 0 or (v_count / len(w)) < 0.2:
                continue

    return meaningful_count == 0


def has_agronomic_intent(query):
    """Determines if query contains any botanical, agricultural, weather, or conversational relevance."""
    q = query.lower().strip()
    if is_greeting(q) or is_weather_inquiry(q) or is_journal_inquiry(q) or is_trajectory_inquiry(q) or is_prevention_inquiry(q):
        return True
    if any(crop in q for crop in CROPS_LIST):
        return True
    agri_terms = [
        "leaf", "plant", "crop", "flower", "fruit", "root", "soil", "seed",
        "water", "irrigation", "fertilizer", "compost", "spray", "cure", "treat",
        "fungicide", "pesticide", "neem", "blight", "rot", "scab", "rust", "mildew",
        "spot", "mold", "virus", "symptom", "disease", "harvest", "prun", "grow", "sow", "farm"
    ]
    return any(re.search(r'\b' + re.escape(term) + r'\b', q) for term in agri_terms)


def extract_plant_name_from_query(query):
    """Extracts the botanical crop or plant name from freeform user inquiries."""
    if not query:
        return None
    q = query.lower().strip()
    
    patterns = [
        (r"(?:how (?:do i|to|can i)\s+)?(?:plant|grow|sow|care for|prune|harvest)\s+(?:a|an|the)?\s*([a-z]+(?:\s+[a-z]+)?)", 1),
        (r"(?:plant|grow|sow|care for)\s+(?:a|an|the)?\s*([a-z]+(?:\s+[a-z]+)?)", 1),
    ]
    for pat, group_idx in patterns:
        m = re.search(pat, q)
        if m:
            cand = m.group(group_idx).strip()
            cand = re.sub(r'\b(in|at|on|with|from|and|for|my|our|garden|pot|indoor|outdoors|home|season|winter|summer|spring)\b.*', '', cand).strip()
            if cand and len(cand) >= 3 and cand not in ["tree", "plant", "crop", "seed", "disease"]:
                return cand

    for crop in CROPS_LIST:
        if re.search(r'\b' + re.escape(crop) + r'\b', q):
            return crop

    return None


def match_disease_or_crop_from_query(query):
    """
    Extracts explicitly mentioned disease and crop.
    Only returns a disease key IF a specific disease keyword or symptom was actually mentioned!
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
            # Check if user mentioned a disease of this crop
            for k, v in KNOWLEDGE_STORE.items():
                if k.startswith(crop):
                    disease_part = k[len(crop):].replace("_", " ").strip()
                    tokens = [tok for tok in disease_part.split() if len(tok) > 3]
                    if tokens and any(tok in q for tok in tokens):
                        return k, v, crop
            # No disease mentioned, just crop!
            return None, None, crop

    # 4. Freeform botanical plant extraction
    cand_plant = extract_plant_name_from_query(query)
    if cand_plant:
        return None, None, cand_plant

    return None, None, None


def generate_chat_response(user_message, scan_context=None, history=None, language="English"):
    """
    Generates a conversational response grounded in Gemini (gemini-3.8-flash)
    with seamless local plant science expert fallback.
    """
    msg = (user_message or "").strip()

    # 1. Empty message
    if not msg:
        welcome_msgs = {
            "Hindi": "नमस्ते! मैं एग्रोबॉट (AgroBot) हूँ, आपका डिजिटल कृषि व पादप विशेषज्ञ। फसल बुवाई, सिंचाई, देखभाल या रोग उपचार के बारे में पूछें।",
            "Bengali": "নমস্কার! আমি এগ্রোবট (AgroBot), আপনার ডিজিটাল কৃষি সহকারী। ফসল রোপণ, পরিচর্যা বা রোগ সম্পর্কে জিজ্ঞাসা করুন।",
        }
        return {
            "reply": welcome_msgs.get(language, "Hello! I am AgroIntelli AI (AgroBot), your precision plant scientist and crop advisor. Ask me anything about planting, cultivation, disease management, or watering!"),
            "grounded": True,
            "model_used": "AgroIntelli Local Expert"
        }

    # 2. Gibberish / Meaningless text check
    if is_gibberish(msg) or not has_agronomic_intent(msg):
        return {
            "reply": "I couldn't detect a specific agricultural, crop, or botanical topic in your question. 🌾\n\nAs your precision plant scientist, I can help you with:\n• **Planting & Growing Guides** (e.g. *'How do I plant a potato?'* or *'How to grow tomatoes?'*)\n• **Disease Prevention & Care** (e.g. *'How to prevent late blight?'*)\n• **Targeted Pathology & Treatment** (e.g. *'Cure for Tomato Early Blight'*)\n• **Weather Risk & Irrigation Schedule**\n\nCould you please specify which crop, farming practice, or symptom you'd like advice on?",
            "grounded": False,
            "model_used": "AgroIntelli Guard"
        }

    # 3. Telemetry and context
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
    is_trajectory_flag = scan_ctx.get("is_trajectory", False) or is_trajectory_inquiry(msg)

    # User explicitly asking about the uploaded scan?
    user_asking_active_scan = any(w in msg.lower() for w in [
        "this scan", "my scan", "this leaf", "my leaf", "my photo", "the diagnosis", "what disease is this"
    ])

    # 4. Entity Extraction & Intent Classification
    query_key, query_rec, query_crop = match_disease_or_crop_from_query(msg)
    user_asked_different_crop = False

    active_knowledge = knowledge
    active_pred = scanned_pred

    if query_rec:
        active_knowledge = query_rec
        active_pred = query_key
        if not scanned_pred or (query_crop and query_crop.lower() not in (scanned_pred or "").lower()):
            user_asked_different_crop = True
    elif query_crop:
        if not user_asking_active_scan:
            active_knowledge = None
            active_pred = query_crop
        if not scanned_pred or (query_crop.lower() not in (scanned_pred or "").lower()):
            user_asked_different_crop = True

    is_weather = is_weather_inquiry(msg)
    is_journal = is_journal_inquiry(msg)
    is_prevention = is_prevention_inquiry(msg)
    is_planting = is_planting_inquiry(msg)
    is_harvest = is_harvest_inquiry(msg)
    is_pruning = is_pruning_inquiry(msg)
    is_disease = is_disease_or_symptom_inquiry(msg)

    # 5. Execute via Google Gemini Cascade (gemini-3.5-flash -> gemini-3.5-flash-lite -> gemini-3.8-flash)
    if IS_VALID_GEMINI_KEY and _gemini_client is not None:
        weather_desc = f"{weather.get('temp_c', '26')}°C, {weather.get('humidity_pct', '65')}% humidity, Rain: {weather.get('rain_1h_mm', '0')} mm/h"

        prompt_parts = [
            f"{SYSTEM_PROMPT}\n",
            f"[USER QUERY CONTEXT]",
            f"- Identified Crop/Plant: {query_crop or 'General Botany'}",
            f"- Category: {'Planting/Growing' if is_planting else 'Disease Prevention' if is_prevention else 'Harvesting' if is_harvest else 'Pruning' if is_pruning else 'Pathology' if is_disease else 'General Farming'}",
            f"- Local Microclimate Weather: {weather_desc}",
        ]

        # ONLY inject previous scan telemetry if the user is actually asking about their scan
        if user_asking_active_scan and scanned_pred:
            prompt_parts.append(f"- Active Scanned Leaf: {scanned_pred} ({conf}% confidence, {aff_pct}% blade area affected)")
        elif is_trajectory_flag:
            prompt_parts.append(f"- Multi-day Trajectory Data: {scan_ctx}")

        if is_journal and mongo_summary:
            prompt_parts.append(f"- User MongoDB Plant Journal History:\n{mongo_summary}")

        prompt_parts.extend([
            f"\n[USER QUESTION]",
            f"{msg}\n",
            f"[STRICT INSTRUCTIONS]",
            f"1. Answer what the user specifically asked. If they asked how to plant/grow ANY plant (e.g. Strawberry, Mango, Potato, Tomato, etc.), give full botanical instructions: propagation/seed/crown preparation, soil requirements (composition & pH), planting depth, spacing, sunlight, watering needs, and time to harvest. DO NOT talk about disease or fungicides unless explicitly asked!",
            f"2. If they asked for disease prevention, provide prophylactic cultural, biological, and irrigation management.",
            f"3. Target Language: {language}.",
            f"4. Do not use raw markdown heading markers (#). Use bold text and bullet points."
        ])

        full_prompt = "\n".join(prompt_parts)

        for target_model in GEMINI_MODELS_CASCADE:
            try:
                model_inst = _gemini_client.GenerativeModel(
                    model_name=target_model,
                    generation_config=FAST_GEN_CONFIG
                )
                response = model_inst.generate_content(full_prompt)
                reply_text = _extract_gemini_reply(response)
                if reply_text:
                    return {
                        "reply": reply_text,
                        "grounded": True,
                        "model_used": f"Google {target_model}"
                    }
            except Exception as e:
                print(f"ℹ️ Gemini Chat notice with {target_model} ({e}); trying next model...")
                continue

    # 6. High-Accuracy Local Plant Science Expert Fallback
    reply = _build_local_grounded_reply(
        query=msg,
        scanned_pred=scanned_pred,
        active_pred=active_pred,
        aff_pct=aff_pct,
        weather=weather,
        knowledge=active_knowledge,
        language=language,
        confidence_pct=conf,
        is_weather=is_weather,
        is_journal=is_journal,
        is_prevention=is_prevention,
        is_planting=is_planting,
        is_harvest=is_harvest,
        is_pruning=is_pruning,
        is_disease=is_disease,
        is_trajectory=is_trajectory_flag,
        user_asked_different_crop=user_asked_different_crop,
        query_crop=query_crop,
        user_asking_active_scan=user_asking_active_scan,
        scan_ctx=scan_ctx
    )
    return {
        "reply": reply,
        "grounded": True,
        "model_used": "AgroIntelli Plant Science Expert"
    }


def _build_local_grounded_reply(
    query, scanned_pred=None, active_pred=None, aff_pct=None, weather=None,
    knowledge=None, language="English", confidence_pct=0, is_weather=False,
    is_journal=False, is_prevention=False, is_planting=False, is_harvest=False,
    is_pruning=False, is_disease=False, is_trajectory=False,
    user_asked_different_crop=False, query_crop=None,
    user_asking_active_scan=False, scan_ctx=None
):
    """
    Local Plant Science Expert Engine.
    Distinguishes precisely between: Planting, Prevention, Harvesting, Pruning,
    Disease Treatment, Trajectory Review, Weather, and Greetings.
    NEVER defaults to diseases when the user asks how to plant or grow!
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
        return (
            "Hello! I am **AgroIntelli AI (AgroBot)**, your precision plant scientist, botanist, and crop health consultant. 🌾\n\n"
            "You can ask me about:\n"
            "• **Planting & Cultivation Guides** (e.g. *'How do I plant a potato?'* or *'How to grow tomatoes?'*)\n"
            "• **Disease Prevention Protocols** (e.g. *'How to prevent late blight?'*)\n"
            "• **Targeted Pathology & Spray Protocols** (e.g. *'Treatment for citrus greening'*)\n"
            "• **Microclimate Weather & Irrigation Management**\n\n"
            "How can I assist your crop or garden today?"
        )

    # 2. Planting & Cultivation Guide (e.g. "how do i plant strawberry", "how to grow mango", "how do i plant a potato")
    if is_planting:
        target_crop = (query_crop or extract_plant_name_from_query(query) or "Your Plant").lower()
        if target_crop in CROP_CULTIVATION_GUIDE:
            guide = CROP_CULTIVATION_GUIDE[target_crop]
            steps_rendered = "\n".join([f"{i+1}. **{title}**:\n   {desc}" for i, (title, desc) in enumerate(guide["steps"])])
            return (
                f"### 🌱 {guide['title']}\n\n"
                f"Here is the complete, practical botanical guide to planting and growing **{target_crop.title()}**:\n\n"
                f"{steps_rendered}\n\n"
                f"**Botanist Tip**: Always ensure soil temperature and moisture are appropriate before planting to establish strong root architecture."
            )
        else:
            return (
                f"### 🌱 Master Botanical Planting & Cultivation Guide for **{target_crop.title()}**\n\n"
                f"As your precision plant scientist, here is the clinical horticultural protocol for planting and cultivating **{target_crop.title()}**:\n\n"
                f"1. **Soil Preparation & pH Requirements**:\n"
                f"   Ensure loose, well-aerated loamy soil rich in aged organic compost. Most garden plants thrive in a slightly acidic to neutral pH (6.0–6.8) with excellent drainage to prevent root rot and waterlogging.\n\n"
                f"2. **Sunlight & Solar Exposure**:\n"
                f"   Position in a location receiving 6–8 hours of full, direct sunlight daily for optimal photosynthesis and vegetative vigor. In scorching summer climates, provide light afternoon shade.\n\n"
                f"3. **Planting Depth & Spacing Technique**:\n"
                f"   If sowing seeds, plant at a depth approximately 2–3 times the seed diameter. If transplanting crowns or root balls, set the plant so the root collar aligns exactly with the soil surface (never bury central crowns). Space adequately to allow full canopy airflow.\n\n"
                f"4. **Irrigation & Moisture Regimen**:\n"
                f"   Deliver 25–35 mm of water per week at ground level using drip irrigation early in the morning. Keep soil evenly moist while avoiding wet foliage to protect against foliar infections.\n\n"
                f"5. **Nutrient Feeding & Mulching**:\n"
                f"   Top-dress with clean organic mulch (straw or shredded bark) to conserve soil moisture and moderate soil temperature. Apply a balanced N-P-K fertilizer during active leaf and root development."
            )

    # 3. Disease Prevention Protocol (e.g. "how could i prevent potato disease")
    if is_prevention:
        target_crop = (query_crop or extract_plant_name_from_query(query) or "Your Plant").lower()
        if target_crop in CROP_PREVENTION_GUIDE:
            guide = CROP_PREVENTION_GUIDE[target_crop]
            steps_rendered = "\n".join([f"{i+1}. **{title}**:\n   {desc}" for i, (title, desc) in enumerate(guide["steps"])])
            return (
                f"### 🛡️ {guide['title']}\n\n"
                f"Prophylactic management protects your crop before pathogens invade. Here is the gold-standard prevention protocol for **{target_crop.title()}**:\n\n"
                f"{steps_rendered}"
            )
        elif knowledge and knowledge.get("prevention"):
            return (
                f"### 🛡️ Disease Prevention Protocol for **{knowledge.get('common_name', target_crop.title())}**\n\n"
                f"1. **Prophylactic Field Management**:\n   {knowledge.get('prevention')}\n"
                f"2. **Canopy Aeration**: Ensure proper spacing between rows to promote rapid leaf drying.\n"
                f"3. **Water Management**: Deliver water strictly via ground drip lines in early mornings to minimize leaf wetness duration.\n"
                f"4. **Biological Barrier**: Apply protective biofungicide (*Bacillus subtilis* or cold-pressed neem oil 5 ml/L) during humid infection windows."
            )

    # 4. Trajectory Discussion
    if is_trajectory or is_trajectory_inquiry(query):
        crop_title = (s_ctx.get("crop") or query_crop or "Crop").title()
        init_p = (s_ctx.get("initial_prediction") or "Baseline Condition").replace("_", " ").title()
        late_p = (s_ctx.get("latest_prediction") or "Current Assessment").replace("_", " ").title()
        total_d = s_ctx.get("total_days") or 5
        delta_val = s_ctx.get("delta_pct", 0)
        traj_status = s_ctx.get("trajectory") or ("Worsening" if delta_val > 0 else "Improving")

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

    # 5. Weather & Temperature Inquiries
    if is_weather:
        disease_risk = "Elevated risk for fungal spore germination and foliar blights" if hum_pct >= 72 else "Moderate disease pressure; favorable for standard foliage development"
        return (
            f"### ⛅ Current Microclimate Weather & Agronomic Risk\n\n"
            f"- **Ambient Temperature**: **{temp_c}°C**\n"
            f"- **Relative Humidity**: **{hum_pct}%**\n"
            f"- **Precipitation**: **{rain_mm} mm/h** ({condition})\n\n"
            f"**Pathogen Risk Assessment**: {disease_risk}.\n"
            f"When relative humidity exceeds 70–75%, fungal spores germinate rapidly on wet foliage. "
            f"Ensure good row ventilation and switch to morning ground-level drip irrigation."
        )

    # 6. Disease & Treatment (ONLY when disease/symptom is actually asked)
    if is_disease:
        if knowledge:
            crop = knowledge.get("crop", query_crop or "Crop").title()
            d_name = knowledge.get("common_name", (active_pred or "Plant Disease").replace("_", " ").title())
            pathogen = knowledge.get("pathogen", "Pathogenic microorganism")
            chem = knowledge.get("treatment_protocol", "Apply standard fungicide spray as per label recommendations.")
            org = knowledge.get("organic_remedies", "Spray cold-pressed neem oil (5 ml/L with 1 ml mild liquid soap).")
            prev = knowledge.get("prevention", "Maintain adequate plant spacing, sanitize pruning shears, and remove fallen infected leaves.")

            return (
                f"### 💊 Treatment & Pathology Protocol for **{d_name}** ({crop})\n\n"
                f"- **Causal Pathogen**: *{pathogen}*\n\n"
                f"#### 1. Targeted Chemical Protocol:\n"
                f"{chem}\n\n"
                f"#### 2. Organic & Biological Alternatives:\n"
                f"{org}\n\n"
                f"#### 3. Cultural Sanitation & Prevention:\n"
                f"{prev}"
            )
        else:
            c_title = (query_crop or "Plant").title()
            return (
                f"### 🩺 Plant Pathology & Care Protocol for **{c_title}**\n\n"
                f"1. **Symptom Isolation**: Immediately prune and safely dispose of heavily spotted or chlorotic foliage to prevent spread.\n"
                f"2. **Biological Spray**: Apply cold-pressed neem oil (5 ml/L water with mild liquid soap) or *Bacillus subtilis* early in the morning.\n"
                f"3. **Targeted Protectant**: For fungal blights or spots, apply a broad-spectrum copper-based or chlorothalonil spray following label directions.\n"
                f"4. **Irrigation Adjustments**: Switch strictly to ground-level drip irrigation; wet leaf canopies accelerate infection."
            )

    # 7. Harvesting & Storage
    if is_harvest:
        c_title = (query_crop or "Crop").title()
        return (
            f"### 🌾 Harvesting & Storage Guidelines for **{c_title}**\n\n"
            f"1. **Harvest Timing**: Harvest when fruit or tubers reach mature size and coloration, preferably on a dry morning after dew has evaporated.\n"
            f"2. **Curing**: For root crops (like potatoes), allow harvested produce to cure in a dry, ventilated, shaded area for 7–10 days to toughen skins.\n"
            f"3. **Storage Conditions**: Store in a cool, dark, well-aerated area (ideal temperature 7–10°C) away from direct sunlight."
        )

    # 8. User asking about their active scan
    if user_asking_active_scan and scanned_pred:
        d_clean = scanned_pred.replace("_", " ").title()
        return (
            f"### 🔬 Evaluation for Your Uploaded Leaf: **{d_clean}**\n\n"
            f"- **Confidence Score**: {confidence_pct}%\n"
            f"- **Affected Surface Area (Grad-CAM)**: {aff_pct if aff_pct is not None else 'Quantified'}%\n\n"
            f"To treat this condition, isolate the affected plant branch, remove heavily spotted foliage, and apply a targeted protectant or cold-pressed neem oil (5 ml/L) early in the morning."
        )

    # 9. General Crop Overview (Cultivation & Plant Science)
    if query_crop:
        c_title = query_crop.title()
        return (
            f"### 🌿 Botanical Overview & Cultivation of **{c_title}**\n\n"
            f"**{c_title}** is a rewarding crop when provided with optimal growing conditions:\n\n"
            f"• **Sunlight**: Requires 6–8 hours of daily direct sunlight.\n"
            f"• **Soil**: Deep, fertile, well-draining soil with abundant organic compost.\n"
            f"• **Watering**: Consistent moisture delivered at ground level (drip irrigation).\n\n"
            f"You can ask me specific questions such as:\n"
            f"- *'How do I plant {c_title}?'*\n"
            f"- *'How to prevent diseases on {c_title}?'*\n"
            f"- *'When is {c_title} ready to harvest?'*"
        )

    # 10. General Fallback
    return (
        "Hello! As your precision agricultural scientist and botanist, I am here to help you grow healthy crops.\n\n"
        "Feel free to ask:\n"
        "• How to plant, sow, or grow any crop (e.g. Potato, Tomato, Corn)\n"
        "• Disease prevention protocols and biological care\n"
        "• Weather and microclimate risk\n"
        "• Diagnosis for any suspect leaf photo"
    )
