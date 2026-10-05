# AgroIntelli: Multimodal Crop Disease Diagnostic & Progression Intelligence System

AgroIntelli is an end-to-end, production-ready precision agriculture platform combining deep learning computer vision, multimodal weather fusion, explainable AI (Grad-CAM), and temporal disease progression tracking.

Built as a final-year engineering capstone project, AgroIntelli empowers farmers and agronomists to diagnose leaf pathologies in the field, track lesion progression across time (e.g., Day 1 to Day 7), correlate disease spread with local weather conditions (rain, humidity, temperature), and receive targeted agronomic treatment protocols.

---

## 🌟 Key Features

1. **Dual Model Architecture Selection**:
   - **⚡ MobileNetV3-Large**: Optimized for low-latency edge inference on low-power agricultural IoT devices and mobile devices.
   - **🎯 EfficientNet-B0**: High-capacity compound-scaled convolutional network delivering deep feature extraction and superior diagnostic nuance.
   - Easily switch between architectures directly in the web interface or via the REST API (`architecture=mobilenet` vs `architecture=efficientnet`).

2. **Multimodal Weather Integration**:
   - Fuses visual leaf representations with live OpenWeatherMap microclimate telemetry (temperature, relative humidity, rainfall, wind speed).
   - Informs both diagnostic confidence and disease progression risk.

3. **Explainable AI (Grad-CAM) & Affected Area Quantification**:
   - Generates pixel-level Class Activation Maps (Grad-CAM) showing exact leaf regions driving the diagnostic prediction.
   - Computes quantitative **% Leaf Area Affected** via Otsu adaptive thresholding on saliency gradients.

4. **Dynamic Batch Progression Analysis**:
   - Upload 3 sequential leaf images across any custom timeline.
   - **User-Defined Days**: Customize the day values directly (e.g., Day 1, Day 4, Day 12) rather than hardcoded intervals.
   - Computes disease velocity, trajectory trend (`IMPROVING`, `WORSENING`, `STABLE`), and net severity change ($\Delta$).

5. **Temporal Progression & Re-check Engine**:
   - Track individual plant leaves over time (Day 1 ➔ Day 3 ➔ Day 7).
   - Direct side-by-side visual comparison showing Day 1 baseline vs latest scan.
   - Dual Grad-CAM comparison showing whether lesions contracted or expanded.
   - **Smart Date & Stage Handling**: Displays the previous day and date the leaf was analyzed; automatically defaults to **Today's Date** for the follow-up, while letting the user freely modify both the date and the day label.
   - **Weather-Correlated Progression Intelligence**: Evaluates consecutive rain and elevated humidity ($\ge 75\%$), correlating persistent leaf surface moisture with accelerated fungal spore germination (*Phytophthora infestans*, powdery mildew, early blight).

6. **Cloud Persistence with MongoDB & Guest Mode**:
   - Secure farmer authentication (Sign In / Sign Up) + zero-barrier Guest Mode.
   - Records every inspection with `day_number`, date, timestamp, weather telemetry, Grad-CAM overlays, and compressed thumbnail images.
   - Full timeline history expandable per plant sample.

7. **Externalized Knowledge Base & Decoupled Rules**:
   - Disease risk rules, pathogen taxonomy, microclimate tolerances, chemical treatments, and organic remedies are decoupled into `backend/data/disease_knowledge.csv`.
   - Modifiable directly in Excel or Google Sheets without touching Python code.
   - Exposes REST endpoints (`GET /api/knowledge` and `GET /api/knowledge/<disease_id>`).

8. **AgroBot AI — Grounded Agricultural Chatbot**:
   - Integrated floating AI assistant powered by Google Gemini (`gemini-flash-latest`) and grounded in the active leaf diagnostic telemetry.
   - Automatically ingests current leaf disease prediction, Grad-CAM lesion area %, live microclimate weather (temperature, humidity, rain), and the structured pathology knowledge base.
   - Provides strictly agronomic guidance: chemical fungicide concentrations (g/L), spray timing, rain wash-off safeguards, organic neem oil recipes, and cultural sanitation.
   - Includes quick-prompt chips, session reset, and localized fallback expert mode.

---

## 📁 Repository Structure

```text
AgroIntelli/
├── backend/
│   ├── app.py                     # Flask REST Controller (routes, CORS, request dispatch)
│   ├── chatbot.py                 # Gemini LLM Chatbot Engine grounded in telemetry & CSV knowledge
│   ├── inference.py               # Dual AI Model Engine (MobileNetV3 & EfficientNet-B0 inference)
│   ├── gradcam.py                 # Grad-CAM XAI & Adaptive Otsu Lesion Quantification
│   ├── weather.py                 # Live Weather API integration & Microclimate Progression
│   ├── batch.py                   # Multi-image Chronological Trajectory & Velocity Analysis
│   ├── database.py                # MongoDB Atlas Cloud Persistence & User Authentication
│   ├── data/
│   │   └── disease_knowledge.csv  # Decoupled pathology, risk, treatment & organic remedy database
│   └── models/                    # Model weights & label mappings (.gitkeep)
├── frontend/
│   └── index.html                 # Single-page web application with responsive UI/UX
├── notebooks/                     # Canonical 3-notebook pipeline
│   ├── 01_data_loading_and_preprocessing.ipynb   # Dataset loader, EDA, weather synthesis & augmentation
│   ├── 02_train_mobilenet_v3.ipynb               # MobileNetV3 multimodal training, Grad-CAM & export
│   └── 03_train_efficientnet_b0.ipynb            # EfficientNet-B0 training, benchmarking & export
├── .env                           # Local environment configuration (API keys, MongoDB)
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git ignore rules for caches, weights, and logs
├── requirements.txt               # Pinned Python package dependencies
├── run.py                         # Unified launcher (starts Flask & opens browser)
└── README.md                      # Comprehensive project documentation
```

---

## 🧠 Notebook First Pipeline

The machine learning workflow is structured cleanly into three sequentially executable Jupyter notebooks:

| Notebook | Focus | Key Highlights |
| :--- | :--- | :--- |
| **`01_data_loading_and_preprocessing.ipynb`** | Data Engineering | Directory traversal, stratification (70/15/15), synthetic weather vector generation, TF data pipelines with cached augmentation. |
| **`02_train_mobilenet_v3.ipynb`** | MobileNetV3 Training | Lightweight multimodal fusion, Grad-CAM on layer `Conv_1`, sanity checking, Otsu affected % estimation, export to `backend/models/`. |
| **`03_train_efficientnet_b0.ipynb`** | EfficientNet-B0 Training | Deep multimodal fusion, 2-stage fine-tuning (frozen head ➔ unfreeze top 20 layers), Grad-CAM on `top_conv`, comparative benchmarking table. |

---

## 🚀 Quickstart: How to Load from GitHub and Run

Follow these step-by-step instructions to clone the repository from GitHub, set up your Python environment, and start the application.

### 1. Clone the Repository from GitHub

Open your terminal or PowerShell and clone the official repository:

```bash
git clone https://github.com/rahulkr90930/Agrointelli.git
cd Agrointelli
```

Make sure you are on the `v2` branch:

```bash
git checkout v2
```

---

### 2. Set Up Python Virtual Environment

Make sure you have **Python 3.10** or **3.11** installed.

#### On Windows (PowerShell or Command Prompt):
```powershell
python -m venv venv
.\venv\Scripts\activate
```

#### On macOS / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### 3. Install Dependencies

Install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables (`.env`)

Copy the provided template to create your `.env` file:

#### On Windows (PowerShell):
```powershell
Copy-Item .env.example .env
```

#### On Linux / macOS:
```bash
cp .env.example .env
```

Open `.env` in any text editor and fill in your credentials:

```env
# MongoDB Atlas Connection
MONGO_URI=mongodb+srv://<username>:<password>@cluster0.c5skeva.mongodb.net/?appName=Cluster0
MONGO_DB_NAME=agrointelli

# OpenWeatherMap API Key (Free tier supported)
OWM_API_KEY=your_openweathermap_api_key_here

# Server Configuration
PORT=5000
FLASK_DEBUG=False
```

> **Note**: If MongoDB is not configured or offline, AgroIntelli automatically falls back to an encrypted local JSON store (`backend/data_store.json`), so all features remain functional even without internet!

---

### 5. Run the Application

Start both the backend server and open the web frontend with one command:

```bash
python run.py
```

- The script starts the Flask REST API on `http://127.0.0.1:5000`.
- It will automatically launch `frontend/index.html` in your default web browser.

#### Manual Startup (Alternative):
If you prefer running the components separately:

1. **Start Backend**:
   ```bash
   python backend/app.py
   ```
2. **Open Frontend**:
   Double click `frontend/index.html` or open it in your browser.

---

## 📖 How to Use the Key Features

### 1. AI Model Selection
- On the main dashboard, choose between **⚡ MobileNetV3** (fastest edge inference) and **🎯 EfficientNet-B0** (deepest convolutional representation).
- Your choice will be reflected in the real-time diagnostic badge.

### 2. Custom Batch Progression Analysis
- Click the **Batch Progression** card at the top.
- For each of the 3 timeline slots, enter your own custom day numbers in the **Day** input (e.g., Sample 1: `Day 1`, Sample 2: `Day 4`, Sample 3: `Day 12`).
- Drop or select leaf photos for each slot.
- Click **📈 Run Batch Progression** to view the trajectory curve, spread velocity per day, and weather risk notes.

### 3. "Check on Past Disease" & Smart Leaf Re-Check
- After running a single leaf scan, click **💾 Save Leaf to Journal** at the bottom to store it in MongoDB.
- Click **🔍 Check on Past Disease** in the top navigation bar to open your tracked leaf records.
- Click **🔬 Re-check this Leaf**:
  - The modal automatically displays the **Previous Analysis Date & Day**.
  - The **Analysis Date** is pre-filled with **Today's Date** by default, and the next day number is automatically calculated based on elapsed days.
  - You can edit both the date and the day label if desired.
  - Upload the latest photo and click **🔬 Run Comparative Analysis** to view side-by-side leaf images, dual Grad-CAM heatmaps, daily spread velocity, and weather correlation.

---

## 📡 REST API Reference

### Health Check
`GET /health`
Returns backend status, loaded model architectures, and database connectivity.

### Single Leaf Diagnostic
`POST /predict`
- **Body (`multipart/form-data`)**:
  - `image`: Leaf image file (`.jpg`, `.png`).
  - `architecture`: `mobilenet` or `efficientnet` (default: `mobilenet`).
  - `field_mode`: `true` or `false` (enables field illumination enhancement ensemble).
  - `weather`: `true` or `false` (fetches live microclimate data).
  - `lat`, `lon`: Coordinates for weather API.
- **Returns**: Top-3 class predictions, confidence %, Grad-CAM heatmap, % affected tissue, weather snapshot, agronomic treatment advice.

### Batch Progression Analysis
`POST /batch_predict`
- **Body (`multipart/form-data`)**:
  - `images`: Array of 3 sequential images.
  - `labels`: JSON array of custom labels, e.g. `["Day 1", "Day 4", "Day 12"]`.
  - `architecture`: `mobilenet` or `efficientnet`.
  - `field_mode`: `true` or `false`.
  - `use_weather`: `true` or `false`.
- **Returns**: Progression trend (`IMPROVING`, `WORSENING`, `STABLE`), progression rate per day, comparative delta ($\Delta$), and combined weather progression risk notes.

### Leaf Journal & Re-check
- `POST /api/records/save` — Persist a leaf scan with `plant_name`, `day_number`, `date`, `weather`, and image thumbnail.
- `GET /api/records?user_id=<id>` — Fetch all tracked leaves for a user/guest.
- `POST /api/records/<record_id>/checkin` — Submit a follow-up photo for an existing tracked leaf with custom `date` and `day_label`. Returns side-by-side images, dual Grad-CAM heatmaps, elapsed days, daily spread rate, and weather-correlated spread rationale.

### Knowledge Base & Chatbot Data Access
- `GET /api/knowledge` — Retrieve the full structured agronomic knowledge base (10 disease/healthy profiles) in JSON format.
- `GET /api/knowledge/<disease_id>` — Fetch dedicated pathology profile (pathogen, microclimate thresholds, chemical treatments, organic remedies, prevention) for direct prompt injection into LLMs, LangChain, or agricultural chatbots.
