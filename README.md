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

4. **Temporal Progression & Re-check Engine**:
   - Track individual plant leaves over time (Day 1 ➔ Day 3 ➔ Day 7).
   - Direct side-by-side visual comparison showing Day 1 baseline vs latest scan.
   - Dual Grad-CAM comparison showing whether lesions contracted or expanded.
   - Calculates **Daily Spread Velocity (% spread / day)** and net severity delta ($\Delta$).
   - **Weather-Correlated Progression Intelligence**: Evaluates consecutive rain and elevated humidity ($\ge 75\%$), correlating persistent leaf surface moisture with accelerated fungal spore germination (*Phytophthora infestans*, powdery mildew, early blight).

5. **Cloud Persistence with MongoDB & Guest Mode**:
   - Secure farmer authentication (Sign In / Sign Up) + zero-barrier Guest Mode.
   - Records every inspection with `day_number`, timestamp, weather telemetry, Grad-CAM overlays, and compressed thumbnail images.
   - Full timeline history expandable per plant sample.

---

## 📁 Repository Structure

```text
AgroIntelli_Project/
├── backend/
│   ├── app.py                     # Flask REST API (Inference, MongoDB, Weather, Grad-CAM)
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

## 🚀 Getting Started

### 1. Environment Setup

Ensure you have **Python 3.10** or **3.11** installed. Clone the repository and install dependencies:

```bash
cd AgroIntelli_Project
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Configuration (`.env`)

Copy `.env.example` to `.env` and provide your credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```env
# MongoDB Atlas Connection
MONGO_URI=mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?appName=Cluster0
MONGO_DB_NAME=agrointelli

# OpenWeatherMap API Key
OWM_API_KEY=your_openweathermap_api_key_here

# Flask Configuration
PORT=5000
FLASK_DEBUG=False
```

### 3. Launching the Application

Run the unified cross-platform launcher:

```bash
python run.py
```

This starts the Flask backend on `http://127.0.0.1:5000` and automatically launches your browser to `frontend/index.html`.

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
  - `images`: Array of 3 sequential images (Day 1, Day 5, Day 10).
  - `architecture`: `mobilenet` or `efficientnet`.
  - `field_mode`: `true` or `false`.
  - `weather`: `true` or `false`.
- **Returns**: Progression trend (`IMPROVING`, `WORSENING`, `STABLE`), progression rate per day, comparative delta ($\Delta$), and combined weather progression risk notes.

### Leaf Journal & Re-check
- `POST /journal/save` — Persist a leaf scan with `plant_name`, `day_number`, `date`, `weather`, and image thumbnail.
- `GET /journal/list?user_id=<id>` — Fetch all tracked leaves for a user/guest.
- `POST /journal/recheck` — Submit a follow-up photo for an existing tracked leaf. Returns side-by-side images, dual Grad-CAM heatmaps, elapsed days, daily spread rate, and weather-correlated spread rationale.
