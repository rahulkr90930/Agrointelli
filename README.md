# AgroIntelli

AgroIntelli is a crop disease intelligence project with three parts:

- `notebooks/AgroIntelli_Phase1.ipynb` — training and experimentation notebook
- `backend/app.py` — Flask API for prediction, weather, and health checks
- `frontend/index.html` — static UI that talks to the Flask backend

## What this project does

- Upload a leaf image
- Run disease detection in **Field** or **Lab** mode
- Optionally inject live weather data
- Upload a **batch timeline** such as Day 1, Day 5, and Day 10
- See whether the disease is worsening, improving, or staying stable
- Show top-3 predictions, confidence, severity estimate, quality warnings, and agronomic advice

## Folder layout

```text
AgroIntelli_Project/
├─ backend/
│  ├─ app.py
│  └─ models/
│     └─ .gitkeep
├─ frontend/
│  └─ index.html
├─ notebooks/
│  └─ AgroIntelli_Phase1.ipynb
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Python / TensorFlow setup

This project is intended to run in a Python 3.11 virtual environment inside Jupyter. TensorFlow’s official install docs support Python 3.9–3.12 and recommend installing via `pip` in a virtual environment.

## Install

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Run the application

To start both the backend server and automatically open the frontend web interface in your default browser, run the unified launcher script from the project root:

```bash
python run.py
```

Alternatively, you can run them manually:

### Run the backend manually
```bash
python backend/app.py
```
The backend listens on `http://127.0.0.1:5000`.

### Run the frontend manually
Open `frontend/index.html` in your browser.

## Model files

The backend looks for trained artifacts in `backend/models/`:

- `agrointelli_phase1_final.keras`
- `class_index_map.json`
- `advice.json`

The notebook saves those artifacts into `agrointelli_data/artifacts/models/`. After training, copy them into `backend/models/` before serving the app.

## Notebook workflow

Open `notebooks/AgroIntelli_Phase1.ipynb` and run it top to bottom. It prepares the dataset, trains the dual-input EfficientNetB0 model, and exports the model artifacts and summary files.

## Notes

- If the model is missing, the backend falls back to demo mode so the UI still works.
- The weather toggle depends on live internet access.
- The frontend expects the backend on port `5000`.


## Batch progression feature

The app now includes a batch analysis mode that compares three images in order and returns a progression verdict. The frontend labels the uploads as Day 1, Day 5, and Day 10, while the backend compares the lesion/severity score across the sequence.

The weather toggle applies one live weather snapshot to the entire batch, which is useful for a current risk view. For true historical weather-by-day analysis, you would need to pass separate weather snapshots from your own data source.

## API

- `POST /batch_predict` — compare multiple images in one request
