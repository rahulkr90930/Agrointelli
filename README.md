# AgroIntelli

AgroIntelli is a crop-health web app for leaf disease diagnosis, Grad-CAM visual explanations, weather-aware advice, grower discussions, and plant scan timelines.

## Requirements

- **Python 3.11** (the project is pinned to Python 3.11.9; use a 64-bit installation).
- Git.
- Internet access for package installation and live weather. MongoDB Atlas and Gemini are optional.

The supported local setup uses a Python 3.11 virtual environment. Do not use Python 3.10, 3.12, or a system-wide environment for this setup.

## Clone and run on Windows

Open PowerShell in the folder where you want the project and run:

```powershell
git clone --branch v2 --single-branch https://github.com/rahulkr90930/Agrointelli.git
cd Agrointelli

py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run.py
```

`python --version` should report Python 3.11.x after activation. If PowerShell blocks virtual-environment activation, use this for the current terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

The launcher loads the TensorFlow model, starts the Flask app, and opens the site in your browser at <http://127.0.0.1:5000>. Keep the terminal open while using AgroIntelli; press **Ctrl+C** there to stop it. The frontend is served by Flask, so do not open `frontend\index.html` directly as a file.

## Run the Streamlit edition

The separate Streamlit app reuses the existing model and backend modules without changing the Flask frontend or API. It provides crop diagnosis, Grad-CAM results, weather context, and a plant journal with follow-up scans and record deletion. Sign in or create an account in the sidebar to access journal records across sessions; guest journals are limited to the current Streamlit session. The existing Flask app remains the version with community discussions, AgroBot, and browser voice features.

Activate the Python 3.11 virtual environment and install the project requirements as above, then run from the repository root:

```powershell
streamlit run streamlit_app.py
```

Streamlit opens the app in your browser. Stop it with **Ctrl+C** in the terminal.

### Deploy on Streamlit Community Cloud

1. Push the repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app using this repository, the `v2` branch, and `streamlit_app.py` as the entrypoint.
3. In **Advanced settings**, select **Python 3.11** to match the tested TensorFlow environment.
4. Add a `MONGO_URI` secret pointing to your own MongoDB Atlas database so accounts and journal records persist across app restarts. You can also set `MONGO_DB_NAME` and `OWM_API_KEY` secrets. Do not commit credentials or a `secrets.toml` file.

Without MongoDB, accounts and records use the local JSON fallback. Files on hosted app instances may be ephemeral, so configure MongoDB for persistent hosted accounts and journals. Streamlit Community Cloud's free resources and availability are subject to its current limits.

### macOS or Linux

Install Python 3.11, then from the repository directory run:

```bash
python3.11 --version
python3.11 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run.py
```

## Optional services

The app can run without external credentials:

- **MongoDB Atlas:** Configure `MONGO_URI` and optionally `MONGO_DB_NAME` to persist accounts, plant journals, discussions, comments, and votes in MongoDB. Without a usable MongoDB connection, the app falls back to local JSON storage in `backend\data_store.json`. The fallback file is local and is not encrypted.
- **Weather:** Configure `OWM_API_KEY` for OpenWeatherMap. Location is resolved automatically; Open-Meteo is used as a fallback when available.
- **Gemini chatbot:** Configure `GEMINI_API_KEY` to enable Gemini replies. AgroIntelli uses its local crop-knowledge chatbot when Gemini is not configured or unavailable.

To use these services, copy the example settings file and edit the placeholders:

```powershell
Copy-Item .env.example .env
```

Never commit `.env` or put real credentials in `.env.example`. Restart `python run.py` after changing environment settings.

## Main features

- Diagnose supported crop leaves with the bundled TensorFlow model and Grad-CAM overlays.
- Review disease references and crop-specific treatment guidance.
- Add initial scans to the plant journal and append dated follow-up scans to the same plant timeline.
- Delete a plant journal record and its scan timeline from either the Flask or Streamlit journal.
- Run progression analysis across multiple images with editable day numbers.
- Ask the AgroBot questions in English, Hindi, or Bengali. Voice input and speech output depend on browser support; speech output is enabled by default and can be muted.
- Browse grower discussions. Guests can upvote or downvote; posting and replying require signing in.

## Project layout

```text
backend/       Flask API, inference, chatbot, weather and persistence modules
frontend/      Web interface assets
notebooks/     Model training and data preparation notebooks
run.py         Local application launcher
streamlit_app.py  Optional Streamlit diagnosis and journal entry point
requirements.txt
```

## Troubleshooting

- **Wrong Python version:** Activate `.venv` and run `python --version`. Recreate the environment with `py -3.11 -m venv .venv` if it is not Python 3.11.
- **TensorFlow or package installation fails:** Confirm Python is 64-bit and 3.11, activate `.venv`, then retry `python -m pip install -r requirements.txt`.
- **The app does not open:** Wait for TensorFlow and the model to finish loading. Open <http://127.0.0.1:5000> manually and check <http://127.0.0.1:5000/health>.
- **No Gemini or MongoDB connection:** These services are optional. Check the `.env` settings and restart the app; local chatbot and JSON persistence are available as fallbacks.
