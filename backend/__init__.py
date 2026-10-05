"""
AgroIntelli Backend Package
Exposes modular sub-systems for Inference, Grad-CAM, Weather, Batch Progression, and Database.
"""

from .database import db_store
from .weather import fetch_live_weather_snapshot, evaluate_weather_progression
from .gradcam import generate_gradcam_and_affected_pct
from .inference import run_prediction, predict_with_context, load_model
from .batch import batch_progress_summary, normalize_labels
