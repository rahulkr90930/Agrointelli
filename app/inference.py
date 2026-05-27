"""Lightweight inference engine for AgroIntelli.

Uses tflite-runtime (or tf.lite fallback) + PIL + numpy.
No full TensorFlow dependency — suitable for Android APK.
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageStat

# ---------- TFLite interpreter (works with tflite-runtime OR full tf) ----------
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# ---------- Paths (relative to this file) ----------
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "agrointelli_quant.tflite"
LABELS_PATH = ARTIFACTS / "labels.txt"
ADVICE_PATH = ARTIFACTS / "advice.json"


class DiseaseEngine:
    """Offline plant-disease classifier with quality & severity helpers."""

    def __init__(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"TFLite model not found: {MODEL_PATH}\n"
                "Run the training notebook first to export the model."
            )

        self.labels = self._load_labels()
        self.advice_map = self._load_advice()
        self.interpreter = Interpreter(model_path=str(MODEL_PATH))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        inp_shape = self.input_details[0]["shape"]  # e.g. [1, 160, 160, 3]
        self.input_size = int(inp_shape[1])

    # ------------------------------------------------------------------ helpers
    def _load_labels(self):
        if LABELS_PATH.exists():
            return [
                line.strip()
                for line in LABELS_PATH.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        return []

    def _load_advice(self):
        if ADVICE_PATH.exists():
            return json.loads(ADVICE_PATH.read_text(encoding="utf-8"))
        return {}

    # --------------------------------------------------------- image quality
    def image_quality_check(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            return {"ok": False, "reason": "image not readable"}

        img = img.resize((self.input_size, self.input_size))
        stat = ImageStat.Stat(img)
        brightness = float(sum(stat.mean) / 3.0)
        contrast = float(sum(stat.stddev) / 3.0)

        gray = img.convert("L")
        arr = np.array(gray, dtype=np.float32)
        gx = float(np.abs(np.diff(arr, axis=1)).mean()) if arr.shape[1] > 1 else 0.0
        gy = float(np.abs(np.diff(arr, axis=0)).mean()) if arr.shape[0] > 1 else 0.0
        sharpness = gx + gy

        warnings = []
        if brightness < 50:
            warnings.append("too dark")
        elif brightness > 205:
            warnings.append("too bright")
        if contrast < 20:
            warnings.append("low contrast")
        if sharpness < 10:
            warnings.append("blurry")

        return {
            "ok": len(warnings) == 0,
            "brightness": brightness,
            "contrast": contrast,
            "sharpness": sharpness,
            "warnings": warnings,
        }

    # --------------------------------------------------------- severity proxy
    def severity_proxy(self, image_path):
        try:
            img = Image.open(image_path).convert("RGB").resize(
                (self.input_size, self.input_size)
            )
        except Exception:
            return {"severity": "unknown", "lesion_ratio": 0.0}

        arr = np.array(img).astype(np.int16)
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

        leaf = (g > r * 0.9) & (g > b * 0.85) & (g > 30)
        if leaf.sum() < 100:
            return {"severity": "unknown", "lesion_ratio": 0.0}

        lesion = leaf & ((g < 120) | (r > 140) | (b > 140))
        ratio = float(lesion.sum() / leaf.sum())

        if ratio < 0.08:
            sev = "healthy/very mild"
        elif ratio < 0.22:
            sev = "early/moderate"
        else:
            sev = "severe"

        return {"severity": sev, "lesion_ratio": ratio}

    # --------------------------------------------------------- advice
    def advice_for(self, class_name):
        low = class_name.lower()
        for key, payload in self.advice_map.items():
            if key.lower() == low:
                return payload.get("summary", "")
        return "Keep monitoring the plant and follow agronomy guidance."

    # --------------------------------------------------------- preprocessing
    def preprocess(self, image_path):
        """Load, resize, and prepare image array for MobileNetV3.
        
        Note: The exported TFLite model contains an internal Rescaling layer
        that automatically maps inputs from [0, 255] to [-1, 1]. We must feed
        raw [0, 255] float32 values here to prevent double-scaling.
        """
        img = Image.open(image_path).convert("RGB")
        img = img.resize((self.input_size, self.input_size))
        arr = np.array(img, dtype=np.float32)
        return np.expand_dims(arr, axis=0)

    # --------------------------------------------------------- predict
    def predict(self, image_path, field_mode=True):
        quality = self.image_quality_check(image_path)
        severity = self.severity_proxy(image_path)

        x = self.preprocess(image_path)
        self.interpreter.set_tensor(self.input_details[0]["index"], x)
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        order = np.argsort(probs)[::-1]
        top3 = [
            (self.labels[i] if i < len(self.labels) else str(i), float(probs[i]))
            for i in order[:3]
        ]

        best_class, best_conf = top3[0]

        if field_mode:
            if best_conf >= 0.85:
                mode = "high confidence"
            elif best_conf >= 0.55:
                mode = "medium confidence"
            else:
                mode = "low confidence"
        else:
            mode = "lab mode"

        return {
            "prediction": best_class,
            "confidence": float(best_conf),
            "top3": top3,
            "mode": mode,
            "quality": quality,
            "severity_proxy": severity,
            "advice": self.advice_for(best_class),
        }
