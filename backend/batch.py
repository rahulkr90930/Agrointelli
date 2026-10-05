"""
AgroIntelli — Batch Timeline Progression Module
Analyzes multi-image chronological sequences, measures lesion trajectory,
and correlates spreading velocity across user-defined elapsed days.
"""

try:
    from .weather import parse_day_num
except (ImportError, ValueError):
    from weather import parse_day_num


def normalize_labels(labels, count):
    """Normalize input labels list, defaulting to Day 1, Day 5, Day 10 if missing."""
    defaults = ["Day 1", "Day 5", "Day 10"]
    out = []
    for i in range(count):
        if i < len(labels) and str(labels[i]).strip():
            out.append(str(labels[i]).strip())
        elif i < len(defaults):
            out.append(defaults[i])
        else:
            out.append(f"Day {i * 5 + 1}")
    return out


def batch_progress_summary(entries, weather=None):
    """Summarise multi-image progression factoring in day elapsed intervals and weather."""
    if not entries:
        return {
            "trend": "insufficient data",
            "delta": 0.0,
            "change_per_step": 0.0,
            "same_prediction": False,
            "explanation": "No images were provided.",
        }

    if len(entries) == 1:
        return {
            "trend": "single sample",
            "delta": 0.0,
            "change_per_step": 0.0,
            "same_prediction": True,
            "explanation": "Upload at least two images to detect progression.",
        }

    # Extract parsed day numbers
    day_numbers = []
    for idx, item in enumerate(entries):
        d_num = parse_day_num(item.get("label", ""))
        day_numbers.append(d_num if d_num is not None else (idx * 5 + 1))

    total_days = max(day_numbers[-1] - day_numbers[0], 1)
    values = [item["severity_score"] for item in entries]
    delta = float(values[-1] - values[0])
    rate_per_day = round(delta / total_days, 4)

    steps = [values[i] - values[i - 1] for i in range(1, len(values))]
    change_per_step = float(sum(steps) / len(steps))
    same_prediction = len({item["result"]["prediction"] for item in entries}) == 1

    cw = weather or {}
    c_rain = float(cw.get("rain_1h_mm", 0.0) or 0.0)
    c_hum  = float(cw.get("humidity_pct", 70.0) or 70.0)
    is_rainy = c_rain > 0 or c_hum >= 75

    if delta > 0.05:
        trend = "worsening"
        if is_rainy:
            explanation = (
                f"Accelerated Progression across {total_days} days (+{round(delta*100, 1)}% total, +{round(rate_per_day*100, 2)}%/day). "
                f"Persistent high humidity ({c_hum}%) and rainfall ({c_rain}mm) compounded fungal spore proliferation."
            )
        else:
            explanation = f"Lesion severity expanded by +{round(delta*100, 1)}% over {total_days} days (+{round(rate_per_day*100, 2)}%/day)."
    elif delta < -0.05:
        trend = "improving"
        explanation = (
            f"Therapeutic Recovery: Lesion coverage contracted by {abs(round(delta*100, 1))}% across {total_days} days "
            f"(-{abs(round(rate_per_day*100, 2))}%/day), confirming successful treatment."
        )
    else:
        trend = "stable / controlled"
        explanation = f"Disease severity remained stable ({round(delta*100, 1)}% change over {total_days} days)."

    if same_prediction:
        explanation += " The diagnosed disease stayed consistent across all stages."
    else:
        explanation += " Note: Detected disease class shifted across samples."

    return {
        "trend": trend,
        "delta": round(delta, 4),
        "total_days": total_days,
        "rate_per_day": rate_per_day,
        "change_per_step": round(change_per_step, 4),
        "same_prediction": same_prediction,
        "explanation": explanation,
    }
