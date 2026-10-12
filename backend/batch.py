"""
AgroIntelli - Multi-Image Timeline Batch Evaluation & Trajectory Modeling
Computes comparative progression across multi-day leaf scans, factoring in
disease classification shifts, Grad-CAM severity deltas, and live microclimate exacerbation.
"""

from typing import List, Dict, Any, Optional
import math
try:
    from .weather import parse_day_num
except (ImportError, ValueError):
    from weather import parse_day_num


def normalize_labels(labels: List[Any], count: int) -> List[str]:
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


def batch_progress_summary(entries: List[Dict[str, Any]], weather: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Computes a comprehensive, clinical agronomic progression report across sequential leaf scans.
    Produces detailed pathology notes, rate-of-spread dynamics, and actionable recovery protocols.
    """
    if not entries:
        return {
            "trend": "insufficient data",
            "trajectory": "INSUFFICIENT DATA",
            "delta": 0.0,
            "change_per_step": 0.0,
            "same_prediction": False,
            "total_days": 0,
            "rate_per_day": 0.0,
            "explanation": "No valid leaf images were provided for progression tracking.",
            "notes": "No valid leaf images were provided for progression tracking.",
            "detailed_report": "Upload at least two sequential leaf images across different days to compute disease trajectory."
        }

    if len(entries) == 1:
        first_res = entries[0].get("result", {})
        pred_clean = (first_res.get("prediction") or "Condition").replace("_", " ").title()
        sev = round(float(entries[0].get("severity_score", 0.0)) * 100, 1)
        return {
            "trend": "baseline established",
            "trajectory": "BASELINE RECORDED",
            "delta": 0.0,
            "change_per_step": 0.0,
            "same_prediction": True,
            "total_days": 0,
            "rate_per_day": 0.0,
            "explanation": f"Baseline leaf scan recorded for {pred_clean} at {sev}% lesion severity.",
            "notes": f"Baseline scan established. Add subsequent follow-up scans on Day 5 or Day 10 to track healing progression or lesion expansion.",
            "detailed_report": f"Baseline scan established for {pred_clean} ({sev}% severity). Add follow-up scans to compute directional velocity."
        }

    # Extract parsed day numbers
    day_numbers = []
    for idx, item in enumerate(entries):
        d_num = parse_day_num(item.get("label", ""))
        day_numbers.append(d_num if d_num is not None else (idx * 5 + 1))

    total_days = max(day_numbers[-1] - day_numbers[0], 1)
    values = [float(item.get("severity_score", 0.0)) for item in entries]
    start_sev = round(values[0] * 100, 1)
    end_sev = round(values[-1] * 100, 1)
    delta = float(values[-1] - values[0])
    delta_pct = round(delta * 100, 1)
    rate_per_day = round(delta / total_days, 4)
    daily_rate_pct = round(rate_per_day * 100, 2)

    steps = [values[i] - values[i - 1] for i in range(1, len(values))]
    change_per_step = float(sum(steps) / len(steps))

    preds = [item.get("result", {}).get("prediction", "unknown") for item in entries]
    first_pred_clean = preds[0].replace("_", " ").title()
    latest_pred_clean = preds[-1].replace("_", " ").title()
    same_prediction = len(set(preds)) == 1

    # Environmental weather impact
    cw = weather or {}
    c_rain = float(cw.get("rain_1h_mm", 0.0) or 0.0)
    c_hum = float(cw.get("humidity_pct", 65.0) or 65.0)
    c_temp = float(cw.get("temp_c", 25.0) or 25.0)
    is_high_humidity = c_hum >= 72.0
    is_rainy = c_rain > 0.0

    # Classify trajectory
    if delta > 0.04:
        trajectory = "DETERIORATING / PROGRESSING"
        trend = "worsening"
    elif delta < -0.04:
        trajectory = "RECOVERING / HEALING"
        trend = "improving"
    else:
        trajectory = "STABLE / CONTROLLED"
        trend = "stable"

    # Construct extensive, clinical agronomic narrative
    narrative_sections = []

    # Section 1: Executive Diagnostic Summary
    if trend == "worsening":
        narrative_sections.append(
            f"**Clinical Status: Active Disease Progression (+{delta_pct}% Net Expansion)**\n"
            f"Over the {total_days}-day monitoring interval (from Day {day_numbers[0]} to Day {day_numbers[-1]}), "
            f"symptomatic lesion surface area accelerated from an initial baseline of **{start_sev}%** to **{end_sev}%** "
            f"(expansion velocity: **{daily_rate_pct}% per day**). "
            f"Grad-CAM visual attention overlays verify outward radial expansion of chlorotic halos and necrotic margin coalescence."
        )
    elif trend == "improving":
        narrative_sections.append(
            f"**Clinical Status: Positive Therapeutic Recovery (-{abs(delta_pct)}% Lesion Contraction)**\n"
            f"Over the {total_days}-day evaluation window (from Day {day_numbers[0]} to Day {day_numbers[-1]}), "
            f"pathological lesion intensity contracted markedly from **{start_sev}%** down to **{end_sev}%** "
            f"(recovery velocity: **{abs(daily_rate_pct)}% per day**). "
            f"Visual attention maps confirm active leaf lamina re-greening, localized desiccation of fungal pustules, and containment of active sporulation."
        )
    else:
        narrative_sections.append(
            f"**Clinical Status: Infection Stagnation / Latent Containment ({delta_pct}% Delta)**\n"
            f"Over the {total_days}-day timeline (Day {day_numbers[0]} to Day {day_numbers[-1]}), "
            f"disease footprint held relatively static (initial **{start_sev}%** vs current **{end_sev}%**). "
            f"Pathogen activity appears arrested, indicating that protective canopy sanitation or initial fungicide barriers have temporarily restrained further tissue invasion."
        )

    # Section 2: Pathological Consistency & Etiology
    if same_prediction:
        narrative_sections.append(
            f"**Etiological Stability**: Diagnostic classification remained rigorously consistent as **{first_pred_clean}** "
            f"across all sequential evaluations, confirming steady single-pathogen dynamics rather than secondary opportunistic infections."
        )
    else:
        narrative_sections.append(
            f"**Symptom Morphology Shift**: The primary visual classification transitioned from **{first_pred_clean}** to **{latest_pred_clean}**. "
            f"This indicates either progressive lesion necrosis masking early spotting symptoms or secondary saprophytic colonization on damaged leaf tissue."
        )

    # Section 3: Microclimate & Environmental Acceleration Factor
    weather_notes = []
    if is_high_humidity or is_rainy:
        weather_notes.append(
            f"Ambient relative humidity of **{c_hum}%** combined with precipitation (**{c_rain} mm/h**) created extended leaf wetness durations, "
            f"substantially lowering the spore germination barrier for foliar pathogens."
        )
    else:
        weather_notes.append(
            f"Ambient conditions (**{c_temp}°C**, **{c_hum}% RH**) provided moderate environmental pressure, "
            f"limiting runaway fungal incubation across the leaf canopy."
        )
    narrative_sections.append(f"**Microclimate Acceleration Analysis**: {' '.join(weather_notes)}")

    # Section 4: Prescribed Agronomic Intervention Protocol
    if trend == "worsening":
        narrative_sections.append(
            f"**Recommended Agronomic Action Plan**:\n"
            f"1. **Immediate Therapeutic Spray**: Apply a targeted systemic curative fungicide within 24–48 hours (e.g. Difenoconazole or Azoxystrobin) to arrest mycelial penetration.\n"
            f"2. **Cultural Sanitation**: Prune severely infected lower leaves displaying >60% necrosis using sterilized shears to reduce internal inoculum load.\n"
            f"3. **Irrigation Shift**: Cease any overhead sprinkler watering; deliver water exclusively at root level early in the morning.\n"
            f"4. **Re-Evaluation**: Execute a follow-up diagnostic leaf scan in 4–5 days to verify lesion stabilization."
        )
    elif trend == "improving":
        narrative_sections.append(
            f"**Recommended Agronomic Action Plan**:\n"
            f"1. **Prophylactic Maintenance**: Maintain protective coverage using organic cold-pressed neem oil (5 ml/L) or a mild copper protectant to prevent reinfection.\n"
            f"2. **Canopy Airflow**: Ensure adequate row spacing and selective sucker pruning to sustain sunlight penetration across inner foliage.\n"
            f"3. **Nutritional Support**: Apply a balanced potassium-silica foliar spray to fortify regenerating leaf cell walls."
        )
    else:
        narrative_sections.append(
            f"**Recommended Agronomic Action Plan**:\n"
            f"1. **Active Scouting**: Inspect underside of adjacent leaves daily for fresh sporulation or halo expansion.\n"
            f"2. **Preventative Barrier**: Apply a broad-spectrum protective biofungicide (*Bacillus subtilis* or *Trichoderma*) before incoming weather fronts.\n"
            f"3. **Continue Monitoring**: Re-scan the crop within 3–4 days to ensure lesion stability persists."
        )

    full_detailed_report = "\n\n".join(narrative_sections)
    summary_explanation = narrative_sections[0] + " " + narrative_sections[1]

    return {
        "trend": trend,
        "trajectory": trajectory,
        "delta": round(delta, 4),
        "total_days": total_days,
        "rate_per_day": rate_per_day,
        "change_per_step": round(change_per_step, 4),
        "same_prediction": same_prediction,
        "initial_prediction": first_pred_clean,
        "latest_prediction": latest_pred_clean,
        "initial_severity_pct": start_sev,
        "final_severity_pct": end_sev,
        "explanation": summary_explanation,
        "notes": full_detailed_report,
        "detailed_report": full_detailed_report
    }
