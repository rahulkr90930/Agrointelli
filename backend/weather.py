"""
AgroIntelli — Weather & Microclimate Intelligence Module
Fetches live OpenWeatherMap microclimate data, normalizes weather feature vectors,
and correlates meteorological factors (rain, humidity, temperature) with disease progression.
"""

import os
import re
import requests
from datetime import datetime
from pathlib import Path

# Load environment variable for API key
OWM_API_KEY = os.environ.get("OWM_API_KEY", "0af35111bc4b73ebb0b71d505e063680")


def parse_day_num(label):
    """Extract numeric day index from strings like 'Day 1', 'Day 7', 'Day 14'."""
    if not label:
        return None
    match = re.search(r'\d+', str(label))
    return int(match.group()) if match else None


def normalize_weather_vector(weather_dict):
    """Normalize microclimate dict into standardized model input vector."""
    if not weather_dict:
        return [0.5, 0.7, 0.0, 0.1]
    temp = float(weather_dict.get("temp_c", 25.0) or 25.0)
    hum  = float(weather_dict.get("humidity_pct", 70.0) or 70.0)
    rain = float(weather_dict.get("rain_1h_mm", 0.0) or 0.0)
    wind = float(weather_dict.get("wind_kmh", 10.0) or 10.0)

    # Scale values to [0, 1] bounds
    temp_norm = max(0.0, min(1.0, (temp + 10.0) / 60.0))
    hum_norm  = max(0.0, min(1.0, hum / 100.0))
    rain_norm = max(0.0, min(1.0, rain / 50.0))
    wind_norm = max(0.0, min(1.0, wind / 80.0))
    return [temp_norm, hum_norm, rain_norm, wind_norm]


def fetch_live_weather_snapshot(lat=None, lon=None):
    """Fetch live weather using HTTPS OpenWeatherMap API with automatic Open-Meteo zero-key fallback."""
    city, country = "Local Field", "IN"
    if lat is None or lon is None:
        try:
            resp = requests.get("https://ipapi.co/json/", timeout=4)
            if resp.status_code == 200:
                loc_data = resp.json()
                lat = loc_data.get("latitude", 22.57)
                lon = loc_data.get("longitude", 88.36)
                city = loc_data.get("city", "Local Field")
                country = loc_data.get("country_code", "IN")
            else:
                raise ValueError("ipapi failed")
        except Exception:
            try:
                loc_data = requests.get("http://ip-api.com/json/", timeout=4).json()
                lat = loc_data.get("lat", 22.57)
                lon = loc_data.get("lon", 88.36)
                city = loc_data.get("city", "Local Field")
                country = loc_data.get("country", "IN")
            except Exception:
                lat, lon = 22.57, 88.36

    # 1. Try OpenWeatherMap HTTPS
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={OWM_API_KEY}&units=metric"
        )
        w_resp = requests.get(url, timeout=6)
        w_data = w_resp.json()
        if w_resp.status_code == 200 and w_data.get("cod") == 200:
            return {
                "success": True,
                "weather": {
                    "city": city or w_data.get("name", "Field Station"),
                    "country": country or w_data.get("sys", {}).get("country", ""),
                    "temp_c": w_data["main"]["temp"],
                    "feels_like_c": w_data["main"]["feels_like"],
                    "humidity_pct": w_data["main"]["humidity"],
                    "condition": w_data["weather"][0]["description"],
                    "wind_kmh": round(w_data["wind"]["speed"] * 3.6, 1),
                    "rain_1h_mm": w_data.get("rain", {}).get("1h", 0.0),
                    "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                },
            }
    except Exception as e:
        print(f"OpenWeatherMap API notice: {e}")

    # 2. Try Open-Meteo free HTTPS API (No API key needed)
    try:
        om_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true&hourly=relative_humidity_2m,precipitation"
        om_resp = requests.get(om_url, timeout=6)
        if om_resp.status_code == 200:
            om_data = om_resp.json()
            cw = om_data.get("current_weather", {})
            temp = cw.get("temperature", 25.0)
            wind = round(cw.get("windspeed", 10.0), 1)
            # Estimate humidity from hourly
            hourly_hum = om_data.get("hourly", {}).get("relative_humidity_2m", [70])
            hum = hourly_hum[0] if hourly_hum else 70
            hourly_precip = om_data.get("hourly", {}).get("precipitation", [0.0])
            rain = hourly_precip[0] if hourly_precip else 0.0

            return {
                "success": True,
                "weather": {
                    "city": city,
                    "country": country,
                    "temp_c": temp,
                    "feels_like_c": temp,
                    "humidity_pct": hum,
                    "condition": "live satellite microclimate",
                    "wind_kmh": wind,
                    "rain_1h_mm": rain,
                    "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                },
            }
    except Exception as e:
        print(f"Open-Meteo fallback notice: {e}")

    # 3. Default robust fallback
    return {
        "success": False,
        "weather": {
            "city": city,
            "country": country,
            "temp_c": 28.0,
            "feels_like_c": 30.0,
            "humidity_pct": 72,
            "condition": "partly cloudy",
            "wind_kmh": 12.0,
            "rain_1h_mm": 0.0,
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
    }


def evaluate_weather_progression(prev_weather, curr_weather, days_elapsed, prev_pred, curr_pred, prev_aff, curr_aff, curr_day_label="Current"):
    """
    Correlates elapsed days and meteorological conditions (rainfall, humidity, temperature)
    with disease trajectory according to plant pathology principles.
    """
    delta = round(curr_aff - prev_aff, 1)
    rate_per_day = round(delta / max(days_elapsed, 1), 2)

    pw = prev_weather or {}
    cw = curr_weather or {}

    p_rain = float(pw.get("rain_1h_mm", 0.0) or 0.0)
    p_hum  = float(pw.get("humidity_pct", 70.0) or 70.0)
    p_temp = float(pw.get("temp_c", 25.0) or 25.0)

    c_rain = float(cw.get("rain_1h_mm", 0.0) or 0.0)
    c_hum  = float(cw.get("humidity_pct", 70.0) or 70.0)
    c_temp = float(cw.get("temp_c", 25.0) or 25.0)

    # Weather indicators
    both_wet = (p_rain > 0 or p_hum >= 75) and (c_rain > 0 or c_hum >= 75)
    curr_wet = c_rain > 0 or c_hum >= 75
    curr_dry = c_rain == 0 and c_hum < 60

    if curr_pred != prev_pred:
        verdict = "NEW_DISEASE"
        status_tag = "⚠️ Condition Shift"
        expl = f"Pathology shifted from {prev_pred.replace('_',' ')} to {curr_pred.replace('_',' ')} over {days_elapsed} days."
        if curr_wet:
            expl += f" Sustained foliage wetness (Rain: {c_rain}mm, {c_hum}% humidity) provided an opportunistic entry pathway for secondary infection."
        else:
            expl += " Inspect affected tissue for secondary pathogens and revise management."
    elif delta >= 3.0:
        verdict = "WORSENED"
        status_tag = "🔴 Accelerated Progression"
        if both_wet:
            expl = (
                f"Critical Moisture Correlation: Both baseline and {curr_day_label} experienced high moisture and rainfall "
                f"({c_rain}mm rain, {c_hum}% humidity). Continuous leaf wetness over {days_elapsed} days accelerated fungal spore "
                f"germination and mycelial spread, expanding necrotic lesions by +{delta}% (+{rate_per_day}%/day)."
            )
        elif curr_wet:
            expl = (
                f"Wet Microclimate Risk: Recent precipitation ({c_rain}mm) and high humidity ({c_hum}%) over {days_elapsed} days "
                f"fueled pathogen sporulation, increasing affected leaf surface by +{delta}% (+{rate_per_day}%/day)."
            )
        else:
            expl = (
                f"Active lesion expansion observed: +{delta}% spread over {days_elapsed} days (+{rate_per_day}%/day). "
                f"Adjust fungicide application schedule."
            )
    elif delta <= -3.0:
        verdict = "IMPROVED"
        status_tag = "🟢 Significant Healing"
        if curr_dry:
            expl = (
                f"Favorable Arid Conditions: Lesion coverage contracted by {abs(delta)}% over {days_elapsed} days "
                f"(-{abs(rate_per_day)}%/day). Dry canopy conditions ({c_hum}% humidity, 0mm rain) suppressed airborne spore dispersal."
            )
        else:
            expl = (
                f"Positive Therapeutic Response: Lesion coverage contracted by {abs(delta)}% over {days_elapsed} days "
                f"(-{abs(rate_per_day)}%/day). Treatment has successfully contained pathogen proliferation."
            )
    else:
        verdict = "STABLE"
        status_tag = "🟡 Stable / Monitored"
        expl = (
            f"Lesion severity is steady ({delta > 0 and '+' or ''}{delta}% over {days_elapsed} days, {rate_per_day}%/day). "
            f"Continue routine surveillance."
        )

    return {
        "verdict": verdict,
        "status_tag": status_tag,
        "explanation": expl,
        "delta": delta,
        "days_elapsed": days_elapsed,
        "rate_per_day": rate_per_day,
        "weather_context": {
            "previous": {"rain": p_rain, "humidity": p_hum, "temp": p_temp},
            "current": {"rain": c_rain, "humidity": c_hum, "temp": c_temp},
            "consecutive_wet_days": both_wet
        }
    }
