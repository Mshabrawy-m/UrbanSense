"""
api_integration.py
──────────────────
Real-time data layer for the Smart Urban Noise Analyzer.

Real data sources (Open-Meteo, no API key):
  - Weather Forecast API     → temperature, wind, precipitation, humidity
  - Air Quality API (current) → PM2.5 (μg/m³) when available

Modeled:
  - Traffic_Count            → deterministic simulation (hour, weekend, city)
  - PM2.5 fallback           → empirical estimate from traffic + weather if AQ fails
"""

import math
import logging
import random
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

CITY_COORDS = {
    "New York": (40.7128, -74.0060),
    "London":   (51.5074, -0.1278),
    "Cairo":    (30.0444,  31.2357),
    "Tokyo":    (35.6762, 139.6503),
    "Paris":    (48.8566,   2.3522),
}

_TRAFFIC_MULT = {
    "New York": 1.2, "London": 1.0, "Cairo": 1.4,
    "Tokyo":    1.5, "Paris":  1.1,
}

_BASE_TRAFFIC = {
    "New York": 1200, "London": 1000, "Cairo": 1500,
    "Tokyo":    1800, "Paris":  1100,
}

_CITY_BASE_NOISE = {
    "New York": 68.0, "London": 64.0, "Cairo": 72.0,
    "Tokyo":    62.0, "Paris":  65.0,
}


# ── Traffic simulation (deterministic) ────────────────────────────────────────
def _simulate_traffic(hour: int, is_weekend: bool, city: str) -> int:
    """
    Deterministic traffic estimate seeded by (hour, is_weekend, city).
    Same inputs always return the same value — no UI instability on rerun.
    """
    seed = hash((hour, is_weekend, city)) % (2 ** 32)
    rng  = random.Random(seed)
    mult = _TRAFFIC_MULT.get(city, 1.0)

    if 7 <= hour <= 9 or 16 <= hour <= 19:
        base = rng.randint(1200, 2000)
    elif 0 <= hour <= 5:
        base = rng.randint(50, 300)
    elif 10 <= hour <= 15:
        base = rng.randint(600, 1100)
    else:
        base = rng.randint(400, 800)

    if is_weekend:
        base = int(base * 0.65)
    return max(50, int(base * mult))


# ── Real weather fetch ─────────────────────────────────────────────────────────
def get_weather(lat: float, lon: float) -> dict:
    """
    Fetches current weather from Open-Meteo (free, no API key).
    Returns a dict with keys: temperature, wind_speed, precipitation, humidity.
    Sets '_is_fallback': True if the API call failed.
    """
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&current=temperature_2m,wind_speed_10m,precipitation,relative_humidity_2m"
        )
        r = requests.get(url, timeout=7)
        r.raise_for_status()
        current = r.json().get("current", {})
        return {
            "temperature":   current.get("temperature_2m",       15.0),
            "wind_speed":    current.get("wind_speed_10m",        10.0),
            "precipitation": current.get("precipitation",          0.0),
            "humidity":      current.get("relative_humidity_2m",  60.0),
            "_is_fallback":  False,
        }
    except Exception as e:
        logger.warning("Open-Meteo forecast failed for (%.4f, %.4f): %s", lat, lon, e)
        return {
            "temperature": 15.0, "wind_speed": 10.0,
            "precipitation": 0.0, "humidity": 60.0,
            "_is_fallback": True,
        }


# ── Real PM2.5 (Open-Meteo Air Quality API, current) ──────────────────────────
def get_air_quality_current(lat: float, lon: float) -> dict:
    """
    Latest PM2.5 from Open-Meteo Air Quality API.
    Returns { "pm2_5": float | None, "_is_fallback": bool }.
    """
    try:
        url = (
            "https://air-quality-api.open-meteo.com/v1/air-quality"
            f"?latitude={lat}&longitude={lon}&current=pm2_5"
        )
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        cur = r.json().get("current") or {}
        v = cur.get("pm2_5")
        if v is None:
            return {"pm2_5": None, "_is_fallback": True}
        return {"pm2_5": float(v), "_is_fallback": False}
    except Exception as e:
        logger.warning("Open-Meteo air quality (current) failed: %s", e)
        return {"pm2_5": None, "_is_fallback": True}


# ── PM2.5 estimation (deterministic) ──────────────────────────────────────────
def estimate_pm25(traffic: int, humidity: float, precipitation: float) -> float:
    """
    Deterministic empirical PM2.5 estimate.
    Reference: Tai et al. (2010) Atmos. Chem. Phys.
    """
    base    = traffic / 80.0
    humid_f = 1.0 + 0.008 * max(0, humidity - 50)
    rain_f  = max(0.3, 1.0 - 0.4 * min(precipitation, 2.0))
    return round(max(3.0, base * humid_f * rain_f + 3.0), 1)


# ── Engineered feature helpers ─────────────────────────────────────────────────
def _compute_engineered(hour: int, month: int, temp: float, humid: float,
                         traffic: int, city: str) -> tuple:
    """Returns (hour_sin, hour_cos, month_sin, month_cos, heat_index,
                traffic_ratio, traffic_log, city_base_noise)."""
    hour_sin  = round(math.sin(2 * math.pi * hour  / 24), 6)
    hour_cos  = round(math.cos(2 * math.pi * hour  / 24), 6)
    month_sin = round(math.sin(2 * math.pi * month / 12), 6)
    month_cos = round(math.cos(2 * math.pi * month / 12), 6)

    if temp > 20 and humid > 40:
        heat_index = round(
            -8.78469475556
            + 1.61139411    * temp
            + 2.33854883889 * humid
            - 0.14611605    * temp * humid
            - 0.012308094   * temp ** 2
            - 0.0164248277778 * humid ** 2
            + 0.002211732   * temp ** 2 * humid
            + 0.00072546    * temp * humid ** 2
            - 0.000003582   * temp ** 2 * humid ** 2,
            1
        )
    else:
        heat_index = round(temp, 1)

    traffic_ratio    = round(traffic / _BASE_TRAFFIC.get(city, 1200), 3)
    traffic_log      = round(math.log10(max(traffic, 1)), 4)
    city_base_noise  = _CITY_BASE_NOISE.get(city, 65.0)

    return hour_sin, hour_cos, month_sin, month_cos, heat_index, traffic_ratio, traffic_log, city_base_noise


# ── Main feature builder ───────────────────────────────────────────────────────
def get_realtime_features(city: str) -> list:
    """
    Returns the full 18-feature vector matching FEATURES in model_training.py:
    [day_of_week, is_weekend, is_rush_hour, is_night,
     Traffic_Count, temperature, wind_speed, precipitation, humidity, pm25,
     hour_sin, hour_cos, month_sin, month_cos,
     heat_index, traffic_ratio, traffic_log, city_base_noise]
    """
    now        = datetime.now()
    hour       = now.hour
    dow        = now.weekday()
    month      = now.month
    is_weekend = int(dow >= 5)
    is_rush    = int(7 <= hour <= 9 or 16 <= hour <= 19)
    is_night   = int(0 <= hour <= 5)

    lat, lon = CITY_COORDS.get(city, (40.7128, -74.0060))
    traffic  = _simulate_traffic(hour, bool(is_weekend), city)
    weather  = get_weather(lat, lon)
    aq       = get_air_quality_current(lat, lon)
    if aq["pm2_5"] is not None and not aq.get("_is_fallback", True):
        pm25 = round(float(aq["pm2_5"]), 1)
    else:
        pm25 = estimate_pm25(traffic, weather["humidity"], weather["precipitation"])

    eng = _compute_engineered(hour, month, weather["temperature"],
                               weather["humidity"], traffic, city)
    # eng = (hour_sin, hour_cos, month_sin, month_cos, heat_index,
    #        traffic_ratio, traffic_log, city_base_noise)

    return [
        dow, is_weekend, is_rush, is_night,
        traffic,
        weather["temperature"],
        weather["wind_speed"],
        weather["precipitation"],
        weather["humidity"],
        pm25,
        *eng,   # 8 engineered features
    ]


def get_weather_display(city: str) -> dict:
    """
    Returns a display dict for the UI.
    Reuses the same weather fetch — no second API call.
    Includes '_is_fallback' flag so the UI can warn the user.
    """
    lat, lon   = CITY_COORDS.get(city, (40.7128, -74.0060))
    now        = datetime.now()
    hour       = now.hour
    dow        = now.weekday()
    is_weekend = dow >= 5
    w          = get_weather(lat, lon)
    traffic    = _simulate_traffic(hour, is_weekend, city)
    aq         = get_air_quality_current(lat, lon)
    if aq["pm2_5"] is not None and not aq.get("_is_fallback", True):
        pm25 = round(float(aq["pm2_5"]), 1)
        pm25_src = "Open-Meteo Air Quality (real-time PM2.5)"
    else:
        pm25 = estimate_pm25(traffic, w["humidity"], w["precipitation"])
        pm25_src = "Estimated (traffic + humidity + precipitation)"
    return {
        "temperature":   w["temperature"],
        "wind_speed":    w["wind_speed"],
        "precipitation": w["precipitation"],
        "humidity":      w["humidity"],
        "pm25":          pm25,
        "pm25_source":   pm25_src,
        "traffic":       traffic,
        "is_raining":    w["precipitation"] > 0.1,
        "_is_fallback":  w["_is_fallback"],
        "source":        "Open-Meteo (live)" if not w["_is_fallback"] else "Fallback defaults",
    }


if __name__ == "__main__":
    for city in CITY_COORDS:
        feats = get_realtime_features(city)
        print(f"{city}: {feats}")
