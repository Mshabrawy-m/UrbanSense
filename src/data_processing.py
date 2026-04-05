"""
data_processing.py
------------------
Builds the training dataset from public APIs (no key required):

Real data (Open-Meteo):
  - Weather Archive — temperature, wind, precipitation, humidity
  - Air Quality Archive — hourly PM2.5 (μg/m³) when the API returns values

Modeled (no global open hourly traffic / noise sensor grid):
  - Traffic_Count — deterministic time-of-day + city profile
  - Noise_Level_dB — physics-inspired model from traffic + weather (training target)

If weather or air-quality fetch fails, documented fallbacks are used; see `src.data_quality.check_processed_file`.
"""

import logging
import time
import requests
import pandas as pd
import numpy as np
logger = logging.getLogger(__name__)

# ── City configuration ─────────────────────────────────────────────────────────
CITIES = {
    "New York": {"lat": 40.7128, "lon": -74.0060, "base_noise": 68, "base_traffic": 1200, "traffic_mult": 1.2},
    "London":   {"lat": 51.5074, "lon": -0.1278,  "base_noise": 64, "base_traffic": 1000, "traffic_mult": 1.0},
    "Cairo":    {"lat": 30.0444, "lon":  31.2357,  "base_noise": 72, "base_traffic": 1500, "traffic_mult": 1.4},
    "Tokyo":    {"lat": 35.6762, "lon": 139.6503,  "base_noise": 62, "base_traffic": 1800, "traffic_mult": 1.5},
    "Paris":    {"lat": 48.8566, "lon":   2.3522,  "base_noise": 65, "base_traffic": 1100, "traffic_mult": 1.1},
}

START_DATE  = "2024-01-01"
END_DATE    = "2024-06-30"

# Paths relative to project root
DATA_RAW       = "data/data.csv"
DATA_PROCESSED = "data/processed_data.csv"

# Required columns for schema validation
REQUIRED_COLS = {"DateTime", "City", "Noise_Level_dB", "Traffic_Count",
                 "temperature", "wind_speed", "precipitation", "humidity", "pm25"}


# ── Schema validation ──────────────────────────────────────────────────────────
def validate_schema(df: pd.DataFrame, source: str = "CSV") -> None:
    """Raises ValueError with a clear message if required columns are missing."""
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"[Schema Error] {source} is missing required columns: {sorted(missing)}. "
            f"Found: {sorted(df.columns.tolist())}"
        )


# ── Real weather fetch ─────────────────────────────────────────────────────────
def fetch_real_weather(city: str, cfg: dict) -> pd.DataFrame:
    """
    Fetches real hourly weather from Open-Meteo Archive API.
    Returns a DataFrame with columns: DateTime, temperature, wind_speed,
    precipitation, humidity.
    """
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={cfg['lat']}&longitude={cfg['lon']}"
        f"&start_date={START_DATE}&end_date={END_DATE}"
        "&hourly=temperature_2m,wind_speed_10m,precipitation,relative_humidity_2m"
        "&timezone=auto"
    )
    print(f"  Fetching real weather for {city}...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        h = r.json()["hourly"]
        df = pd.DataFrame({
            "DateTime":      pd.to_datetime(h["time"]),
            "temperature":   h["temperature_2m"],
            "wind_speed":    h["wind_speed_10m"],
            "precipitation": h["precipitation"],
            "humidity":      h["relative_humidity_2m"],
        })
        df.dropna(inplace=True)
        print(f"OK — {len(df)} rows")
        return df
    except Exception as e:
        logger.warning("Open-Meteo archive failed for %s: %s — using fallback", city, e)
        print(f"FAILED ({e}) — using synthetic fallback")
        return _synthetic_weather_fallback(cfg)


def _synthetic_weather_fallback(cfg: dict) -> pd.DataFrame:
    """Fallback if API is unreachable."""
    np.random.seed(42)
    dates  = pd.date_range(START_DATE, END_DATE, freq="h")
    months = dates.month
    hours  = dates.hour
    seasonal = 15 + 10 * np.sin((months - 3) * np.pi / 6)
    temp   = seasonal + 5 * np.sin((hours - 6) * np.pi / 12) + np.random.normal(0, 1.5, len(dates))
    wind   = np.maximum(0, np.random.normal(12, 5, len(dates)))
    precip = np.maximum(0, np.random.exponential(0.3, len(dates)))
    humid  = np.clip(60 + np.random.normal(0, 15, len(dates)), 20, 100)
    return pd.DataFrame({
        "DateTime":      dates,
        "temperature":   temp.round(1),
        "wind_speed":    wind.round(1),
        "precipitation": precip.round(2),
        "humidity":      humid.round(1),
    })


# ── Real PM2.5 (Open-Meteo Air Quality API) ───────────────────────────────────
def fetch_air_quality_pm25(lat: float, lon: float) -> pd.DataFrame:
    """
    Hourly PM2.5 from Open-Meteo Air Quality API (CAMS / merged reanalysis).
    Same date range as weather archive. Left-merge in build_dataset; gaps use estimate_pm25.
    """
    url = (
        "https://air-quality-api.open-meteo.com/v1/air-quality"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={START_DATE}&end_date={END_DATE}"
        "&hourly=pm2_5&timezone=auto"
    )
    print("  Fetching real PM2.5 (air quality API)...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        h = r.json().get("hourly") or {}
        if not h.get("time"):
            print("empty response — will estimate PM2.5")
            return pd.DataFrame(columns=["DateTime", "pm2_5"])
        df = pd.DataFrame({
            "DateTime": pd.to_datetime(h["time"]),
            "pm2_5":    pd.to_numeric(h.get("pm2_5"), errors="coerce"),
        })
        print(f"OK — {len(df)} rows")
        return df
    except Exception as e:
        logger.warning("Air quality archive failed: %s", e)
        print(f"FAILED ({e}) — will estimate PM2.5")
        return pd.DataFrame(columns=["DateTime", "pm2_5"])


# ── Traffic simulation ─────────────────────────────────────────────────────────
def simulate_traffic(hour: int, dow: int, city_cfg: dict) -> int:
    """
    Deterministic traffic model seeded by (hour, dow) for reproducibility.
    Rush hours: 7–9, 16–19 | Night: 0–5 | Weekend reduction.
    """
    rng = np.random.default_rng(seed=hour + dow * 24)
    is_weekend = dow >= 5
    if 7 <= hour <= 9 or 16 <= hour <= 19:
        base = int(rng.integers(1200, 2000))
    elif 0 <= hour <= 5:
        base = int(rng.integers(50, 300))
    elif 10 <= hour <= 15:
        base = int(rng.integers(600, 1100))
    else:
        base = int(rng.integers(400, 800))
    if is_weekend:
        base = int(base * 0.65)
    return max(50, int(base * city_cfg["traffic_mult"]))


# ── PM2.5 estimation (deterministic) ──────────────────────────────────────────
def estimate_pm25(traffic: int, humidity: float, precipitation: float) -> float:
    """
    Deterministic empirical PM2.5 estimate.
    - Traffic → primary particulate source
    - Humidity > 50% → hygroscopic growth (Tai et al. 2010)
    - Precipitation → washout effect
    No random component — same inputs always produce same output.
    """
    base    = traffic / 80.0
    humid_f = 1.0 + 0.008 * max(0, humidity - 50)
    rain_f  = max(0.3, 1.0 - 0.4 * min(precipitation, 2.0))
    return round(max(3.0, base * humid_f * rain_f + 3.0), 1)


# ── Noise model ───────────────────────────────────────────────────────────────
def estimate_noise(traffic: int, hour: int, dow: int,
                   wind_speed: float, precipitation: float,
                   city_cfg: dict) -> float:
    """Empirical urban noise model (ISO 9613 inspired). Seeded for reproducibility."""
    rng        = np.random.default_rng(seed=hour + dow * 24 + traffic)
    is_weekend = int(dow >= 5)
    is_rush    = int(7 <= hour <= 9 or 16 <= hour <= 19)
    is_night   = int(0 <= hour <= 5)
    noise = (
        city_cfg["base_noise"]
        + 8.0 * np.log10(max(traffic, 1) / city_cfg["base_traffic"] + 1)
        + 2.5 * is_rush
        - 4.0 * is_weekend
        - 6.0 * is_night
        + (1.5 if 0.1 < precipitation <= 2.0 else 0)
        - (1.0 if precipitation > 2.0 else 0)
        - (0.8 if wind_speed > 25 else 0)
        + float(rng.normal(0, 2.0))
    )
    return round(float(np.clip(noise, 42, 95)), 1)


# ── Main builder ──────────────────────────────────────────────────────────────
def build_dataset(output_path: str = DATA_RAW) -> pd.DataFrame:
    """Real weather + real hourly PM2.5 when available; modeled traffic + noise label."""
    print("=" * 60)
    print("  Building dataset — Open-Meteo: weather + air quality (PM2.5)")
    print(f"  Period: {START_DATE} → {END_DATE}")
    print("=" * 60)

    all_records = []
    n_pm25_api = 0
    n_pm25_est = 0

    for city, cfg in CITIES.items():
        weather_df = fetch_real_weather(city, cfg)
        weather_df["DateTime"] = pd.to_datetime(weather_df["DateTime"])

        aq_df = fetch_air_quality_pm25(cfg["lat"], cfg["lon"])
        if not aq_df.empty:
            aq_df["DateTime"] = pd.to_datetime(aq_df["DateTime"])
            merged = weather_df.merge(aq_df, on="DateTime", how="left")
        else:
            merged = weather_df.copy()
            merged["pm2_5"] = np.nan

        merged = merged.set_index("DateTime")

        for ts, row in merged.iterrows():
            hour    = ts.hour
            dow     = ts.weekday()
            traffic = simulate_traffic(hour, dow, cfg)
            pm25_raw = row["pm2_5"] if "pm2_5" in row.index else np.nan
            if pd.notna(pm25_raw):
                pm25 = round(float(pm25_raw), 1)
                src = "open_meteo_aq"
                n_pm25_api += 1
            else:
                pm25 = estimate_pm25(traffic, float(row["humidity"]), float(row["precipitation"]))
                src = "estimated"
                n_pm25_est += 1

            noise = estimate_noise(traffic, hour, dow,
                                   float(row["wind_speed"]), float(row["precipitation"]), cfg)
            all_records.append({
                "DateTime":       ts,
                "City":           city,
                "Noise_Level_dB": noise,
                "Traffic_Count":  traffic,
                "temperature":    round(float(row["temperature"]), 1),
                "wind_speed":     round(float(row["wind_speed"]), 1),
                "precipitation":  round(float(row["precipitation"]), 2),
                "humidity":       round(float(row["humidity"]), 1),
                "pm25":           pm25,
                "pm25_source":    src,
            })
        time.sleep(0.35)

    df = pd.DataFrame(all_records)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df):,} rows to {output_path}")
    print(
        f"PM2.5: {n_pm25_api:,} hourly values from Open-Meteo Air Quality API; "
        f"{n_pm25_est:,} filled with traffic/weather estimate (merge gaps / API miss)."
    )
    return df


def load_and_process_data(csv_path: str = DATA_RAW) -> pd.DataFrame:
    """Loads raw data, validates schema, engineers all ML features."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} rows.")

    # ── Schema validation ──────────────────────────────────────────────────
    validate_schema(df, source=csv_path)

    df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    df.dropna(subset=["DateTime", "Noise_Level_dB", "Traffic_Count"], inplace=True)

    df["hour"]         = df["DateTime"].dt.hour
    df["day_of_week"]  = df["DateTime"].dt.dayofweek
    df["month"]        = df["DateTime"].dt.month
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].apply(lambda h: int(7 <= h <= 9 or 16 <= h <= 19))
    df["is_night"]     = df["hour"].apply(lambda h: int(0 <= h <= 5))

    for col in ["Noise_Level_dB", "Traffic_Count", "temperature",
                "wind_speed", "precipitation", "humidity", "pm25"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["Noise_Level_dB", "Traffic_Count"], inplace=True)
    df["City"] = df["City"].fillna("Unknown").astype(str)

    # ── Enhanced feature engineering ──────────────────────────────────────
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["heat_index"] = (
        -8.78469475556
        + 1.61139411    * df["temperature"]
        + 2.33854883889 * df["humidity"]
        - 0.14611605    * df["temperature"] * df["humidity"]
        - 0.012308094   * df["temperature"] ** 2
        - 0.0164248277778 * df["humidity"] ** 2
        + 0.002211732   * df["temperature"] ** 2 * df["humidity"]
        + 0.00072546    * df["temperature"] * df["humidity"] ** 2
        - 0.000003582   * df["temperature"] ** 2 * df["humidity"] ** 2
    ).round(1)

    # Vectorized traffic_ratio (no row-by-row apply)
    city_baselines = pd.Series({
        "New York": 1200, "London": 1000, "Cairo": 1500,
        "Tokyo": 1800,    "Paris":  1100,
    })
    df["traffic_ratio"] = (
        df["Traffic_Count"] / df["City"].map(city_baselines).fillna(1200)
    ).round(3)

    # Log-transformed traffic (compresses large values, improves tree splits)
    df["traffic_log"] = np.log10(df["Traffic_Count"].clip(lower=1)).round(4)

    # City base noise level (domain knowledge constant per city)
    city_base_noise_map = pd.Series({
        "New York": 68.0, "London": 64.0, "Cairo": 72.0,
        "Tokyo": 62.0,    "Paris":  65.0,
    })
    df["city_base_noise"] = df["City"].map(city_base_noise_map).fillna(65.0)

    def noise_zone(db):
        if db < 55: return 0
        if db < 65: return 1
        if db < 75: return 2
        return 3
    df["noise_zone"] = df["Noise_Level_dB"].apply(noise_zone)

    df = df.sort_values(["City", "DateTime"])
    df["noise_delta_3h"] = (
        df.groupby("City")["Noise_Level_dB"]
          .transform(lambda x: x.diff(3).round(2))
    ).fillna(0)

    df.to_csv(DATA_PROCESSED, index=False)
    print(f"Saved {DATA_PROCESSED} — {len(df):,} rows, {len(df.columns)} columns")
    return df


if __name__ == "__main__":
    build_dataset(DATA_RAW)
    load_and_process_data(DATA_RAW)
