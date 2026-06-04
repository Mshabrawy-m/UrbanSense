"""
src/insights.py
───────────────
Peak noise detection and automatic insight generation engine.

All functions are modular and return structured data — no Streamlit
dependencies here, so they can be used in the chatbot, API, or UI.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional


# ── Noise thresholds (WHO) ─────────────────────────────────────────────────────
THRESHOLDS = {
    "quiet":     55,
    "moderate":  65,
    "loud":      75,
    "very_loud": 85,
}

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March",    4: "April",
    5: "May",     6: "June",     7: "July",      8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


# ══════════════════════════════════════════════════════════════════════════════
#  PEAK DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_peak_hours(df: pd.DataFrame, city: Optional[str] = None,
                      top_n: int = 3) -> pd.DataFrame:
    """
    Returns the top N noisiest hours of the day (averaged across all data).
    If city is given, filters to that city first.
    """
    d = df[df["City"] == city] if city else df
    hourly = (d.groupby("hour")["Noise_Level_dB"]
               .agg(["mean", "max", "std", "count"])
               .reset_index())
    hourly.columns = ["hour", "avg_db", "max_db", "std_db", "count"]
    hourly = hourly.sort_values("avg_db", ascending=False).head(top_n)
    return hourly.reset_index(drop=True)


def detect_peak_days(df: pd.DataFrame, city: Optional[str] = None) -> pd.DataFrame:
    """Returns average noise per day of week, sorted noisiest first."""
    d = df[df["City"] == city] if city else df
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday",
                 "Friday", "Saturday", "Sunday"]
    daily = (d.groupby("day_of_week")["Noise_Level_dB"]
              .mean().reset_index())
    daily["day_name"] = daily["day_of_week"].map(lambda x: day_names[x])
    return daily.sort_values("Noise_Level_dB", ascending=False).reset_index(drop=True)


def detect_peak_months(df: pd.DataFrame, city: Optional[str] = None) -> pd.DataFrame:
    """Returns average noise per month, sorted noisiest first."""
    d = df[df["City"] == city] if city else df
    monthly = (d.groupby("month")["Noise_Level_dB"]
                .mean().reset_index())
    monthly["month_name"] = monthly["month"].map(MONTH_NAMES)
    return monthly.sort_values("Noise_Level_dB", ascending=False).reset_index(drop=True)


def detect_city_peaks(df: pd.DataFrame) -> pd.DataFrame:
    """Returns peak hour and peak noise for each city."""
    rows = []
    for city in df["City"].unique():
        d = df[df["City"] == city]
        hourly_avg = d.groupby("hour")["Noise_Level_dB"].mean()
        peak_hour  = int(hourly_avg.idxmax())
        peak_db    = round(hourly_avg.max(), 1)
        quiet_hour = int(hourly_avg.idxmin())
        quiet_db   = round(hourly_avg.min(), 1)
        overall_avg = round(d["Noise_Level_dB"].mean(), 1)
        rows.append({
            "City":       city,
            "Peak Hour":  f"{peak_hour:02d}:00",
            "Peak dB":    peak_db,
            "Quiet Hour": f"{quiet_hour:02d}:00",
            "Quiet dB":   quiet_db,
            "Avg dB":     overall_avg,
        })
    return pd.DataFrame(rows).sort_values("Peak dB", ascending=False).reset_index(drop=True)


def detect_exceedances(df: pd.DataFrame, threshold: float = 70.0,
                       city: Optional[str] = None) -> dict:
    """
    Returns statistics about how often noise exceeds a given threshold.
    """
    d = df[df["City"] == city] if city else df
    total     = len(d)
    exceeded  = (d["Noise_Level_dB"] > threshold).sum()
    pct       = round(100 * exceeded / total, 1) if total > 0 else 0
    by_hour   = (d[d["Noise_Level_dB"] > threshold]
                 .groupby("hour").size()
                 .sort_values(ascending=False))
    worst_hour = int(by_hour.index[0]) if len(by_hour) > 0 else None
    return {
        "threshold":   threshold,
        "total":       total,
        "exceeded":    int(exceeded),
        "pct":         pct,
        "worst_hour":  worst_hour,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  AUTOMATIC INSIGHT GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_dataset_insights(df: pd.DataFrame) -> list[dict]:
    """
    Generates a list of human-readable insight dicts from the full dataset.
    Each dict has: { "icon", "title", "text", "severity" }
    severity: "info" | "warning" | "success" | "error"
    """
    insights = []

    # ── Peak hour insight ──────────────────────────────────────────────────
    hourly_avg = df.groupby("hour")["Noise_Level_dB"].mean()
    peak_hour  = int(hourly_avg.idxmax())
    peak_db    = round(hourly_avg.max(), 1)
    quiet_hour = int(hourly_avg.idxmin())
    quiet_db   = round(hourly_avg.min(), 1)

    insights.append({
        "icon": "🔔",
        "title": f"Peak noise at {peak_hour:02d}:00",
        "text": (f"The noisiest hour across all cities is {peak_hour:02d}:00 "
                 f"with an average of {peak_db} dB. "
                 f"This coincides with {'morning rush hour.' if 7 <= peak_hour <= 9 else 'evening rush hour.' if 16 <= peak_hour <= 19 else 'daytime activity.'}"),
        "severity": "warning" if peak_db > 65 else "info",
    })

    insights.append({
        "icon": "🌙",
        "title": f"Quietest hour at {quiet_hour:02d}:00",
        "text": (f"Noise drops to {quiet_db} dB at {quiet_hour:02d}:00 — "
                 f"{'well within WHO safe levels.' if quiet_db < 55 else 'still above WHO quiet threshold of 55 dB.'}"),
        "severity": "success" if quiet_db < 55 else "info",
    })

    # ── Loudest / quietest city ────────────────────────────────────────────
    city_avg  = df.groupby("City")["Noise_Level_dB"].mean()
    loudest   = city_avg.idxmax()
    quietest  = city_avg.idxmin()
    loud_db   = round(city_avg.max(), 1)
    quiet_c_db = round(city_avg.min(), 1)

    insights.append({
        "icon": "🏙️",
        "title": f"{loudest} is the loudest city",
        "text": (f"{loudest} averages {loud_db} dB — "
                 f"{'exceeding WHO caution level of 65 dB.' if loud_db > 65 else 'within moderate range.'}"),
        "severity": "error" if loud_db > 75 else "warning" if loud_db > 65 else "info",
    })

    insights.append({
        "icon": "🌿",
        "title": f"{quietest} is the quietest city",
        "text": f"{quietest} averages {quiet_c_db} dB — the most acoustically comfortable city in the dataset.",
        "severity": "success",
    })

    # ── Weekend vs weekday ─────────────────────────────────────────────────
    wknd_avg = df[df["is_weekend"] == 1]["Noise_Level_dB"].mean()
    wkdy_avg = df[df["is_weekend"] == 0]["Noise_Level_dB"].mean()
    diff     = round(abs(wknd_avg - wkdy_avg), 1)
    louder   = "Weekdays" if wkdy_avg > wknd_avg else "Weekends"

    insights.append({
        "icon": "📅",
        "title": f"{louder} are {diff} dB louder",
        "text": (f"Weekday avg: {wkdy_avg:.1f} dB vs Weekend avg: {wknd_avg:.1f} dB. "
                 f"{'Traffic-driven weekday activity dominates noise levels.' if louder == 'Weekdays' else 'Weekend leisure activity drives higher noise.'}"),
        "severity": "info",
    })

    # ── Rush hour impact ───────────────────────────────────────────────────
    rush_avg    = df[df["is_rush_hour"] == 1]["Noise_Level_dB"].mean()
    offpeak_avg = df[df["is_rush_hour"] == 0]["Noise_Level_dB"].mean()
    rush_delta  = round(rush_avg - offpeak_avg, 1)

    insights.append({
        "icon": "🚗",
        "title": f"Rush hour adds {rush_delta} dB",
        "text": (f"Rush hours (7–9 AM, 4–7 PM) average {rush_avg:.1f} dB "
                 f"vs {offpeak_avg:.1f} dB off-peak — a {rush_delta} dB increase."),
        "severity": "warning" if rush_delta > 5 else "info",
    })

    # ── Traffic correlation ────────────────────────────────────────────────
    corr = round(df["Noise_Level_dB"].corr(df["Traffic_Count"]), 2)
    insights.append({
        "icon": "📈",
        "title": f"Traffic explains {int(corr**2 * 100)}% of noise variance",
        "text": (f"Pearson correlation between traffic and noise: r = {corr}. "
                 f"{'Strong linear relationship — traffic is the dominant noise driver.' if corr > 0.7 else 'Moderate relationship — other factors also contribute significantly.'}"),
        "severity": "info",
    })

    # ── WHO exceedance ─────────────────────────────────────────────────────
    exc = detect_exceedances(df, threshold=65.0)
    insights.append({
        "icon": "⚠️",
        "title": f"{exc['pct']}% of readings exceed 65 dB (WHO caution)",
        "text": (f"{exc['exceeded']:,} out of {exc['total']:,} hourly readings exceed the WHO "
                 f"65 dB caution threshold. "
                 + (f"Worst hour: {exc['worst_hour']:02d}:00." if exc['worst_hour'] is not None else "")),
        "severity": "error" if exc["pct"] > 50 else "warning",
    })

    # ── Seasonal insight ───────────────────────────────────────────────────
    if "month" in df.columns:
        monthly = df.groupby("month")["Noise_Level_dB"].mean()
        noisiest_m = int(monthly.idxmax())
        quietest_m = int(monthly.idxmin())
        insights.append({
            "icon": "🗓️",
            "title": f"{MONTH_NAMES[noisiest_m]} is the noisiest month",
            "text": (f"Average noise peaks in {MONTH_NAMES[noisiest_m]} ({monthly.max():.1f} dB) "
                     f"and is lowest in {MONTH_NAMES[quietest_m]} ({monthly.min():.1f} dB). "
                     f"Seasonal variation: {monthly.max() - monthly.min():.1f} dB range."),
            "severity": "info",
        })

    return insights


def generate_statistical_insights(df: pd.DataFrame) -> list[dict]:
    """
    Distribution and inference-focused insights (skew, kurtosis, spread, ANOVA).
    Each dict matches show_insights: { icon, title, text, severity }.
    """
    insights: list[dict] = []
    n = df["Noise_Level_dB"].dropna()
    if len(n) < 10:
        return insights

    skew = float(n.skew())
    kurt = float(n.kurtosis())
    q1, q2, q3 = float(n.quantile(0.25)), float(n.quantile(0.5)), float(n.quantile(0.75))
    iqr = round(q3 - q1, 2)
    mean_v = float(n.mean())
    std_v = float(n.std())
    cv = round(100 * std_v / mean_v, 2) if mean_v else 0.0

    shape = (
        "right-skewed (more high-noise tail)" if skew > 0.5
        else "left-skewed" if skew < -0.5
        else "approximately symmetric"
    )
    tail = (
        "heavier than Gaussian tails" if kurt > 0.5
        else "lighter tails" if kurt < -0.5
        else "tail weight similar to normal"
    )
    insights.append({
        "icon": "📐",
        "title": f"Shape: skew {skew:.2f}, excess kurtosis {kurt:.2f}",
        "text": (
            f"The noise distribution is {shape}. Excess kurtosis {kurt:.2f} indicates {tail}. "
            f"Median {q2:.1f} dB vs mean {mean_v:.1f} dB."
        ),
        "severity": "info",
    })

    insights.append({
        "icon": "📏",
        "title": f"Spread: IQR = {iqr} dB (Q1–Q3)",
        "text": (
            f"Middle 50% of readings fall between {q1:.1f} and {q3:.1f} dB. "
            f"Coefficient of variation CV = {cv}% (relative dispersion around the mean)."
        ),
        "severity": "info",
    })

    r_pearson = float(df["Noise_Level_dB"].corr(df["Traffic_Count"]))
    rho = None
    try:
        from scipy.stats import spearmanr
        rho, sp_p = spearmanr(df["Noise_Level_dB"], df["Traffic_Count"], nan_policy="omit")
        rho = float(rho)
    except Exception:
        sp_p = None

    if rho is not None:
        insights.append({
            "icon": "🔗",
            "title": "Traffic association (Pearson vs Spearman)",
            "text": (
                f"Pearson r = {r_pearson:.3f}; Spearman ρ = {rho:.3f}"
                + (f" (p = {sp_p:.2e})" if sp_p is not None and np.isfinite(sp_p) else "")
                + ". Similar values suggest a fairly monotonic traffic–noise relationship."
            ),
            "severity": "info",
        })
    else:
        insights.append({
            "icon": "🔗",
            "title": "Traffic linear correlation",
            "text": f"Pearson correlation between traffic and noise: r = {r_pearson:.3f}.",
            "severity": "info",
        })

    cities = sorted(df["City"].unique())
    if len(cities) >= 2:
        try:
            from scipy.stats import f_oneway
            groups = [df.loc[df["City"] == c, "Noise_Level_dB"].dropna().values for c in cities]
            groups = [g for g in groups if len(g) > 0]
            if len(groups) >= 2:
                f_stat, p_val = f_oneway(*groups)
                sig = p_val < 0.001
                insights.append({
                    "icon": "🔬",
                    "title": "Do city means differ? (one-way ANOVA)",
                    "text": (
                        f"F = {f_stat:.2f}, p = {p_val:.2e}. "
                        + (
                            "Strong evidence that mean noise differs across cities (α = 0.001)."
                            if sig
                            else "Use city-level summaries below; group means may still differ in post-hoc tests."
                        )
                    ),
                    "severity": "warning" if sig else "info",
                })
        except Exception:
            pass

    # Between-city variance vs within-city (eta-squared style descriptive)
    try:
        grand = float(df["Noise_Level_dB"].mean())
        ss_between = sum(
            len(g) * (float(g.mean()) - grand) ** 2
            for _, g in df.groupby("City")["Noise_Level_dB"]
        )
        ss_total = float(((df["Noise_Level_dB"] - grand) ** 2).sum())
        eta2 = round(ss_between / ss_total, 3) if ss_total > 0 else 0.0
        insights.append({
            "icon": "📊",
            "title": f"City explains ~{int(eta2 * 100)}% of noise variance (η²)",
            "text": (
                "Eta-squared approximates how much overall variance is between-city vs within-city. "
                f"η² = {eta2} (descriptive effect size for city factor)."
            ),
            "severity": "info",
        })
    except Exception:
        pass

    return insights


def generate_city_insights(df: pd.DataFrame, city: str) -> list[dict]:
    """Generates city-specific insights."""
    d = df[df["City"] == city]
    if d.empty:
        return []

    insights = []
    hourly = d.groupby("hour")["Noise_Level_dB"].mean()
    peak_h = int(hourly.idxmax())
    peak_db = round(hourly.max(), 1)

    insights.append({
        "icon": "🔔",
        "title": f"Peak noise at {peak_h:02d}:00 ({peak_db} dB)",
        "text": f"{city}'s noisiest hour is {peak_h:02d}:00 averaging {peak_db} dB.",
        "severity": "error" if peak_db > 75 else "warning" if peak_db > 65 else "info",
    })

    exc = detect_exceedances(d, threshold=70.0)
    insights.append({
        "icon": "📊",
        "title": f"{exc['pct']}% of hours exceed 70 dB",
        "text": f"{exc['exceeded']:,} of {exc['total']:,} hourly readings in {city} exceed 70 dB.",
        "severity": "error" if exc["pct"] > 40 else "warning" if exc["pct"] > 20 else "success",
    })

    return insights


def generate_realtime_insights(pred_db: float, city: str,
                                traffic: int, weather: dict) -> list[dict]:
    """
    Generates insights for a single real-time prediction.
    weather dict: { temperature, wind_speed, precipitation, humidity }
    """
    insights = []
    now = datetime.now()
    hour = now.hour

    # Noise level assessment
    if pred_db > 75:
        insights.append({
            "icon": "🚨",
            "title": "Dangerous noise level",
            "text": f"Predicted {pred_db:.1f} dB exceeds WHO 75 dB danger threshold. Prolonged exposure risks hearing damage.",
            "severity": "error",
        })
    elif pred_db > 65:
        insights.append({
            "icon": "⚠️",
            "title": "Elevated noise level",
            "text": f"Predicted {pred_db:.1f} dB exceeds WHO 65 dB caution level. Sleep disruption and stress risk.",
            "severity": "warning",
        })
    else:
        insights.append({
            "icon": "✅",
            "title": "Acceptable noise level",
            "text": f"Predicted {pred_db:.1f} dB is within WHO safe range for {city}.",
            "severity": "success",
        })

    # Traffic context
    if traffic > 1800:
        insights.append({
            "icon": "🚗",
            "title": "Heavy traffic detected",
            "text": f"Estimated {traffic:,} vehicles/hr — primary noise driver at this hour.",
            "severity": "warning",
        })

    # Weather effects
    if weather.get("precipitation", 0) > 0.5:
        insights.append({
            "icon": "🌧️",
            "title": "Rain affecting noise",
            "text": f"Precipitation of {weather['precipitation']} mm/hr adds ambient noise but may reduce traffic.",
            "severity": "info",
        })

    if weather.get("wind_speed", 0) > 30:
        insights.append({
            "icon": "💨",
            "title": "High wind masking noise",
            "text": f"Wind at {weather['wind_speed']} km/h may mask traffic noise but adds wind noise.",
            "severity": "info",
        })

    # Time context
    if 7 <= hour <= 9:
        insights.append({
            "icon": "🌅",
            "title": "Morning rush hour",
            "text": "Peak commute period — expect elevated noise for the next 1–2 hours.",
            "severity": "warning",
        })
    elif 16 <= hour <= 19:
        insights.append({
            "icon": "🌆",
            "title": "Evening rush hour",
            "text": "Evening commute peak — noise typically subsides after 19:00.",
            "severity": "warning",
        })
    elif 0 <= hour <= 5:
        insights.append({
            "icon": "🌙",
            "title": "Night-time quiet period",
            "text": "WHO recommends < 40 dB for undisturbed sleep. Any noise above 55 dB may cause sleep disruption.",
            "severity": "info",
        })

    return insights


# ══════════════════════════════════════════════════════════════════════════════
#  REAL-TIME CHART DATA BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_realtime_history(
    model,
    city: str,
    n_hours: int = 24,
    pm25_live: Optional[float] = None,
) -> pd.DataFrame:
    """
    Full calendar-day forecast: hours 00:00–23:00 (local), one weather snapshot,
    per-hour traffic from _simulate_traffic (same as live features).

    pm25_live: if set (e.g. Open-Meteo current PM2.5), use for every hour so this
    matches the Prediction page in Live mode. If None, estimate PM2.5 per hour
    from traffic + weather (closer to training pipeline).

    n_hours: kept for call-site compatibility; the chart always uses 24 hourly slots.
    """
    _ = n_hours
    import math
    from src.api_integration import (CITY_COORDS, _simulate_traffic,
                                      get_weather, estimate_pm25, _BASE_TRAFFIC,
                                      _CITY_BASE_NOISE)
    from src.model_training import FEATURES

    now = datetime.now()
    lat, lon = CITY_COORDS.get(city, (40.7128, -74.0060))
    weather  = get_weather(lat, lon)

    dow     = now.weekday()
    month   = now.month
    is_wknd = int(dow >= 5)

    rows = []
    for h in range(24):
        is_rush  = int(7 <= h <= 9 or 16 <= h <= 19)
        is_night = int(0 <= h <= 5)
        traffic  = _simulate_traffic(h, bool(is_wknd), city)
        if pm25_live is not None:
            pm25 = round(float(pm25_live), 1)
        else:
            pm25 = estimate_pm25(traffic, weather["humidity"], weather["precipitation"])

        hour_sin  = math.sin(2 * math.pi * h     / 24)
        hour_cos  = math.cos(2 * math.pi * h     / 24)
        month_sin = math.sin(2 * math.pi * month / 12)
        month_cos = math.cos(2 * math.pi * month / 12)

        t, hum = weather["temperature"], weather["humidity"]
        if t > 20 and hum > 40:
            heat_index = (
                -8.78469475556 + 1.61139411 * t + 2.33854883889 * hum
                - 0.14611605 * t * hum - 0.012308094 * t ** 2 - 0.0164248277778 * hum ** 2
                + 0.002211732 * t ** 2 * hum + 0.00072546 * t * hum ** 2
                - 0.000003582 * t ** 2 * hum ** 2
            )
        else:
            heat_index = float(t)

        traffic_ratio = traffic / _BASE_TRAFFIC.get(city, 1200)
        traffic_log   = math.log10(max(traffic, 1))
        city_base_noise = _CITY_BASE_NOISE.get(city, 65.0)

        feat_row = pd.DataFrame([[
            dow, is_wknd, is_rush, is_night,
            traffic, t, weather["wind_speed"],
            weather["precipitation"], hum, pm25,
            hour_sin, hour_cos, month_sin, month_cos,
            heat_index, traffic_ratio, traffic_log, city_base_noise,
        ]], columns=FEATURES)
        pred = model.predict(feat_row)[0]
        rows.append({
            "hour_label":   f"{h:02d}:00",
            "hour":         h,
            "Predicted dB": round(pred, 1),
            "Traffic":      traffic,
        })

    return pd.DataFrame(rows)
