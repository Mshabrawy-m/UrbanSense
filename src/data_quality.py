"""
data_quality.py — Validate raw and processed datasets (ranges, duplicates, PM2.5 source).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

REQUIRED_PROCESSED = [
    "DateTime", "City", "Noise_Level_dB", "Traffic_Count",
    "temperature", "wind_speed", "precipitation", "humidity", "pm25",
]


def _bounds_check(series: pd.Series, low: float, high: float, name: str) -> list[str]:
    out: list[str] = []
    if series.empty:
        return out
    bad = ((series < low) | (series > high)) & series.notna()
    n = int(bad.sum())
    if n:
        out.append(f"{name}: {n} values outside [{low}, {high}]")
    return out


def check_processed_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    """Run schema + quality checks on processed (or raw) hourly table."""
    report: dict[str, Any] = {
        "ok": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    miss = [c for c in REQUIRED_PROCESSED if c not in df.columns]
    if miss:
        report["ok"] = False
        report["errors"].append(f"Missing columns: {miss}")
        return report

    n = len(df)
    report["stats"]["rows"] = n
    report["stats"]["cities"] = int(df["City"].nunique()) if n else 0
    report["stats"]["date_min"] = str(df["DateTime"].min()) if n else None
    report["stats"]["date_max"] = str(df["DateTime"].max()) if n else None

    dup = df.duplicated(subset=["City", "DateTime"], keep=False).sum()
    if dup:
        report["ok"] = False
        report["errors"].append(f"Duplicate City+DateTime rows: {int(dup)}")

    null_pct = df[REQUIRED_PROCESSED].isna().mean().round(4).to_dict()
    report["stats"]["null_fraction_by_col"] = null_pct
    high_null = [k for k, v in null_pct.items() if v > 0.01]
    if high_null:
        report["warnings"].append(f"Columns with >1% missing: {high_null}")

    report["warnings"] += _bounds_check(df["temperature"], -50, 55, "temperature")
    report["warnings"] += _bounds_check(df["humidity"], 0, 100, "humidity")
    report["warnings"] += _bounds_check(df["wind_speed"], 0, 200, "wind_speed")
    report["warnings"] += _bounds_check(df["precipitation"], 0, 500, "precipitation")
    report["warnings"] += _bounds_check(df["pm25"], 0, 600, "pm25")
    report["warnings"] += _bounds_check(df["Noise_Level_dB"], 35, 110, "Noise_Level_dB")
    report["warnings"] += _bounds_check(df["Traffic_Count"], 1, 100000, "Traffic_Count")

    if "pm25_source" in df.columns:
        vc = df["pm25_source"].value_counts(dropna=False)
        report["stats"]["pm25_source_counts"] = vc.to_dict()
        api_share = float((df["pm25_source"] == "open_meteo_aq").mean()) if n else 0.0
        report["stats"]["pm25_open_meteo_share"] = round(api_share, 4)

    r = df["Noise_Level_dB"].corr(df["Traffic_Count"])
    report["stats"]["corr_noise_traffic"] = round(float(r), 4) if pd.notna(r) else None

    if report["errors"]:
        report["ok"] = False
    return report


def check_processed_file(path: str = "data/processed_data.csv") -> dict[str, Any]:
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return {"ok": False, "errors": [f"Cannot read {path}: {e}"], "warnings": [], "stats": {}}
    if "DateTime" in df.columns:
        df["DateTime"] = pd.to_datetime(df["DateTime"], errors="coerce")
    return check_processed_dataframe(df)


def summarize_for_ui(report: dict[str, Any]) -> str:
    lines = []
    lines.append("**Status:** " + ("PASS" if report.get("ok") else "FAIL"))
    if report.get("errors"):
        lines.append("**Errors:** " + "; ".join(report["errors"]))
    if report.get("warnings"):
        lines.append("**Warnings:** " + "; ".join(report["warnings"]))
    st = report.get("stats") or {}
    if st.get("rows") is not None:
        lines.append(f"**Rows:** {st['rows']:,} | **Cities:** {st.get('cities', '—')}")
    if st.get("pm25_open_meteo_share") is not None:
        lines.append(
            f"**PM2.5 from Open-Meteo (in file):** {100 * st['pm25_open_meteo_share']:.1f}% of rows"
        )
    return "\n\n".join(lines)
