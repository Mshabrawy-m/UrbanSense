# 🏙️ UrbanSense AI
### Smart Urban Noise Intelligence Platform

> **Graduation Project** — Faculty of Computer Science  
> Real-time urban noise level prediction powered by Machine Learning, live weather APIs, and a Groq AI chatbot.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org)
[![Groq](https://img.shields.io/badge/AI-Groq%20Llama%203.1-purple)](https://groq.com)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)](LICENSE)

---

## 📖 Project Description

**UrbanSense AI** is a full-stack data science graduation project that addresses the growing problem of urban noise pollution across major global cities. The system combines real historical weather data fetched from the Open-Meteo API, physics-inspired noise modeling, and a trained Machine Learning pipeline to predict hourly noise levels in decibels (dB) — and provides real-time predictions reflecting current atmospheric conditions.

### Problem Statement

Urban noise pollution is a major public health concern. The WHO estimates that over 100 million people in Europe alone are exposed to harmful traffic noise levels (>55 dB). However, real-time global noise sensor grids do not exist as open data. This project bridges that gap by:

1. **Collecting** real historical weather data (temperature, wind, humidity, precipitation) from the Open-Meteo free API for 5 major cities over 6 months
2. **Modeling** traffic density deterministically from time-of-day and city profiles
3. **Synthesizing** a physics-inspired noise label using an ISO 9613-style acoustic model as the training target
4. **Training** an ML ensemble to learn the traffic–weather–noise mapping
5. **Deploying** a Streamlit dashboard with live API data, real-time predictions, AI chatbot, and deep EDA

### What Makes It Different

- **No fake or random training data** — weather is real (Open-Meteo Archive), traffic is deterministic and reproducible, noise labels follow acoustic physics
- **Live predictions** — every dashboard load fetches real current weather from Open-Meteo and runs the trained model
- **AI chatbot** — a Groq-powered Llama 3.1 assistant with 7 intent types, multi-turn memory, and full dataset context
- **18 engineered features** — including cyclic time encodings, heat index, traffic ratio, log-transformed traffic, and city base noise

---

## 🖥️ Application Pages

| Page | Description |
|------|-------------|
| **🏙️ Dashboard** | Live multi-city noise gauges, 24-hour forecast bar chart, heatmap, noise category breakdown, automatic insights |
| **📊 EDA** | 5-tab interactive analysis — Time Patterns, Correlations, City Comparison, Distribution, Statistical Summary |
| **🎯 Prediction** | City + mode selector, live weather display, gauge chart, 24-hour forecast, traffic sensitivity, multi-city batch |
| **🤖 Chatbot** | Groq Llama 3.1 AI with intent routing, quick-chip buttons, multi-turn conversation memory |
| **📈 Model Info** | Side-by-side model comparison, feature importances, actual vs predicted scatter, residual histogram |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI / Frontend** | Streamlit 1.32+, Plotly 5.18+ |
| **ML Pipeline** | scikit-learn — Random Forest, Gradient Boosting, Ridge, Linear Regression |
| **Real Weather** | [Open-Meteo Archive + Forecast API](https://open-meteo.com/) — free, no API key required |
| **Real PM2.5** | [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api) — CAMS reanalysis |
| **AI Chatbot** | [Groq API](https://groq.com/) — Llama 3.1 8B Instant (fast inference) |
| **Data** | 21,840 hourly records · 5 cities · Jan–Jun 2024 |
| **Language** | Python 3.10+ |

---

## 🌍 Cities Covered

| City | Latitude | Longitude | Base Noise |
|------|----------|-----------|------------|
| 🗽 New York | 40.71 | -74.01 | 68.0 dB |
| 🎡 London | 51.51 | -0.13 | 64.0 dB |
| 🌄 Cairo | 30.04 | 31.24 | 72.0 dB |
| 🗼 Tokyo | 35.68 | 139.65 | 62.0 dB |
| 🗺️ Paris | 48.86 | 2.35 | 65.0 dB |

---

## 📁 Project Structure

```text
UrbanSense/
├── app.py                   # Main Streamlit application (5 pages, 689 lines)
├── setup.py                 # One-time setup: fetch data + train model
├── requirements.txt         # Python dependencies
│
├── src/
│   ├── api_integration.py   # Real-time Open-Meteo weather + PM2.5 fetch, 18-feature builder
│   ├── data_processing.py   # Dataset builder — real weather, deterministic traffic, noise model
│   ├── model_training.py    # ML training pipeline — 4 models, cross-val, best saved to pkl
│   ├── eda_visualizer.py    # 10+ Plotly charts across 5 EDA tabs (EDAVisualizer class)
│   ├── insights.py          # Auto insight engine + realtime chart builder
│   ├── chatbot.py           # Groq AI chatbot — intent routing, multi-turn memory
│   └── data_quality.py      # Data validation, schema checks, quality reports
│
├── data/
│   └── (generated by setup.py — .gitignored)
│
├── models/
│   └── (generated by setup.py — .gitignored)
│
└── tests/
    └── quality_test.py      # Full test suite — 9 sections, 30+ assertions
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Mshabrawy-m/UrbanSense.git
cd UrbanSense
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Groq API key (for chatbot)

```bash
# Windows
set GROQ_API_KEY=your_key_here

# macOS / Linux
export GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

### 5. Run setup — fetches data and trains the model

```bash
python setup.py
```

> This fetches 6 months of real hourly weather from Open-Meteo for all 5 cities and trains 4 ML models. Takes **~2–4 minutes** depending on internet speed.

### 6. Launch the dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔄 Data Pipeline

```
Open-Meteo Archive API (free, no key)
  └── Real hourly weather: temperature, wind, precipitation, humidity
        └── src/data_processing.py
              ├── Traffic simulation  — deterministic, seeded by (hour, dow, city)
              ├── PM2.5 source       — Open-Meteo Air Quality API (fallback: estimate)
              ├── Noise label        — ISO 9613-inspired formula: city_base + log(traffic) ± weather
              └── processed_data.csv — 21,840 rows × 22 columns
                    └── src/model_training.py
                          └── noise_model.pkl  — best model selected by RMSE
                          └── model_meta.pkl   — metrics for all 4 models
```

---

## 🤖 ML Models Comparison

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| **Random Forest ✅** | **1.95** | **1.37** | **0.89** |
| Gradient Boosting | 2.29 | 1.79 | 0.85 |
| Ridge Regression | 3.44 | 2.77 | 0.66 |
| Linear Regression | 3.44 | 2.77 | 0.66 |

**18 Engineered Features:**

| Category | Features |
|----------|----------|
| Time | `day_of_week`, `is_weekend`, `is_rush_hour`, `is_night` |
| Cyclic encoding | `hour_sin`, `hour_cos`, `month_sin`, `month_cos` |
| Weather (real) | `temperature`, `wind_speed`, `precipitation`, `humidity` |
| Air Quality | `pm25` (Open-Meteo AQ or estimated) |
| Derived | `heat_index`, `traffic_ratio`, `traffic_log`, `city_base_noise` |
| Traffic | `Traffic_Count` |

---

## 💬 AI Chatbot Capabilities

Powered by **Groq Llama 3.1 8B Instant** with 7 intent types and multi-turn conversation memory:

| Intent | Trigger Keywords | Example |
|--------|-----------------|---------|
| **Prediction** | predict, forecast, estimate | *"Predict noise in Cairo right now"* |
| **Compare** | compare, vs, versus | *"Compare New York vs Tokyo noise"* |
| **Trend** | trend, pattern, monthly | *"What are the monthly noise patterns?"* |
| **Health** | health, risk, danger | *"What are the health risks of loud noise?"* |
| **Advice** | tips, reduce, mitigate | *"How can I reduce urban noise?"* |
| **Model Info** | model, accuracy, RMSE | *"How accurate is the model?"* |
| **General** | anything else | *"Which city is the loudest?"* |

---

## 🏥 WHO Noise Guidelines Reference

| Level | Category | Health Risk |
|-------|----------|-------------|
| < 55 dB | 🟢 Quiet | Safe for all activities |
| 55–65 dB | 🟡 Moderate | Mild annoyance, reduced concentration |
| 65–75 dB | 🟠 Loud | Sleep disruption, stress, cardiovascular risk |
| > 75 dB | 🔴 Very Loud | Hearing damage risk with prolonged exposure |

---

## ✅ System Health Status

All modules verified — last audit: April 2026

| Component | Status | Details |
|-----------|--------|---------|
| `src/api_integration.py` | ✅ Clean | 18-feature vector, fallback logic for API failures |
| `src/data_processing.py` | ✅ Clean | Schema validation, real weather + PM2.5 merge |
| `src/model_training.py` | ✅ Clean | FEATURES list matches api_integration exactly |
| `src/insights.py` | ✅ Clean | Modular, no Streamlit dependencies |
| `src/eda_visualizer.py` | ✅ Clean | 5 tabs, 10+ Plotly charts |
| `src/chatbot.py` | ✅ Clean | API key via env var, no hardcoded secrets |
| `src/data_quality.py` | ✅ Clean | Data validation utility |
| `tests/quality_test.py` | ✅ Clean | 9 test sections, 30+ assertions |
| `.gitignore` | ✅ Clean | Ignores `*.pkl`, `*.csv`, `__pycache__`, `.env` |
| `requirements.txt` | ✅ Clean | All 9 dependencies pinned |

---

## 📸 Screenshots

> _Add screenshots of the running Streamlit dashboard here._

---

## 👥 Authors

> Add your name, university, and supervisor here.

---

## 📄 License

This project was developed for academic / graduation purposes.
