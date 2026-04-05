# 🏙️ UrbanSense AI

> Graduation Project — Urban Noise Intelligence System  
> Real-time noise level prediction using Machine Learning, live weather APIs, and an AI-powered chatbot.

---

## Overview

The Smart Urban Noise Analyzer is a full-stack data science application that predicts urban noise levels across 5 global cities using real historical weather data, realistic traffic modeling, and an ensemble ML pipeline. It features an interactive Streamlit dashboard, exploratory data analysis, live predictions, and a Groq-powered AI chatbot.

---

## Features

| Module | Description |
|--------|-------------|
| **Dashboard** | Live noise estimates for all cities, heatmaps, KPI metrics |
| **EDA** | 10+ interactive Plotly charts across 5 analysis tabs |
| **Prediction** | Real-time prediction with live Open-Meteo weather, 24-hr forecast, sensitivity analysis |
| **Chatbot** | Groq AI (Llama 3.1) with 7 intent types and multi-turn memory |
| **Model Info** | Model comparison, feature importances, residual analysis |

---

## Tech Stack

- **Frontend:** Streamlit, Plotly
- **ML:** scikit-learn (Random Forest, Gradient Boosting, Linear/Ridge Regression)
- **Real Data:** [Open-Meteo Archive API](https://open-meteo.com/) — free, no API key
- **AI Chatbot:** [Groq API](https://groq.com/) — Llama 3.1 8B Instant
- **Data:** 21,840 hourly records across 5 cities (Jan–Jun 2024)

---

## Cities Covered

| City | Lat | Lon |
|------|-----|-----|
| New York | 40.71 | -74.01 |
| London | 51.51 | -0.13 |
| Cairo | 30.04 | 31.24 |
| Tokyo | 35.68 | 139.65 |
| Paris | 48.86 | 2.35 |

---

## Project Structure

```text
├── app.py                  # Main Streamlit application
├── src/
│   ├── api_integration.py  # Real-time weather API + feature builder
│   ├── data_processing.py  # Dataset builder (real weather + noise model)
│   ├── eda_visualizer.py   # EDA charts and Streamlit display
│   ├── model_training.py   # ML training pipeline
│   ├── chatbot.py          # Groq AI chatbot logic
│   ├── insights.py         # Automatic insight generation
│   └── data_quality.py     # Data validation scripts
├── setup.py                # One-time setup script
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/smart-urban-noise-analyzer.git
cd smart-urban-noise-analyzer
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

### 4. Run setup (builds dataset + trains model)

```bash
python setup.py
```

This fetches ~6 months of real hourly weather data from Open-Meteo for all 5 cities and trains 4 ML models. Takes ~2–3 minutes.

### 5. Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Environment Variables (Required)

The Chatbot uses the Groq API for inference. To use this feature, you must configure a Groq API key:

```bash
# Windows
set GROQ_API_KEY=your_key_here

# macOS / Linux
export GROQ_API_KEY=your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com).

---

## Data Pipeline

```
Open-Meteo Archive API
  └── Real hourly weather (temperature, wind, precipitation, humidity)
        └── data_processing.py
              ├── Traffic simulation (time-of-day + city density model)
              ├── PM2.5 estimation (humidity + precipitation correlation)
              ├── Noise model (ISO 9613-inspired empirical formula)
              └── processed_data.csv (21,840 rows × 15 columns)
                    └── model_training.py
                          └── noise_model.pkl (best model by RMSE)
```

---

## ML Models

| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| Random Forest ✅ | 1.95 | 1.37 | 0.89 |
| Gradient Boosting | 2.29 | 1.79 | 0.85 |
| Ridge Regression | 3.44 | 2.77 | 0.66 |
| Linear Regression | 3.44 | 2.77 | 0.66 |

**Features used (18):** `day_of_week`, `is_weekend`, `is_rush_hour`, `is_night`, `Traffic_Count`, `temperature`, `wind_speed`, `precipitation`, `humidity`, `pm25`, `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `heat_index`, `traffic_ratio`, `traffic_log`, `city_base_noise`

---

## Chatbot Capabilities

The AI chatbot supports 7 intent types:

| Intent | Example Question |
|--------|-----------------|
| Prediction | "Predict noise in Cairo right now" |
| Compare | "Compare New York vs Tokyo noise" |
| Trend | "What are the monthly noise patterns?" |
| Health | "What are the health risks of loud noise?" |
| Advice | "How can I reduce urban noise?" |
| Model Info | "How accurate is the model?" |
| General | "Which city is the loudest?" |

---

## WHO Noise Guidelines Reference

| Level | Category | Health Risk |
|-------|----------|-------------|
| < 55 dB | Quiet | Safe |
| 55–65 dB | Moderate | Mild annoyance |
| 65–75 dB | Loud | Sleep disruption, stress |
| > 75 dB | Very Loud | Hearing damage risk |

---

## Screenshots

> Add screenshots of your running app here.

---

## Authors

> Add your name, university, and supervisor here.

---

## License

This project is for academic purposes.
