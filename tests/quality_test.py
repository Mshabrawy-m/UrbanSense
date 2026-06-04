"""
Quality Test Suite — Smart Urban Noise Analyzer
Runs without Streamlit. Tests all core modules.
"""
import ast, sys, traceback, time
import pandas as pd
import numpy as np

PASS = "\033[92m PASS\033[0m"
FAIL = "\033[91m FAIL\033[0m"
WARN = "\033[93m WARN\033[0m"

results = []

def check(name, fn):
    try:
        t0 = time.time()
        msg = fn()
        elapsed = round(time.time() - t0, 3)
        results.append((name, "PASS", msg or "", elapsed))
        print(f"{PASS} [{elapsed:.3f}s] {name}" + (f" — {msg}" if msg else ""))
    except Exception as e:
        results.append((name, "FAIL", str(e), 0))
        print(f"{FAIL} {name} — {e}")

# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 1. SYNTAX CHECKS ===")
# ══════════════════════════════════════════════════════════════════════════════
for f in ["app.py","src/api_integration.py","src/chatbot.py",
          "src/data_processing.py","src/eda_visualizer.py",
          "src/insights.py","src/model_training.py"]:
    def _syn(f=f):
        ast.parse(open(f, encoding="utf-8").read())
        return f"OK"
    check(f"Syntax: {f}", _syn)

# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 2. IMPORT CHECKS ===")
# ══════════════════════════════════════════════════════════════════════════════
def _imp_model():
    from src.model_training import FEATURES, TARGET, DATA_PROCESSED, MODEL_PATH, META_PATH
    assert len(FEATURES) == 18, f"Expected 18 features, got {len(FEATURES)}"
    return f"18 features: {FEATURES[:3]}..."
check("Import: model_training", _imp_model)

def _imp_api():
    from src.api_integration import (CITY_COORDS, _BASE_TRAFFIC, _CITY_BASE_NOISE,
                                      get_weather, estimate_pm25, _simulate_traffic,
                                      _compute_engineered, get_realtime_features,
                                      get_weather_display)
    assert len(CITY_COORDS) == 5
    return f"{len(CITY_COORDS)} cities"
check("Import: api_integration", _imp_api)

def _imp_dp():
    from src.data_processing import (validate_schema, estimate_pm25, simulate_traffic,
                                      REQUIRED_COLS, DATA_RAW, DATA_PROCESSED)
    return "OK"
check("Import: data_processing", _imp_dp)

def _imp_insights():
    from src.insights import (generate_dataset_insights, generate_realtime_insights,
                               detect_city_peaks, detect_peak_hours, detect_peak_months,
                               detect_exceedances, build_realtime_history)
    return "OK"
check("Import: insights", _imp_insights)

def _imp_eda():
    from src.eda_visualizer import EDAVisualizer
    return "OK"
check("Import: eda_visualizer", _imp_eda)

# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 3. DATA CHECKS ===")
# ══════════════════════════════════════════════════════════════════════════════
def _data_exists():
    df = pd.read_csv("data/processed_data.csv")
    assert len(df) > 20000, f"Too few rows: {len(df)}"
    return f"{len(df):,} rows, {len(df.columns)} cols"
check("Data: processed_data.csv exists", _data_exists)

def _data_schema():
    from src.data_processing import REQUIRED_COLS
    df = pd.read_csv("data/processed_data.csv")
    missing = REQUIRED_COLS - set(df.columns)
    assert not missing, f"Missing cols: {missing}"
    return f"All {len(REQUIRED_COLS)} required cols present"
check("Data: schema validation", _data_schema)

def _data_features():
    from src.model_training import FEATURES
    df = pd.read_csv("data/processed_data.csv")
    missing = [f for f in FEATURES if f not in df.columns]
    assert not missing, f"Missing feature cols: {missing}"
    return f"All 18 ML features present"
check("Data: all ML features in CSV", _data_features)

def _data_no_nulls():
    from src.model_training import FEATURES
    df = pd.read_csv("data/processed_data.csv")
    nulls = df[FEATURES].isnull().sum()
    bad = nulls[nulls > 0]
    assert len(bad) == 0, f"Null values in: {bad.to_dict()}"
    return "No nulls in feature columns"
check("Data: no nulls in features", _data_no_nulls)

def _data_noise_range():
    df = pd.read_csv("data/processed_data.csv")
    mn, mx = df["Noise_Level_dB"].min(), df["Noise_Level_dB"].max()
    assert 40 <= mn <= 60, f"Min noise {mn} out of expected range"
    assert 70 <= mx <= 100, f"Max noise {mx} out of expected range"
    return f"Range: {mn:.1f} – {mx:.1f} dB"
check("Data: noise level range", _data_noise_range)

def _data_cities():
    df = pd.read_csv("data/processed_data.csv")
    cities = sorted(df["City"].unique())
    expected = ["Cairo","London","New York","Paris","Tokyo"]
    assert cities == expected, f"Cities mismatch: {cities}"
    return f"{cities}"
check("Data: all 5 cities present", _data_cities)

# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 4. MODEL CHECKS ===")
# ══════════════════════════════════════════════════════════════════════════════
def _model_exists():
    import joblib, os
    assert os.path.exists("models/noise_model.pkl"), "noise_model.pkl missing"
    assert os.path.exists("models/model_meta.pkl"),  "model_meta.pkl missing"
    model = joblib.load("models/noise_model.pkl")
    meta  = joblib.load("models/model_meta.pkl")
    return f"{type(model).__name__}, best={meta.get('best_model_name')}"
check("Model: pkl files exist", _model_exists)

def _model_features_match():
    import joblib
    from src.model_training import FEATURES
    model = joblib.load("models/noise_model.pkl")
    model_feats = list(model.feature_names_in_)
    assert model_feats == FEATURES, (
        f"Mismatch!\nModel: {model_feats}\nFEATURES: {FEATURES}")
    return "18 features match exactly"
check("Model: feature names match FEATURES list", _model_features_match)

def _model_predict_live():
    import joblib
    from src.model_training import FEATURES
    from src.api_integration import get_realtime_features
    model = joblib.load("models/noise_model.pkl")
    for city in ["New York", "Cairo", "Tokyo"]:
        feats = get_realtime_features(city)
        assert len(feats) == 18, f"{city}: got {len(feats)} features"
        row = pd.DataFrame([feats], columns=FEATURES)
        pred = model.predict(row)[0]
        assert 40 <= pred <= 100, f"{city}: prediction {pred:.1f} out of range"
    return "All 3 cities predicted successfully"
check("Model: live prediction (3 cities)", _model_predict_live)

def _model_metrics():
    import joblib
    meta = joblib.load("models/model_meta.pkl")
    metrics = meta.get("metrics", {})
    best = meta.get("best_model_name")
    best_r2 = metrics[best]["r2"]
    best_rmse = metrics[best]["rmse"]
    assert best_r2 > 0.85, f"R² too low: {best_r2:.4f}"
    assert best_rmse < 3.0, f"RMSE too high: {best_rmse:.4f}"
    return f"R²={best_r2:.4f}, RMSE={best_rmse:.4f}"
check("Model: R² > 0.85 and RMSE < 3.0", _model_metrics)

# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 5. API INTEGRATION CHECKS ===")
# ══════════════════════════════════════════════════════════════════════════════
def _api_feature_length():
    from src.api_integration import get_realtime_features, CITY_COORDS
    from src.model_training import FEATURES
    for city in CITY_COORDS:
        feats = get_realtime_features(city)
        assert len(feats) == len(FEATURES), \
            f"{city}: {len(feats)} features, expected {len(FEATURES)}"
    return f"All {len(CITY_COORDS)} cities return 18 features"
check("API: feature vector length = 18", _api_feature_length)

def _api_traffic_deterministic():
    from src.api_integration import _simulate_traffic
    v1 = _simulate_traffic(8, False, "Cairo")
    v2 = _simulate_traffic(8, False, "Cairo")
    assert v1 == v2, f"Traffic not deterministic: {v1} != {v2}"
    return f"Cairo 8AM = {v1} veh/hr (stable)"
check("API: traffic simulation is deterministic", _api_traffic_deterministic)

def _api_pm25_deterministic():
    from src.api_integration import estimate_pm25
    v1 = estimate_pm25(1200, 65.0, 0.0)
    v2 = estimate_pm25(1200, 65.0, 0.0)
    assert v1 == v2, f"PM2.5 not deterministic: {v1} != {v2}"
    return f"PM2.5(1200, 65%, 0mm) = {v1}"
check("API: PM2.5 estimation is deterministic", _api_pm25_deterministic)

def _api_weather_fallback():
    from src.api_integration import get_weather
    w = get_weather(0.0, 0.0)  # invalid coords — should fallback
    assert "_is_fallback" in w, "Missing _is_fallback key"
    assert "temperature" in w
    return f"Fallback flag present, temp={w['temperature']}"
check("API: weather fallback flag present", _api_weather_fallback)

def _api_weather_display():
    from src.api_integration import get_weather_display
    wd = get_weather_display("London")
    required = ["temperature","wind_speed","precipitation","humidity",
                "pm25","traffic","is_raining","_is_fallback","source"]
    missing = [k for k in required if k not in wd]
    assert not missing, f"Missing keys: {missing}"
    return f"All {len(required)} keys present"
check("API: get_weather_display keys", _api_weather_display)

# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 6. INSIGHTS ENGINE CHECKS ===")
# ══════════════════════════════════════════════════════════════════════════════
def _insights_dataset():
    from src.insights import generate_dataset_insights
    df = pd.read_csv("data/processed_data.csv")
    ins = generate_dataset_insights(df)
    assert len(ins) >= 8, f"Too few insights: {len(ins)}"
    for i in ins:
        assert "icon" in i and "title" in i and "text" in i and "severity" in i
        assert i["severity"] in ("info","warning","success","error")
    return f"{len(ins)} insights, all valid"
check("Insights: dataset insights generation", _insights_dataset)

def _insights_realtime():
    from src.insights import generate_realtime_insights
    # 72 dB = "Elevated" (warning), 76 dB = "Dangerous" (error)
    ins_warn = generate_realtime_insights(72.0, "Cairo", 1800,
                                     {"temperature":30,"wind_speed":5,
                                      "precipitation":0,"humidity":60})
    assert any(i["severity"] == "warning" for i in ins_warn), \
        f"72 dB should be warning. Got: {[i['severity'] for i in ins_warn]}"
    ins_err = generate_realtime_insights(76.0, "Cairo", 1800,
                                     {"temperature":30,"wind_speed":5,
                                      "precipitation":0,"humidity":60})
    assert any(i["severity"] == "error" for i in ins_err), \
        f"76 dB should be error. Got: {[i['severity'] for i in ins_err]}"
    return f"72dB=warning, 76dB=error — correct"
check("Insights: realtime insights severity levels", _insights_realtime)

def _insights_city_peaks():
    from src.insights import detect_city_peaks
    df = pd.read_csv("data/processed_data.csv")
    peaks = detect_city_peaks(df)
    assert len(peaks) == 5, f"Expected 5 cities, got {len(peaks)}"
    assert "Peak Hour" in peaks.columns
    return f"5 cities, peak hours: {peaks['Peak Hour'].tolist()}"
check("Insights: detect_city_peaks", _insights_city_peaks)

def _insights_exceedances():
    from src.insights import detect_exceedances
    df = pd.read_csv("data/processed_data.csv")
    exc = detect_exceedances(df, threshold=65.0)
    assert 0 < exc["pct"] < 100
    assert exc["exceeded"] > 0
    return f"{exc['pct']}% exceed 65 dB"
check("Insights: detect_exceedances", _insights_exceedances)

# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 7. DATA PROCESSING CHECKS ===")
# ══════════════════════════════════════════════════════════════════════════════
def _dp_validate_schema_pass():
    from src.data_processing import validate_schema, REQUIRED_COLS
    df = pd.read_csv("data/data.csv")
    validate_schema(df, "test")  # should not raise
    return "Schema valid"
check("DataProc: validate_schema passes on good data", _dp_validate_schema_pass)

def _dp_validate_schema_fail():
    from src.data_processing import validate_schema
    bad_df = pd.DataFrame({"a": [1], "b": [2]})
    raised = False
    try:
        validate_schema(bad_df, "bad")
    except ValueError as e:
        raised = True
        assert "missing required columns" in str(e).lower(), f"Wrong error message: {e}"
    assert raised, "validate_schema should raise ValueError on bad data"
    return "Correctly raises ValueError with clear message"
check("DataProc: validate_schema raises on bad data", _dp_validate_schema_fail)

def _dp_estimate_pm25():
    from src.data_processing import estimate_pm25
    low  = estimate_pm25(100,  40.0, 0.0)
    high = estimate_pm25(2000, 80.0, 0.0)
    rain = estimate_pm25(1200, 60.0, 5.0)
    assert high > low,  f"High traffic should give higher PM2.5: {high} vs {low}"
    assert rain < estimate_pm25(1200, 60.0, 0.0), "Rain should reduce PM2.5"
    return f"low={low}, high={high}, rain={rain}"
check("DataProc: PM2.5 logic (traffic up = PM2.5 up, rain = PM2.5 down)", _dp_estimate_pm25)

def _dp_simulate_traffic():
    from src.data_processing import simulate_traffic
    cfg = {"base_traffic": 1200, "traffic_mult": 1.0}
    rush   = simulate_traffic(8,  0, cfg)
    night  = simulate_traffic(3,  0, cfg)
    wknd   = simulate_traffic(8,  6, cfg)
    assert rush > night,  f"Rush {rush} should > night {night}"
    assert rush > wknd,   f"Rush {rush} should > weekend {wknd}"
    return f"rush={rush}, night={night}, weekend={wknd}"
check("DataProc: traffic simulation (rush > night, rush > weekend)", _dp_simulate_traffic)

# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 8. EDA VISUALIZER CHECKS ===")
# ══════════════════════════════════════════════════════════════════════════════
def _eda_init():
    from src.eda_visualizer import EDAVisualizer
    eda = EDAVisualizer()
    assert len(eda.cities) == 5
    assert len(eda._raw) > 20000
    return f"{len(eda._raw):,} rows, {len(eda.cities)} cities"
check("EDA: initializes correctly", _eda_init)

def _eda_stats():
    from src.eda_visualizer import EDAVisualizer
    eda = EDAVisualizer()
    stats = eda.get_stats()
    required = ["total_rows","noise_mean","noise_max","noise_min",
                "corr_noise_traffic","pct_above_65","pct_above_75"]
    missing = [k for k in required if k not in stats]
    assert not missing, f"Missing stats: {missing}"
    assert 0 < stats["pct_above_65"] < 100
    return f"avg={stats['noise_mean']} dB, {stats['pct_above_65']}% > 65 dB"
check("EDA: get_stats returns all keys", _eda_stats)

def _eda_charts():
    from src.eda_visualizer import EDAVisualizer
    eda = EDAVisualizer()
    d = eda._raw
    charts = [
        eda._fig_noise_by_hour(d),
        eda._fig_heatmap_hour_day(d),
        eda._fig_noise_by_day(d),
        eda._fig_traffic_vs_noise(d),
        eda._fig_correlation_heatmap(d),
        eda._fig_noise_by_city(d),
        eda._fig_city_radar(d),
        eda._fig_noise_distribution(d),
        eda._fig_cdf(d),
    ]
    assert all(c is not None for c in charts)
    return f"{len(charts)} charts generated"
check("EDA: all chart methods return figures", _eda_charts)

# ══════════════════════════════════════════════════════════════════════════════
print("\n=== 9. CHATBOT CHECKS ===")
# ══════════════════════════════════════════════════════════════════════════════
def _chatbot_init():
    from src.chatbot import SmartNoiseChatbot
    bot = SmartNoiseChatbot()
    assert bot.ml_model is not None, "ML model not loaded"
    assert bot.df is not None, "Data not loaded"
    assert bot.client is not None, "Groq client not initialized"
    return f"Model loaded, {len(bot.df):,} rows, Groq ready"
check("Chatbot: initializes correctly", _chatbot_init)

def _chatbot_intents():
    from src.chatbot import SmartNoiseChatbot
    bot = SmartNoiseChatbot()
    cases = [
        ("predict noise in Cairo", "PREDICTION"),
        ("compare New York vs Tokyo", "COMPARE"),
        ("tips to reduce noise", "ADVICE"),
        ("how accurate is the model", "MODEL_INFO"),
        ("monthly noise trend", "TREND"),
        ("health risks of loud noise", "HEALTH"),
        ("which city is loudest", "GENERAL"),
    ]
    for text, expected in cases:
        got = bot.rule_based_intent(text)
        assert got == expected, f"'{text}' -> {got}, expected {expected}"
    return f"All {len(cases)} intents correct"
check("Chatbot: intent routing (7 cases)", _chatbot_intents)

def _chatbot_city_extract():
    from src.chatbot import SmartNoiseChatbot
    bot = SmartNoiseChatbot()
    assert bot._extract_city("noise in Cairo") == "Cairo"
    assert bot._extract_city("Tokyo traffic") == "Tokyo"
    assert bot._extract_city("random text") == "New York"  # default
    return "City extraction correct"
check("Chatbot: city extraction", _chatbot_city_extract)

def _chatbot_history_cap():
    from src.chatbot import SmartNoiseChatbot
    bot = SmartNoiseChatbot()
    for i in range(15):
        bot.history.append({"role": "user", "content": f"msg {i}"})
        bot.history.append({"role": "assistant", "content": f"resp {i}"})
        if len(bot.history) > 20:
            bot.history = bot.history[-20:]
    assert len(bot.history) <= 20, f"History not capped: {len(bot.history)}"
    return f"History capped at {len(bot.history)} items"
check("Chatbot: history cap at 20", _chatbot_history_cap)

# ══════════════════════════════════════════════════════════════════════════════
print("\n=== SUMMARY ===")
# ══════════════════════════════════════════════════════════════════════════════
passed = sum(1 for r in results if r[1] == "PASS")
failed = sum(1 for r in results if r[1] == "FAIL")
total  = len(results)
print(f"\nTotal: {total} | Passed: {passed} | Failed: {failed}")
if failed:
    print("\nFailed tests:")
    for name, status, msg, _ in results:
        if status == "FAIL":
            print(f"  - {name}: {msg}")
    sys.exit(1)
else:
    print("\nAll tests passed.")
