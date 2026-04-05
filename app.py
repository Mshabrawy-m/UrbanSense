import os, math, logging
import streamlit as st
import joblib, pandas as pd, numpy as np
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime
from src.eda_visualizer import EDAVisualizer
from src.api_integration import (get_realtime_features, get_weather_display,
                                  get_air_quality_current,
                                  CITY_COORDS, _BASE_TRAFFIC, _CITY_BASE_NOISE,
                                  _simulate_traffic)
from src.model_training import FEATURES
from src.insights import (generate_dataset_insights, generate_statistical_insights,
                           generate_realtime_insights,
                           detect_city_peaks, detect_peak_hours, build_realtime_history)
from src.data_quality import check_processed_file, summarize_for_ui

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Wall-clock wait for parallel live API fetches (each city: weather + air quality, sequential)
LIVE_FEATURES_TIMEOUT_SEC = 90

st.set_page_config(page_title="UrbanSense AI", page_icon="🏙️",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""<style>
[data-testid="stMetricValue"]{font-size:1.4rem;font-weight:700}
[data-testid="stMetricLabel"]{font-size:0.8rem;color:#888}
</style>""", unsafe_allow_html=True)

# ── Loaders ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    _ = os.path.getmtime("models/noise_model.pkl")
    return joblib.load("models/noise_model.pkl"), joblib.load("models/model_meta.pkl")

@st.cache_resource
def get_chatbot():
    from src import chatbot as m
    import importlib; importlib.reload(m)
    return m.SmartNoiseChatbot()

@st.cache_data(ttl=3600)
def load_data():
    return pd.read_csv("data/processed_data.csv")

@st.cache_resource
def get_eda():
    return EDAVisualizer()

@st.cache_data(ttl=3600)
def cached_insights():
    return generate_dataset_insights(load_data())

@st.cache_data(ttl=3600)
def cached_statistical_insights():
    return generate_statistical_insights(load_data())

# ── Helpers ────────────────────────────────────────────────────────────────────
def noise_category(db):
    if db < 55: return "Quiet",    "🟢", "#2ecc71"
    if db < 65: return "Moderate", "🟡", "#f39c12"
    if db < 75: return "Loud",     "🟠", "#e67e22"
    return             "Very Loud","🔴", "#e74c3c"

def gauge_chart(value, title="Predicted Noise Level"):
    cat, icon, color = noise_category(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=value,
        number={"suffix": " dB", "font": {"size": 36}},
        title={"text": f"{title}<br><span style='font-size:0.85em'>{icon} {cat}</span>"},
        delta={"reference": 65, "increasing": {"color": "#e74c3c"},
               "decreasing": {"color": "#2ecc71"}},
        gauge={"axis": {"range": [40, 100]}, "bar": {"color": color, "thickness": 0.25},
               "bgcolor": "white",
               "steps": [{"range": [40,55], "color": "#d4edda"},
                          {"range": [55,65], "color": "#fff3cd"},
                          {"range": [65,75], "color": "#fde8d8"},
                          {"range": [75,100],"color": "#f8d7da"}],
               "threshold": {"line": {"color": "black", "width": 3},
                              "thickness": 0.75, "value": 70}}))
    fig.update_layout(height=300, margin=dict(t=80,b=0,l=20,r=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig

def predict_row(model, *args, city="New York", forecast_hour=None, forecast_month=None):
    """forecast_hour / forecast_month: use for hour-by-hour charts (defaults: now)."""
    if len(args) == 18:
        return model.predict(pd.DataFrame([list(args)], columns=FEATURES))[0]
    (dow, is_wknd, is_rush, is_night,
     traffic, temp, wind, precip, humid, pm25) = args[:10]
    hour = datetime.now().hour if forecast_hour is None else int(forecast_hour)
    month = datetime.now().month if forecast_month is None else int(forecast_month)
    hs = math.sin(2*math.pi*hour/24);  hc = math.cos(2*math.pi*hour/24)
    ms = math.sin(2*math.pi*month/12); mc = math.cos(2*math.pi*month/12)
    hi = (-8.78469475556 + 1.61139411*temp + 2.33854883889*humid
          - 0.14611605*temp*humid - 0.012308094*temp**2
          - 0.0164248277778*humid**2 + 0.002211732*temp**2*humid
          + 0.00072546*temp*humid**2 - 0.000003582*temp**2*humid**2
         ) if temp > 20 and humid > 40 else float(temp)
    tr  = traffic / _BASE_TRAFFIC.get(city, 1200)
    tl  = math.log10(max(traffic, 1))
    cbn = _CITY_BASE_NOISE.get(city, 65.0)
    return model.predict(pd.DataFrame(
        [[dow,is_wknd,is_rush,is_night,traffic,temp,wind,precip,humid,pm25,
          hs,hc,ms,mc,hi,tr,tl,cbn]], columns=FEATURES))[0]

def show_insights(insights, ncols=3):
    fn = {"info": st.info, "warning": st.warning,
          "success": st.success, "error": st.error}
    cols = st.columns(ncols)
    for i, ins in enumerate(insights):
        with cols[i % ncols]:
            fn.get(ins["severity"], st.info)(
                f"{ins['icon']} **{ins['title']}**\n\n{ins['text']}")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/city-buildings.png", width=56)
    st.title("UrbanSense AI")
    st.caption("Urban Noise Intelligence Platform")
    st.divider()
    page = st.radio(
        "Navigate",
        ["Dashboard", "EDA", "Data & quality", "Prediction", "Chatbot", "Model Info"],
    )
    st.divider()
    with st.expander("WHO Noise Guidelines"):
        st.markdown("""
| Level | Category | Risk |
|-------|----------|------|
| < 55 dB | Quiet | Safe |
| 55-65 dB | Moderate | Mild |
| 65-75 dB | Loud | Caution |
| > 75 dB | Very Loud | Harmful |
        """)
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.title("🏙️ UrbanSense AI")
    st.markdown("Real-time noise prediction and urban acoustic intelligence across global cities.")
    df = load_data()
    model, _ = load_model()

    _, col_r = st.columns([4, 1])
    with col_r:
        if st.button("Refresh Now"):
            st.cache_data.clear()
            st.rerun()

    now_h = datetime.now().hour
    if 7 <= now_h <= 9 or 16 <= now_h <= 19:
        st.warning("Rush hour detected — elevated noise levels expected.")

    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records",    f"{len(df):,}")
    c2.metric("Cities Monitored", df["City"].nunique())
    c3.metric("Avg Noise",        f"{df['Noise_Level_dB'].mean():.1f} dB")
    c4.metric("Peak Noise",       f"{df['Noise_Level_dB'].max():.1f} dB")
    c5.metric("Noise-Traffic r",  f"{df['Noise_Level_dB'].corr(df['Traffic_Count']):.2f}")

    st.divider()
    st.subheader("Automatic Insights")
    show_insights(cached_insights(), ncols=3)

    st.divider()
    st.subheader("Live Noise Estimates — All Cities")
    with st.spinner("Fetching live data (weather + air quality per city)..."):
        city_feats = {}
        with ThreadPoolExecutor(max_workers=5) as ex:
            fut_map = {ex.submit(get_realtime_features, c): c for c in CITY_COORDS}
            done, not_done = wait(fut_map, timeout=LIVE_FEATURES_TIMEOUT_SEC)
            for fut in done:
                c = fut_map[fut]
                try:
                    city_feats[c] = fut.result()
                except Exception as e:
                    logger.warning("Fetch failed for %s: %s", c, e)
                    city_feats[c] = None
            for fut in not_done:
                c = fut_map[fut]
                logger.warning("Live fetch timed out for %s after %ss", c, LIVE_FEATURES_TIMEOUT_SEC)
                city_feats[c] = None

    live_rows = []
    gauge_cols = st.columns(len(CITY_COORDS))
    for i, city in enumerate(CITY_COORDS):
        feats = city_feats.get(city)
        with gauge_cols[i]:
            if feats is None:
                st.metric(city, "N/A", "Unavailable")
                continue
            pred = predict_row(model, *feats)
            cat, icon, _ = noise_category(pred)
            live_rows.append({"City": city, "Predicted dB": round(pred, 1),
                               "Category": f"{icon} {cat}",
                               "Traffic": feats[4], "Temp (C)": feats[5],
                               "Wind (km/h)": feats[6], "Humidity (%)": feats[8],
                               "PM2.5": feats[9]})
            st.metric(city, f"{pred:.1f} dB", f"{icon} {cat}")

    st.divider()
    col_l, col_r2 = st.columns([3, 2])
    with col_l:
        st.subheader("City Peak Analysis")
        st.dataframe(detect_city_peaks(df), use_container_width=True, hide_index=True)
    with col_r2:
        st.subheader("Historical Avg by City")
        city_avg = df.groupby("City")["Noise_Level_dB"].mean().sort_values()
        fig = px.bar(city_avg, orientation="h",
                     labels={"value": "Avg dB", "index": "City"},
                     color=city_avg.values, color_continuous_scale="RdYlGn_r")
        fig.update_layout(showlegend=False, height=280,
                          margin=dict(t=10,b=10,l=10,r=10),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── 24-Hour Forecast Chart ─────────────────────────────────────────────────
    st.divider()
    st.subheader("Today's 24-Hour Noise Forecast")
    rt_city = st.selectbox("Select City", list(CITY_COORDS.keys()), key="rt_city")
    with st.spinner("Computing forecast..."):
        lat, lon = CITY_COORDS[rt_city]
        aq = get_air_quality_current(lat, lon)
        pm25_lv = None
        if aq.get("pm2_5") is not None and not aq.get("_is_fallback", True):
            pm25_lv = round(float(aq["pm2_5"]), 1)
        rt_df = build_realtime_history(model, rt_city, n_hours=24, pm25_live=pm25_lv)

    def _zone(db):
        if db < 55: return "Quiet"
        if db < 65: return "Moderate"
        if db < 75: return "Loud"
        return "Very Loud"

    rt_df["Zone"] = rt_df["Predicted dB"].apply(_zone)
    zone_colors = {"Quiet": "#2ecc71", "Moderate": "#f1c40f",
                   "Loud": "#e67e22", "Very Loud": "#e74c3c"}

    fig_fc = go.Figure()
    for zone, color in zone_colors.items():
        mask = rt_df["Zone"] == zone
        if mask.any():
            fig_fc.add_trace(go.Bar(
                x=rt_df.loc[mask, "hour_label"],
                y=rt_df.loc[mask, "Predicted dB"],
                name=zone, marker_color=color,
                text=rt_df.loc[mask, "Predicted dB"].apply(lambda v: f"{v:.0f}"),
                textposition="outside", textfont_size=10,
            ))

    # Mark current hour with an annotation (no add_vline on string axis)
    current_label = f"{datetime.now().hour:02d}:00"
    if current_label in rt_df["hour_label"].values:
        fig_fc.add_annotation(
            x=current_label, y=96,
            text="Now", showarrow=True, arrowhead=2,
            arrowcolor="#555", font=dict(color="#555", size=11),
            ax=0, ay=-30,
        )

    # Threshold lines use numeric y only — safe for bar charts with string x
    fig_fc.add_shape(type="line", x0=0, x1=1, xref="paper",
                     y0=65, y1=65, yref="y",
                     line=dict(color="#f39c12", width=1.5, dash="dot"))
    fig_fc.add_annotation(x=1, xref="paper", y=65, yref="y",
                          text="65 dB", showarrow=False,
                          font=dict(color="#f39c12", size=11), xanchor="left")
    fig_fc.add_shape(type="line", x0=0, x1=1, xref="paper",
                     y0=75, y1=75, yref="y",
                     line=dict(color="#e74c3c", width=1.5, dash="dot"))
    fig_fc.add_annotation(x=1, xref="paper", y=75, yref="y",
                          text="75 dB", showarrow=False,
                          font=dict(color="#e74c3c", size=11), xanchor="left")

    fig_fc.update_layout(
        title=f"24-Hour Noise Forecast — {rt_city}",
        xaxis_title="Hour of Day", yaxis_title="Predicted Noise (dB)",
        barmode="stack", height=360,
        yaxis=dict(range=[40, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=60, b=30, l=10, r=80),
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    st.divider()
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.subheader("Noise Heatmap: Hour x Day")
        pivot = df.groupby(["day_of_week","hour"])["Noise_Level_dB"].mean().unstack()
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=list(range(24)),
            y=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
            colorscale="RdYlGn_r", colorbar=dict(title="dB"),
            hovertemplate="%{y} %{x}:00 -> %{z:.1f} dB<extra></extra>"))
        fig.update_layout(xaxis_title="Hour", yaxis_title="Day",
                          height=280, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    with col_h2:
        st.subheader("Noise Category Breakdown")
        def _cat(db):
            if db < 55: return "Quiet"
            if db < 65: return "Moderate"
            if db < 75: return "Loud"
            return "Very Loud"
        df2 = df.copy()
        df2["Category"] = df2["Noise_Level_dB"].apply(_cat)
        counts = df2["Category"].value_counts().reset_index()
        counts.columns = ["Category", "Count"]
        cmap = {"Quiet":"#2ecc71","Moderate":"#f1c40f",
                "Loud":"#e67e22","Very Loud":"#e74c3c"}
        fig = px.pie(counts, names="Category", values="Count",
                     color="Category", color_discrete_map=cmap, hole=0.4)
        fig.update_layout(height=280, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA":
    df_eda = load_data()
    st.title("Exploratory Data Analysis")
    st.markdown(
        "Distributions, correlations, and city comparisons on the processed hourly dataset. "
        "Charts use extra bottom margin so legends and titles do not overlap."
    )

    st.subheader("Statistical insights")
    st.caption("Distribution shape, spread (IQR, CV), correlation robustness, and between-city tests.")
    show_insights(cached_statistical_insights(), ncols=3)

    st.divider()
    st.subheader("Dataset insights")
    with st.expander("View narrative insights (peaks, cities, rush hour)", expanded=False):
        show_insights(cached_insights(), ncols=2)

    st.divider()
    st.subheader("Peak hours")
    peak_city = st.selectbox(
        "City",
        ["All"] + sorted(df_eda["City"].unique().tolist()),
        key="eda_peak_city",
        help="Average noise by hour; top 5 noisiest hours for the selection.",
    )
    peak_df = detect_peak_hours(
        df_eda,
        city=None if peak_city == "All" else peak_city,
        top_n=5,
    )
    fig_peak = px.bar(
        peak_df,
        x="hour",
        y="avg_db",
        error_y="std_db",
        title=f"Top 5 noisiest hours (mean ± std) — {peak_city}",
        labels={"hour": "Hour of day", "avg_db": "Avg noise (dB)"},
        color="avg_db",
        color_continuous_scale="RdYlGn_r",
    )
    ymax = float(peak_df["avg_db"].max()) if len(peak_df) else 55.0
    fig_peak.update_layout(
        coloraxis_showscale=False,
        xaxis=dict(tickmode="linear", dtick=1),
        height=420,
        margin=dict(l=56, r=28, t=56, b=48),
        title=dict(x=0, xanchor="left", font=dict(size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, max(ymax * 1.12, 55)]),
    )
    st.plotly_chart(fig_peak, use_container_width=True)

    st.divider()
    get_eda().display_eda_in_streamlit()


# ══════════════════════════════════════════════════════════════════════════════
#  DATA & QUALITY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Data & quality":
    st.title("Data sources & quality checks")
    st.markdown(
        "**Open-Meteo** provides real hourly weather (archive for training, forecast for live) "
        "and real **PM2.5** from the Air Quality API when the service returns a value. "
        "There is no free global API for hourly **vehicle counts** or **sound-level meters** at these coordinates, "
        "so traffic remains a **deterministic model** and the training **noise label** is a **physics-inspired estimate** "
        "from traffic + weather (the ML model learns that mapping)."
    )

    st.subheader("Field inventory")
    st.dataframe(
        pd.DataFrame(
            {
                "Field": [
                    "temperature, wind, precipitation, humidity",
                    "PM2.5 (pm25)",
                    "Traffic_Count",
                    "Noise_Level_dB (target)",
                ],
                "Source": [
                    "Real — Open-Meteo Weather API",
                    "Real — Open-Meteo Air Quality API (fallback: empirical estimate)",
                    "Modeled — time-of-day + city profile (reproducible)",
                    "Modeled — ISO-style level from traffic + weather + city baseline",
                ],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Checks on `data/processed_data.csv`")
    if not os.path.isfile("data/processed_data.csv"):
        st.error("Processed dataset not found. Run `python setup.py` from the project root.")
    else:
        try:
            _cols = pd.read_csv("data/processed_data.csv", nrows=0).columns.tolist()
            if "pm25_source" not in _cols:
                st.info(
                    "This file was built before **real PM2.5** was merged. "
                    "Run **`python setup.py`** to rebuild `data.csv` / `processed_data.csv` with Open-Meteo Air Quality."
                )
        except Exception:
            pass
        rep = check_processed_file("data/processed_data.csv")
        st.markdown(summarize_for_ui(rep))
        if rep.get("errors"):
            for e in rep["errors"]:
                st.error(e)
        for w in rep.get("warnings") or []:
            st.warning(w)
        pm_share = (rep.get("stats") or {}).get("pm25_open_meteo_share")
        if pm_share is not None and pm_share < 0.5:
            st.info(
                "Less than half of rows show `pm25_source=open_meteo_aq` in the file. "
                "Rebuild with `python setup.py` after updating the pipeline so PM2.5 is merged from the Air Quality API."
            )
        with st.expander("Full report (JSON)"):
            st.json(rep)

    st.divider()
    st.caption(
        "References: [Open-Meteo Weather](https://open-meteo.com/), "
        "[Open-Meteo Air Quality](https://open-meteo.com/en/docs/air-quality-api)."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Prediction":
    st.title("Real-Time Noise Prediction")
    st.markdown("Predict urban noise levels using live weather data and traffic estimates.")
    model, meta = load_model()
    col_cfg, col_result = st.columns(2)

    with col_cfg:
        st.subheader("Configuration")
        city = st.selectbox("City", list(CITY_COORDS.keys()))
        mode = st.radio("Input Mode", ["Live API Data", "Manual Override"], horizontal=True)
        if mode == "Live API Data":
            with st.spinner("Fetching live weather from Open-Meteo..."):
                feats = get_realtime_features(city)
                wd    = get_weather_display(city)
            dow, is_wknd, is_rush, is_night = feats[0], feats[1], feats[2], feats[3]
            traffic, temp, wind, precip, humid, pm25 = (feats[4], feats[5], feats[6],
                                                         feats[7], feats[8], feats[9])
            hour = datetime.now().hour
            st.success("Live data: Open-Meteo weather + air quality (PM2.5 when available)")
            if wd.get("_is_fallback"):
                st.warning("Weather API unavailable — showing fallback defaults.")
            st.caption(wd.get("pm25_source", ""))
            r1, r2, r3 = st.columns(3)
            r1.metric("Temperature", f"{temp}C")
            r2.metric("Wind Speed",  f"{wind} km/h")
            r3.metric("Humidity",    f"{humid}%")
            r4, r5, r6 = st.columns(3)
            r4.metric("Precipitation", f"{precip} mm/hr",
                      delta="Raining" if wd["is_raining"] else "Dry", delta_color="off")
            r5.metric("PM2.5", f"{pm25} μg/m³")
            r6.metric("Traffic (modeled)", f"{traffic} veh/hr",
                      delta="Rush Hour" if is_rush else "Off-Peak", delta_color="off")
        else:
            feats = None
            now = datetime.now()
            hour     = st.slider("Hour of Day", 0, 23, now.hour)
            dow      = st.slider("Day of Week (0=Mon)", 0, 6, now.weekday())
            is_wknd  = int(dow >= 5)
            is_rush  = int(7 <= hour <= 9 or 16 <= hour <= 19)
            is_night = int(0 <= hour <= 5)
            traffic  = st.slider("Traffic Count (vehicles/hr)", 50, 3000, 1000)
            temp     = st.slider("Temperature (C)", -10, 45, 20)
            wind     = st.slider("Wind Speed (km/h)", 0, 60, 12)
            precip   = st.slider("Precipitation (mm/hr)", 0.0, 20.0, 0.0, step=0.1)
            humid    = st.slider("Humidity (%)", 10, 100, 60)
            pm25     = st.slider("PM2.5 (ug/m3)", 5, 150, 15)

    with col_result:
        st.subheader("Prediction Result")
        pred = (predict_row(model, *feats) if feats is not None
                else predict_row(model, dow, is_wknd, is_rush, is_night,
                                 traffic, temp, wind, precip, humid, pm25, city=city))
        cat, icon, _ = noise_category(pred)
        st.plotly_chart(gauge_chart(pred), use_container_width=True)
        fn = {"info": st.info, "warning": st.warning,
              "success": st.success, "error": st.error}
        for ins in generate_realtime_insights(pred, city, traffic,
                    {"temperature": temp, "wind_speed": wind,
                     "precipitation": precip, "humidity": humid}):
            fn.get(ins["severity"], st.info)(
                f"{ins['icon']} **{ins['title']}** — {ins['text']}")

    st.divider()
    st.subheader(f"Hour-by-Hour Forecast — {city} (Today)")
    month_fc = datetime.now().month
    if mode == "Live API Data":
        forecast_df = build_realtime_history(model, city, n_hours=24, pm25_live=float(pm25))
        forecast_df = forecast_df.rename(columns={"hour_label": "Hour"})
        forecast_df["Traffic Est."] = forecast_df["Traffic"]
        forecast_df["Category"] = forecast_df["Predicted dB"].apply(
            lambda v: f"{noise_category(v)[1]} {noise_category(v)[0]}")
        forecast_df = forecast_df.drop(columns=["Traffic"], errors="ignore")
    else:
        forecast_rows = []
        for h in range(24):
            is_r = int(7 <= h <= 9 or 16 <= h <= 19)
            is_n = int(0 <= h <= 5)
            t_est = _simulate_traffic(h, bool(is_wknd), city)
            p = predict_row(
                model, dow, is_wknd, is_r, is_n,
                t_est, temp, wind, precip, humid, pm25, city=city,
                forecast_hour=h, forecast_month=month_fc,
            )
            cat_h, icon_h, _ = noise_category(p)
            forecast_rows.append({
                "Hour": f"{h:02d}:00", "Predicted dB": round(p, 1),
                "Category": f"{icon_h} {cat_h}", "Traffic Est.": t_est,
            })
        forecast_df = pd.DataFrame(forecast_rows)
    fig = px.bar(forecast_df, x="Hour", y="Predicted dB",
                 color="Predicted dB", color_continuous_scale="RdYlGn_r",
                 title=f"24-Hour Noise Forecast — {city}")
    fig.add_hline(y=65, line_dash="dash", line_color="orange", annotation_text="65 dB")
    fig.add_hline(y=75, line_dash="dash", line_color="red",    annotation_text="75 dB")
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    col_tbl, col_sens = st.columns(2)
    with col_tbl:
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    with col_sens:
        st.subheader("Sensitivity: Noise vs Traffic")
        sens_df = pd.DataFrame([
            {"Traffic": t, "Predicted dB": predict_row(
                model, dow, is_wknd, is_rush, is_night,
                t, temp, wind, precip, humid, pm25, city=city)}
            for t in range(100, 2600, 100)
        ])
        fig2 = px.line(sens_df, x="Traffic", y="Predicted dB",
                       labels={"Traffic": "Vehicles/hr"})
        fig2.add_hline(y=70, line_dash="dash", line_color="red",
                       annotation_text="70 dB threshold")
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Multi-City Batch Prediction (Current Conditions)")
    with st.spinner("Fetching live data for all cities..."):
        batch_feats = {}
        with ThreadPoolExecutor(max_workers=5) as ex:
            fut_map = {ex.submit(get_realtime_features, c): c for c in CITY_COORDS}
            done, not_done = wait(fut_map, timeout=LIVE_FEATURES_TIMEOUT_SEC)
            for fut in done:
                c = fut_map[fut]
                try:
                    batch_feats[c] = fut.result()
                except Exception as e:
                    logger.warning("Batch fetch failed for %s: %s", c, e)
                    batch_feats[c] = None
            for fut in not_done:
                c = fut_map[fut]
                logger.warning("Batch fetch timed out for %s after %ss", c, LIVE_FEATURES_TIMEOUT_SEC)
                batch_feats[c] = None
    batch_rows = []
    for c in CITY_COORDS:
        f = batch_feats.get(c)
        if f is None:
            batch_rows.append({"City": c, "Predicted dB": "N/A",
                                "Category": "Unavailable", "Traffic": "-",
                                "Temp (C)": "-", "Wind (km/h)": "-",
                                "Humidity (%)": "-", "PM2.5": "-"})
            continue
        p = predict_row(model, *f)
        cat_c, icon_c, _ = noise_category(p)
        batch_rows.append({"City": c, "Predicted dB": round(p, 1),
                            "Category": f"{icon_c} {cat_c}",
                            "Traffic": f[4], "Temp (C)": f[5],
                            "Wind (km/h)": f[6], "Humidity (%)": f[8], "PM2.5": f[9]})
    st.dataframe(pd.DataFrame(batch_rows).sort_values("Predicted dB", ascending=False),
                 use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Chatbot":
    st.title("Noise Analyzer Chatbot")
    st.markdown("Powered by **Groq AI (Llama 3.1)** — multi-turn conversation with full dataset context.")
    bot = get_chatbot()
    st.markdown("**Quick questions:**")
    chips = ["Which city is the loudest?", "Predict noise in Cairo now",
             "Compare New York vs Tokyo", "What are the health risks of loud noise?",
             "Show noise trends over months", "How accurate is the model?",
             "Tips to reduce urban noise"]
    chip_cols = st.columns(len(chips))
    for i, chip in enumerate(chips):
        if chip_cols[i].button(chip, key=f"chip_{i}", use_container_width=True):
            st.session_state.setdefault("messages", [])
            st.session_state["pending_chip"] = chip
    st.divider()
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": (
            "Hello! I am UrbanSense AI — your urban noise intelligence assistant.\n\n"
            "I can help you with:\n"
            "- Noise predictions for any city\n"
            "- Data insights from 21,840 records across 5 cities\n"
            "- City comparisons (New York, London, Cairo, Tokyo, Paris)\n"
            "- Trend analysis — hourly, daily, monthly patterns\n"
            "- Health impact of noise pollution\n"
            "- Noise reduction advice\n"
            "- Model performance details\n\nWhat would you like to know?"
        )}]
    dispatch = {
        "PREDICTION": bot.handle_prediction, "COMPARE":    bot.handle_compare,
        "ADVICE":     bot.handle_advice,     "MODEL_INFO": bot.handle_model_info,
        "TREND":      bot.handle_trend,      "HEALTH":     bot.handle_health,
        "GENERAL":    bot.handle_general,
    }
    if "pending_chip" in st.session_state:
        chip_prompt = st.session_state.pop("pending_chip")
        st.session_state["messages"].append({"role": "user", "content": chip_prompt})
        response = dispatch[bot.rule_based_intent(chip_prompt)](chip_prompt)
        st.session_state["messages"].append({"role": "assistant", "content": response})
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Ask about noise levels, predictions, health impacts, or city comparisons..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.spinner("Thinking..."):
            response = dispatch[bot.rule_based_intent(prompt)](prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        bot.history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Info":
    st.title("Model Performance and Architecture")
    model, meta = load_model()
    df = load_data()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Cities",        df["City"].nunique())
    c3.metric("Features",      len(FEATURES))
    c4.metric("Date Range",    f"{df['DateTime'].min()[:7]} to {df['DateTime'].max()[:7]}")
    st.divider()
    st.subheader("Model Comparison")
    metrics = meta.get("metrics", {})
    rows = [{"Model": k, "RMSE": round(v["rmse"], 4),
             "MAE": round(v["mae"], 4), "R2": round(v["r2"], 4)}
            for k, v in metrics.items()]
    st.dataframe(pd.DataFrame(rows).sort_values("RMSE"),
                 use_container_width=True, hide_index=True)
    st.success(f"Selected model: {meta.get('best_model_name', 'N/A')}")
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Feature Importances")
        if hasattr(model, "feature_importances_"):
            imp_df = pd.DataFrame({"Feature": FEATURES,
                                   "Importance": model.feature_importances_}
                                  ).sort_values("Importance", ascending=True)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Blues")
            fig.update_layout(showlegend=False, height=420, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Actual vs Predicted")
        X = df[FEATURES]; y = df["Noise_Level_dB"]
        idx = np.random.choice(len(df), size=min(800, len(df)), replace=False)
        preds = model.predict(X.iloc[idx])
        actuals = y.iloc[idx].values
        residuals = actuals - preds
        fig = px.scatter(x=actuals, y=preds, opacity=0.4,
                         labels={"x": "Actual (dB)", "y": "Predicted (dB)"},
                         title="Actual vs Predicted Noise",
                         color=np.abs(residuals), color_continuous_scale="RdYlGn_r")
        fig.add_shape(type="line", x0=actuals.min(), y0=actuals.min(),
                      x1=actuals.max(), y1=actuals.max(),
                      line=dict(color="black", dash="dash"))
        fig.update_layout(coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)
    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Residual Distribution")
        fig = px.histogram(x=residuals, nbins=50, title="Prediction Residuals",
                           labels={"x": "Residual (dB)", "y": "Count"},
                           color_discrete_sequence=["#3498db"])
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        st.subheader("Model Metrics Comparison")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="RMSE", x=[r["Model"] for r in rows],
                             y=[r["RMSE"] for r in rows], marker_color="#e74c3c"))
        fig.add_trace(go.Bar(name="R2",   x=[r["Model"] for r in rows],
                             y=[r["R2"]   for r in rows], marker_color="#2ecc71"))
        fig.update_layout(barmode="group", height=320, title="RMSE vs R2 by Model")
        st.plotly_chart(fig, use_container_width=True)
    st.divider()
    st.subheader("Features Used")
    feat_desc = {
        "day_of_week": "Day of week (0=Mon)", "is_weekend": "Weekend flag (0/1)",
        "is_rush_hour": "Rush hour flag (0/1)", "is_night": "Night flag 0-5 (0/1)",
        "Traffic_Count": "Estimated vehicles/hr",
        "temperature": "Real air temp (C) — Open-Meteo",
        "wind_speed": "Real wind speed (km/h) — Open-Meteo",
        "precipitation": "Real precipitation (mm/hr) — Open-Meteo",
        "humidity": "Real relative humidity (%) — Open-Meteo",
        "pm25": "PM2.5 estimated from humidity + precipitation",
        "hour_sin": "Cyclic sin encoding of hour",
        "hour_cos": "Cyclic cos encoding of hour",
        "month_sin": "Cyclic sin encoding of month",
        "month_cos": "Cyclic cos encoding of month",
        "heat_index": "Perceived temperature (Steadman 1979)",
        "traffic_ratio": "Traffic vs city baseline ratio",
        "traffic_log": "Log10 of traffic count",
        "city_base_noise": "City-specific base noise level",
    }
    fcols = st.columns(3)
    for i, feat in enumerate(FEATURES):
        fcols[i % 3].info(f"**{feat}**\n\n{feat_desc.get(feat, '')}")
