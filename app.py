import os
import math
import logging
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from src.eda_visualizer import EDAVisualizer
from src.api_integration import get_realtime_features, get_weather_display, CITY_COORDS, _BASE_TRAFFIC
from src.model_training import FEATURES
from src.insights import (
    generate_dataset_insights, generate_realtime_insights,
    detect_city_peaks, detect_peak_hours, build_realtime_history,
)

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

st.set_page_config(page_title="UrbanSense AI", page_icon="🏙️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.8rem; color: #888; }
</style>
""", unsafe_allow_html=True)

# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    _mt = os.path.getmtime('models/noise_model.pkl')
    return joblib.load('models/noise_model.pkl'), joblib.load('models/model_meta.pkl')

@st.cache_resource
def get_chatbot():
    from src import chatbot as m
    import importlib
    importlib.reload(m)
    return m.SmartNoiseChatbot()

@st.cache_data(ttl=3600)
def load_data():
    return pd.read_csv('data/processed_data.csv')

@st.cache_resource
def get_eda():
    return EDAVisualizer()

@st.cache_data(ttl=3600)
def cached_insights():
    return generate_dataset_insights(load_data())

# ── Helpers ────────────────────────────────────────────────────────────────────
def noise_category(db):
    if db < 55:  return "Quiet",    "🟢", "#2ecc71"
    if db < 65:  return "Moderate", "🟡", "#f39c12"
    if db < 75:  return "Loud",     "🟠", "#e67e22"
    return             "Very Loud", "🔴", "#e74c3c"


def gauge_chart(value, title="Predicted Noise Level"):
    cat, icon, color = noise_category(value)
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=value,
        number={"suffix": " dB", "font": {"size": 36}},
        title={"text": f"{title}<br><span style='font-size:0.85em'>{icon} {cat}</span>"},
        delta={"reference": 65, "increasing": {"color": "#e74c3c"},
               "decreasing": {"color": "#2ecc71"}},
        gauge={
            "axis": {"range": [40, 100]},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "steps": [
                {"range": [40, 55], "color": "#d4edda"},
                {"range": [55, 65], "color": "#fff3cd"},
                {"range": [65, 75], "color": "#fde8d8"},
                {"range": [75, 100], "color": "#f8d7da"},
            ],
            "threshold": {"line": {"color": "black", "width": 3},
                          "thickness": 0.75, "value": 70},
        },
    ))
    fig.update_layout(height=300, margin=dict(t=80, b=0, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def predict_row(model, *args, city="New York"):
    """
    Two modes:
      predict_row(model, *feats)        -- 18 values from get_realtime_features
      predict_row(model, hour, dow, month, is_wknd, is_rush, is_night,
                  traffic, temp, wind, precip, humid, pm25, city=city)  -- 13 manual
    """
    if len(args) == 18:
        row = pd.DataFrame([list(args)], columns=FEATURES)
        return model.predict(row)[0]

    (hour, dow, month, is_wknd, is_rush, is_night,
     traffic, temp, wind, precip, humid, pm25) = args[:12]

    hs = math.sin(2 * math.pi * hour  / 24)
    hc = math.cos(2 * math.pi * hour  / 24)
    ms = math.sin(2 * math.pi * month / 12)
    mc = math.cos(2 * math.pi * month / 12)
    hi = (
        -8.78469475556 + 1.61139411*temp + 2.33854883889*humid
        - 0.14611605*temp*humid - 0.012308094*temp**2
        - 0.0164248277778*humid**2 + 0.002211732*temp**2*humid
        + 0.00072546*temp*humid**2 - 0.000003582*temp**2*humid**2
    ) if temp > 20 and humid > 40 else float(temp)
    tr = traffic / _BASE_TRAFFIC.get(city, 1000)
    row = pd.DataFrame([[hour, dow, month, is_wknd, is_rush, is_night,
                         traffic, temp, wind, precip, humid, pm25,
                         hs, hc, ms, mc, hi, tr]], columns=FEATURES)
    return model.predict(row)[0]


def show_insights(insights, ncols=3):
    fn = {"info": st.info, "warning": st.warning,
          "success": st.success, "error": st.error}
    cols = st.columns(ncols)
    for i, ins in enumerate(insights):
        with cols[i % ncols]:
            fn.get(ins["severity"], st.info)(
                f"{ins['icon']} **{ins['title']}**\n\n{ins['text']}"
            )


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/speaker.png", width=56)
    st.title("Noise Analyzer")
    st.caption("Urban Noise Intelligence System")
    st.divider()
    st.title("🏙️ UrbanSense AI")
    st.markdown("Real-time noise prediction and urban acoustic intelligence across global cities.")
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
    page = st.radio("Navigation", ["🏠 Dashboard", "📊 EDA", "🔮 Prediction", "💬 Chatbot", "🧠 Model Info"])


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.title("🔊 Smart Urban Noise Analyzer")
    st.markdown("Real-time noise prediction and urban acoustic intelligence across global cities.")

    df = load_data()
    model, _ = load_model()

    _, col_r = st.columns([4, 1])
    with col_r:
        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()

    now_h = datetime.now().hour
    if 7 <= now_h <= 9 or 16 <= now_h <= 19:
        st.warning("Rush hour detected — elevated noise levels expected across all cities.")

    st.divider()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records",    f"{len(df):,}")
    c2.metric("Cities Monitored", df['City'].nunique())
    c3.metric("Avg Noise (dB)",   f"{df['Noise_Level_dB'].mean():.1f}")
    c4.metric("Max Noise (dB)",   f"{df['Noise_Level_dB'].max():.1f}")
    c5.metric("Avg Traffic",      f"{df['Traffic_Count'].mean():.0f}")

    st.divider()
    st.subheader("Automatic Insights")
    show_insights(cached_insights(), ncols=3)

    st.divider()
    st.subheader("Live Noise Estimates — All Cities")
    with st.spinner("Fetching live data..."):
        city_feats = {}
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {ex.submit(get_realtime_features, c): c for c in CITY_COORDS}
            try:
                for fut in as_completed(futures, timeout=10):
                    c = futures[fut]
                    try:
                        city_feats[c] = fut.result()
                    except Exception as e:
                        logger.warning(f"Live fetch failed for {c}: {e}")
                        city_feats[c] = None
            except TimeoutError:
                logger.warning("Timeout fetching live data.")

        live_cols = st.columns(len(CITY_COORDS))
        for i, city in enumerate(CITY_COORDS):
            feats = city_feats.get(city)
            if feats is None: continue
            pred = predict_row(model, *feats)
            cat, icon, _ = noise_category(pred)
            live_cols[i].metric(city, f"{pred:.1f} dB", f"{icon} {cat}")

    st.subheader("Predicted Noise — Last 12 Hours")
    rt_city = st.selectbox("City", list(CITY_COORDS.keys()), key="rt_city")
    with st.spinner("Building rolling history..."):
        rt_df = build_realtime_history(model, rt_city, n_hours=12)

    # Color each bar by noise zone
    def _zone_color(db):
        if db < 55:  return '#2ecc71'
        if db < 65:  return '#f1c40f'
        if db < 75:  return '#e67e22'
        return '#e74c3c'

    rt_df['color'] = rt_df['Predicted dB'].apply(_zone_color)
    rt_df['Zone']  = rt_df['Predicted dB'].apply(
        lambda db: 'Quiet' if db < 55 else 'Moderate' if db < 65 else 'Loud' if db < 75 else 'Very Loud'
    )

    fig_rt = go.Figure()
    for zone, color in [('Quiet','#2ecc71'),('Moderate','#f1c40f'),('Loud','#e67e22'),('Very Loud','#e74c3c')]:
        mask = rt_df['Zone'] == zone
        if mask.any():
            fig_rt.add_trace(go.Bar(
                x=rt_df.loc[mask, 'hour_label'],
                y=rt_df.loc[mask, 'Predicted dB'],
                name=zone,
                marker_color=color,
                text=rt_df.loc[mask, 'Predicted dB'].round(1),
                textposition='outside', textfont_size=11,
            ))
    
    current_hour_label = f"{datetime.now().hour:02d}:00"
    if current_hour_label in rt_df["hour_label"].values:
        fig_rt.add_annotation(
            x=current_hour_label, y=95,
            text="Now", showarrow=True, arrowhead=2,
            arrowcolor="gray", font=dict(color="gray", size=11),
            ax=0, ay=-25,
        )
        fig_rt.add_annotation(
            x=current_hour_label, y=98,
            text="Now", showarrow=True, arrowhead=2,
            arrowcolor="white", font=dict(color="white", size=12),
            ax=0, ay=-30,
        )
    fig_rt.add_hline(y=65, line_dash='dot', line_color='#f39c12', line_width=1.5,
                     annotation_text='65 dB caution', annotation_position='right')
    fig_rt.add_hline(y=75, line_dash='dot', line_color='#e74c3c', line_width=1.5,
                     annotation_text='75 dB danger', annotation_position='right')
    fig_rt.update_layout(
        title=f"12-Hour Noise Estimate by Zone — {rt_city}",
        xaxis_title='Hour', yaxis_title='Predicted Noise (dB)',
        barmode='stack', height=320,
        yaxis=dict(range=[40, 95]),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=60, b=30, l=10, r=80),
    )
    st.plotly_chart(fig_rt, use_container_width=True)

    st.divider()
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.subheader("City Peak Analysis")
        st.dataframe(detect_city_peaks(df), use_container_width=True, hide_index=True)
    with col_r:
        st.subheader("Historical Avg by City")
        city_avg = df.groupby('City')['Noise_Level_dB'].mean().sort_values()
        fig = px.bar(city_avg, orientation='h',
                     labels={'value': 'Avg dB', 'index': 'City'},
                     color=city_avg.values, color_continuous_scale='RdYlGn_r')
        fig.update_layout(showlegend=False, height=280,
                          margin=dict(t=10, b=10, l=10, r=10),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Real-Time Rolling Noise — Last 12 Hours")
    rt_city = st.selectbox("City", list(CITY_COORDS.keys()), key="rt_city_2")
    with st.spinner("Building rolling history..."):
        rt_df = build_realtime_history(model, rt_city, n_hours=12)
    fig_rt = px.area(rt_df, x="hour_label", y="Predicted dB",
                     title=f"Rolling 12-Hour Noise Estimate — {rt_city}",
                     labels={"hour_label": "Hour", "Predicted dB": "Noise (dB)"},
                     color_discrete_sequence=["#3498db"])
    fig_rt.add_hline(y=65, line_dash="dash", line_color="orange", annotation_text="65 dB")
    fig_rt.add_hline(y=75, line_dash="dash", line_color="red",    annotation_text="75 dB")
    fig_rt.update_layout(height=300)
    st.plotly_chart(fig_rt, use_container_width=True)

    st.divider()
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.subheader("Noise Heatmap: Hour x Day")
        pivot = df.groupby(['day_of_week', 'hour'])['Noise_Level_dB'].mean().unstack()
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=list(range(24)),
            y=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            colorscale='RdYlGn_r', colorbar=dict(title='dB'),
            hovertemplate='%{y} %{x}:00 -> %{z:.1f} dB<extra></extra>'
        ))
        fig.update_layout(xaxis_title='Hour', yaxis_title='Day',
                          height=280, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
    with col_h2:
        st.subheader("Noise Category Breakdown")
        def _cat(db):
            if db < 55: return 'Quiet'
            if db < 65: return 'Moderate'
            if db < 75: return 'Loud'
            return 'Very Loud'
        df2 = df.copy()
        df2['Category'] = df2['Noise_Level_dB'].apply(_cat)
        counts = df2['Category'].value_counts().reset_index()
        counts.columns = ['Category', 'Count']
        cmap = {'Quiet': '#2ecc71', 'Moderate': '#f1c40f',
                'Loud': '#e67e22', 'Very Loud': '#e74c3c'}
        fig = px.pie(counts, names='Category', values='Count',
                     color='Category', color_discrete_map=cmap, hole=0.4)
        fig.update_layout(height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)



# ══════════════════════════════════════════════════════════════════════════════
#  EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    df_eda = load_data()
    st.subheader("Dataset Insights")
    with st.expander("View all auto-generated insights", expanded=True):
        show_insights(cached_insights(), ncols=2)

    st.subheader("Peak Hours by City")
    peak_city = st.selectbox("City", ["All"] + sorted(df_eda["City"].unique().tolist()),
                              key="eda_peak_city")
    peak_df = detect_peak_hours(df_eda,
                                city=None if peak_city == "All" else peak_city,
                                top_n=5)
    fig_peak = px.bar(peak_df, x="hour", y="avg_db", error_y="std_db",
                      title=f"Top 5 Noisiest Hours — {peak_city}",
                      labels={"hour": "Hour of Day", "avg_db": "Avg Noise (dB)"},
                      color="avg_db", color_continuous_scale="RdYlGn_r")
    fig_peak.update_layout(coloraxis_showscale=False, xaxis=dict(tickmode="linear"))
    st.plotly_chart(fig_peak, use_container_width=True)

    st.divider()
    get_eda().display_eda_in_streamlit()


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
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
            now = datetime.now()
            hour, dow, month = now.hour, now.weekday(), now.month
            is_wknd, is_rush, is_night = feats[1], feats[2], feats[3]
            traffic, temp, wind, precip, humid, pm25 = (
                feats[4], feats[5], feats[6], feats[7], feats[8], feats[9])
            st.success("Live data from Open-Meteo API")
            st.caption("Source: Open-Meteo Forecast API — real-time, free, no key")
            if wd.get("_is_fallback"):
                st.warning("Weather API unavailable — showing fallback defaults.")
            r1, r2, r3 = st.columns(3)
            r1.metric("Temperature",   f"{temp}C")
            r2.metric("Wind Speed",    f"{wind} km/h")
            r3.metric("Humidity",      f"{humid}%")
            r4, r5, r6 = st.columns(3)
            r4.metric("Precipitation", f"{precip} mm/hr",
                      delta="Raining" if wd["is_raining"] else "Dry",
                      delta_color="off")
            r5.metric("PM2.5 (est.)",  f"{pm25} ug/m3")
            r6.metric("Traffic Est.",  f"{traffic} veh/hr",
                      delta="Rush Hour" if is_rush else "Off-Peak",
                      delta_color="off")
        else:
            feats = None
            now      = datetime.now()
            hour     = st.slider("Hour of Day", 0, 23, now.hour)
            dow      = st.slider("Day of Week (0=Mon)", 0, 6, now.weekday())
            month    = now.month
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
        if feats is not None:
            pred = predict_row(model, *feats)
        else:
            pred = predict_row(model, hour, dow, month, is_wknd, is_rush, is_night,
                               traffic, temp, wind, precip, humid, pm25, city=city)
        cat, icon, _ = noise_category(pred)
        st.plotly_chart(gauge_chart(pred), use_container_width=True)

        weather_ctx = {"temperature": temp, "wind_speed": wind,
                       "precipitation": precip, "humidity": humid}
        fn = {"info": st.info, "warning": st.warning,
              "success": st.success, "error": st.error}
        for ins in generate_realtime_insights(pred, city, traffic, weather_ctx):
            fn.get(ins["severity"], st.info)(
                f"{ins['icon']} **{ins['title']}** — {ins['text']}"
            )

    st.divider()
    st.subheader(f"Hour-by-Hour Forecast — {city} (Today)")
    forecast_rows = []
    for h in range(24):
        is_r = int(7 <= h <= 9 or 16 <= h <= 19)
        is_n = int(0 <= h <= 5)
        t_est = traffic if h == hour else int(traffic * (1.3 if is_r else 0.6))
        p = predict_row(model, h, dow, month, is_wknd, is_r, is_n,
                        t_est, temp, wind, precip, humid, pm25, city=city)
        cat_h, icon_h, _ = noise_category(p)
        forecast_rows.append({"Hour": f"{h:02d}:00", "Predicted dB": round(p, 1),
                               "Category": f"{icon_h} {cat_h}", "Traffic Est.": t_est})
    forecast_df = pd.DataFrame(forecast_rows)
    fig = px.bar(forecast_df, x='Hour', y='Predicted dB',
                 color='Predicted dB', color_continuous_scale='RdYlGn_r',
                 title=f"24-Hour Noise Forecast — {city}")
    fig.add_hline(y=65, line_dash='dash', line_color='orange', annotation_text='65 dB')
    fig.add_hline(y=75, line_dash='dash', line_color='red',    annotation_text='75 dB')
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    col_tbl, col_sens = st.columns(2)
    with col_tbl:
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
    with col_sens:
        st.subheader("Sensitivity: Traffic vs Wind")
        t_vals = np.linspace(100, 3000, 15)
        w_vals = np.linspace(0, 60, 15)
        z = [[predict_row(model, hour, dow, month, is_wknd, is_rush,
                          is_night, int(t), temp, w, precip, humid, pm25, city=city) 
              for t in t_vals] for w in w_vals]
        fig2 = go.Figure(data=go.Contour(
            z=z, x=t_vals, y=w_vals, colorscale='RdYlGn_r',
            colorbar=dict(title='dB'), hovertemplate='Traffic: %{x:.0f}<br>Wind: %{y:.1f} km/h<br>Noise: %{z:.1f} dB<extra></extra>'
        ))
        fig2.update_layout(xaxis_title='Traffic (veh/hr)', yaxis_title='Wind Speed (km/h)',
                           margin=dict(t=10, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Multi-City Batch Prediction (Current Conditions)")
    with st.spinner("Fetching live data for all cities..."):
        batch_feats = {}
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {ex.submit(get_realtime_features, c): c for c in CITY_COORDS}
            try:
                for fut in as_completed(futures, timeout=10):
                    c = futures[fut]
                    try:
                        batch_feats[c] = fut.result()
                    except Exception as e:
                        logger.warning("Batch fetch failed for %s: %s", c, e)
                        batch_feats[c] = None
            except TimeoutError:
                logger.warning("Batch fetch timeout.")
    batch_rows = []
    for c in CITY_COORDS:
        f = batch_feats.get(c)
        if f is None:
            batch_rows.append({"City": c, "Predicted dB": "N/A",
                                "Category": "Unavailable",
                                "Traffic": "-", "Temp (C)": "-",
                                "Wind (km/h)": "-", "Humidity (%)": "-", "PM2.5": "-"})
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
elif page == "💬 Chatbot":
    st.title("Noise Analyzer Chatbot")
    st.markdown("Powered by **Groq AI (Llama 3.1)** — multi-turn conversation with full dataset context.")

    bot = get_chatbot()

    st.markdown("**Quick questions:**")
    chips = [
        "Which city is the loudest?",
        "Predict noise in Cairo now",
        "Compare New York vs Tokyo",
        "What are the health risks of loud noise?",
        "Show noise trends over months",
        "How accurate is the model?",
        "Tips to reduce urban noise",
    ]
    chip_cols = st.columns(len(chips))
    for i, chip in enumerate(chips):
        if chip_cols[i].button(chip, key=f"chip_{i}", use_container_width=True):
            st.session_state.setdefault("messages", [])
            st.session_state["pending_chip"] = chip

    st.divider()

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": (
            "Hello! I am the Smart Urban Noise Analyzer.\n\n"
            "I can help you with:\n"
            "- Noise predictions for any city\n"
            "- Data insights from 21,840 records across 5 cities\n"
            "- City comparisons (New York, London, Cairo, Tokyo, Paris)\n"
            "- Trend analysis — hourly, daily, monthly patterns\n"
            "- Health impact of noise pollution\n"
            "- Noise reduction advice\n"
            "- Model performance details\n\n"
            "What would you like to know?"
        )}]

    dispatch = {
        "PREDICTION": bot.handle_prediction,
        "COMPARE":    bot.handle_compare,
        "ADVICE":     bot.handle_advice,
        "MODEL_INFO": bot.handle_model_info,
        "TREND":      bot.handle_trend,
        "HEALTH":     bot.handle_health,
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
elif page == "🧠 Model Info":
    st.title("Model Performance and Architecture")
    model, meta = load_model()
    df = load_data()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Cities",        df['City'].nunique())
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
            fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                         color='Importance', color_continuous_scale='Blues')
            fig.update_layout(showlegend=False, height=420, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Actual vs Predicted")
        X = df[FEATURES]
        y = df['Noise_Level_dB']
        idx = np.random.choice(len(df), size=min(800, len(df)), replace=False)
        preds     = model.predict(X.iloc[idx])
        actuals   = y.iloc[idx].values
        residuals = actuals - preds
        fig = px.scatter(x=actuals, y=preds, opacity=0.4,
                         labels={'x': 'Actual (dB)', 'y': 'Predicted (dB)'},
                         title='Actual vs Predicted Noise',
                         color=np.abs(residuals), color_continuous_scale='RdYlGn_r')
        fig.add_shape(type='line', x0=actuals.min(), y0=actuals.min(),
                      x1=actuals.max(), y1=actuals.max(),
                      line=dict(color='black', dash='dash'))
        fig.update_layout(coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Residual Distribution")
        fig = px.histogram(x=residuals, nbins=50,
                           title='Prediction Residuals (Actual minus Predicted)',
                           labels={'x': 'Residual (dB)', 'y': 'Count'},
                           color_discrete_sequence=['#3498db'])
        fig.add_vline(x=0, line_dash='dash', line_color='red')
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    with col4:
        st.subheader("Model Metrics Comparison")
        fig = go.Figure()
        fig.add_trace(go.Bar(name='RMSE', x=[r["Model"] for r in rows],
                             y=[r["RMSE"] for r in rows], marker_color='#e74c3c'))
        fig.add_trace(go.Bar(name='R2',   x=[r["Model"] for r in rows],
                             y=[r["R2"]   for r in rows], marker_color='#2ecc71'))
        fig.update_layout(barmode='group', height=320,
                          title='RMSE vs R2 by Model', yaxis_title='Score')
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Features Used")
    feat_desc = {
        'hour': 'Hour of day (0-23)',
        'day_of_week': 'Day of week (0=Mon)',
        'month': 'Month (1-12)',
        'is_weekend': 'Weekend flag (0/1)',
        'is_rush_hour': 'Rush hour flag (0/1)',
        'is_night': 'Night flag 0-5 (0/1)',
        'Traffic_Count': 'Estimated vehicles/hr',
        'temperature': 'Real air temperature (C) — Open-Meteo',
        'wind_speed': 'Real wind speed (km/h) — Open-Meteo',
        'precipitation': 'Real precipitation (mm/hr) — Open-Meteo',
        'humidity': 'Real relative humidity (%) — Open-Meteo',
        'pm25': 'PM2.5 estimated from humidity + precipitation',
        'hour_sin': 'Cyclic sin encoding of hour',
        'hour_cos': 'Cyclic cos encoding of hour',
        'month_sin': 'Cyclic sin encoding of month',
        'month_cos': 'Cyclic cos encoding of month',
        'heat_index': 'Perceived temperature (Steadman 1979)',
        'traffic_ratio': 'Traffic vs city baseline ratio',
    }
    fcols = st.columns(3)
    for i, feat in enumerate(FEATURES):
        fcols[i % 3].info(f"**{feat}**\n\n{feat_desc.get(feat, '')}")
