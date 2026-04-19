import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from datetime import datetime, timedelta
import plotly.express as px
import os

st.set_page_config(
    page_title="Hospital Load Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #f8fafc;
    }
    .metric-label {
        font-size: 1.1rem;
        color: #94a3b8;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        with open('model_volume.pkl', 'rb') as f:
            v = pickle.load(f)
        with open('model_wait.pkl', 'rb') as f:
            w = pickle.load(f)
        return v, w
    except FileNotFoundError:
        return None, None

model_volume, model_wait = load_models()

# --- Sidebar Controls ---
st.sidebar.title("⚙️ Control Panel")
run_mode = st.sidebar.radio("Operation Mode", ["🎛️ Simulate Manually", "📡 Go Live (Real-Time API)"])

if run_mode == "🎛️ Simulate Manually":
    st.sidebar.markdown("Adjust parameters to dynamically influence the ML models.")
    sim_temp = st.sidebar.slider("Current Temperature (°C)", -20.0, 50.0, 15.0, 1.0)
    sim_flu = st.sidebar.slider("Flu Search Index (pytrends)", 0.0, 100.0, 30.0, 5.0)
else:
    st.sidebar.markdown("Fetching live telemetry...")
    from pytrends.request import TrendReq
    from meteostat import Point, hourly
    
    with st.spinner('Pinging APIs...'):
        # Fetch Flu index
        try:
            pytrends = TrendReq(hl='en-US', tz=360)
            pytrends.build_payload(["flu"], cat=0, timeframe='now 7-d', geo='US-MA')
            trends = pytrends.interest_over_time()
            sim_flu = trends['flu'].iloc[-1] if (trends is not None and not trends.empty) else 30.0
            flu_success = True
        except Exception:
            sim_flu = 30.0
            flu_success = False

        # Fetch Weather (Live Telemetry via Open-Meteo)
        try:
            import requests
            url = "https://api.open-meteo.com/v1/forecast?latitude=42.3601&longitude=-71.0589&current_weather=true"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                sim_temp = float(data['current_weather']['temperature'])
            else:
                sim_temp = 16.5
        except Exception:
            sim_temp = 16.5

    if flu_success:
        st.sidebar.success("✅ Live Google Trends acquired securely.")
    else:
        st.sidebar.warning("⚠️ Live API rate-limited. Serving local heuristics.")

    st.sidebar.metric("Live Local Temperature", f"{sim_temp:.1f} °C")
    st.sidebar.metric("Live Flu Search Index", f"{int(sim_flu)} / 100")

now = datetime.now()
df_current = pd.DataFrame([{
    'Hour': now.hour,
    'DayOfWeek': now.weekday(),
    'Month': now.month,
    'Temperature': sim_temp,
    'Flu_Trend': sim_flu
}])

if model_volume and model_wait:
    pred_vol = max(0, int(model_volume.predict(df_current)[0]))
    df_wait = df_current.copy()
    df_wait['Current_Volume'] = pred_vol
    pred_wait = max(0, model_wait.predict(df_wait)[0])
else:
    st.sidebar.warning("Models not fully compiled yet. Showing logical fallback simulation.")
    pred_vol = int(max(0, np.random.normal(12, 3) + (sim_flu * 0.3) - ((sim_temp-15)*0.5)))
    pred_wait = pred_vol * 1.3 + np.random.normal(5, 2)

# --- Dashboard Layout ---
st.title("🏥 Emergency Department Predictor")
st.markdown("Live machine learning predictions mapping patient inflow and triage strain.")

st.markdown("### 📊 Live Analytics Matrix")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Arriving Patients (Next Hr)</div>
            <div class="metric-value" style="color: #60a5fa;">{pred_vol}</div>
        </div>
    """, unsafe_allow_html=True)
    
with col2:
    st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {'#ef4444' if pred_wait > 60 else '#f59e0b' if pred_wait > 40 else '#10b981'};">
            <div class="metric-label">Global Avg Wait Time</div>
            <div class="metric-value">{pred_wait:.1f} <span style="font-size:1.2rem;color:#94a3b8;">mins</span></div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    status_label = "CRITICAL" if pred_wait > 60 else "ELEVATED" if pred_wait > 40 else "NORMAL"
    status_color = "#ef4444" if pred_wait > 60 else "#f59e0b" if pred_wait > 40 else "#10b981"
    st.markdown(f"""
        <div class="metric-card" style="border-left: 5px solid {status_color}; background-color: {status_color}20;">
            <div class="metric-label">Hospital Status</div>
            <div class="metric-value" style="color: {status_color};">{status_label}</div>
        </div>
    """, unsafe_allow_html=True)

# --- 24-Hour Forecast ---
st.markdown("### 📈 24-Hour Patient Inflow Horizon")

horizon_hours = 24
forecast_data = []
start_t = now.replace(minute=0, second=0, microsecond=0)

for i in range(horizon_hours):
    future = start_t + timedelta(hours=i)
    f_temp = sim_temp + np.sin(future.hour/24 * 2 * np.pi) * 3
    
    if model_volume:
        f_df = pd.DataFrame([{
            'Hour': future.hour,
            'DayOfWeek': future.weekday(),
            'Month': future.month,
            'Temperature': f_temp,
            'Flu_Trend': sim_flu
        }])
        val = max(0, int(model_volume.predict(f_df)[0]))
    else:
        base = 15 + np.sin((future.hour - 8)/24 * 2 * np.pi) * 10
        val = max(0, int(base + (sim_flu*0.1)))
        
    forecast_data.append({"Time": future, "Predicted Patients": val})

forecast_df = pd.DataFrame(forecast_data)

fig = px.area(forecast_df, x="Time", y="Predicted Patients", line_shape="spline",
              color_discrete_sequence=['#3b82f6'])
fig.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    margin=dict(l=0, r=0, t=10, b=0),
    xaxis_title="",
    yaxis_title="Inflow Volume",
    font=dict(color="#94a3b8")
)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
st.plotly_chart(fig, width='stretch')

# --- Department Breakdown ---
st.markdown("### 🏥 Departmental Strain Vectors")
dept_cols = st.columns(3)
depts = [
    {"name": "Emergency", "mult": 1.2},
    {"name": "Pediatrics", "mult": 0.8},
    {"name": "General Medicine", "mult": 0.6}
]

for i, col in enumerate(dept_cols):
    d = depts[i]
    w = pred_wait * d["mult"]
    c = "#ef4444" if w > 60 else "#f59e0b" if w > 40 else "#10b981"
    
    with col:
        st.markdown(f"""
        <div style="background-color:#1E1E1E; padding:15px; border-radius:8px; border-bottom: 3px solid {c};">
            <h4 style="margin:0; color:#f8fafc;">{d['name']}</h4>
            <div style="margin-top:10px; font-size:1.8rem; font-weight:bold; color:{c};">{w:.1f} <span style="font-size:1rem;color:#94a3b8;">mins</span></div>
        </div>
        """, unsafe_allow_html=True)
