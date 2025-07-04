import streamlit as st
import pandas as pd
from datetime import date, timedelta
import joblib
import os

from data_load import get_data, add_technical_indicators
from model import train_lightgbm
from visualization import (
    show_ohlc_chart,
    show_bollinger_bands,
    show_daily_return,
    show_volume,
    show_candlestick,
    plot_feature_importance
)

st.set_page_config(page_title="Reliance Stock Dashboard", layout="wide")
st.title("📈 Reliance Stock Dashboard")

# --- Date Picker ---
col1, col2 = st.columns([1, 1])
with col1:
    from_date = st.date_input("From", date.today() - timedelta(days=3650))
with col2:
    to_date = st.date_input("To", date.today())

# --- Load & Prepare Data ---
df = get_data("RELIANCE.NS")
df = df[(df['Date'] >= pd.to_datetime(from_date)) & (df['Date'] <= pd.to_datetime(to_date))]

if df.empty:
    st.error("⚠️ No data available for the selected date range.")
    st.stop()

df = add_technical_indicators(df)

# --- KPIs ---
df['Daily Return'] = df['Close'].pct_change()
df['MA20'] = df['Close'].rolling(window=20).mean()
df['STD20'] = df['Close'].rolling(window=20).std()
df = df.sort_values(by='Date').reset_index(drop=True)
avg_volume = float(df['Volume'].rolling(window=5).mean().iloc[-1])
sma_20 = float(df['MA20'].iloc[-1])
volatility = float(df['STD20'].iloc[-1])

if len(df) >= 2:
    latest_close = float(df['Close'].iloc[-1])
    prev_close = float(df['Close'].iloc[-2])
    price_change_pct = ((latest_close - prev_close) / prev_close) * 100
else:
    latest_close = prev_close = price_change_pct = 0.0

st.subheader("📌 KPI")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Latest Close", f"₹{latest_close:.2f}", f"{price_change_pct:.2f}%")
k2.metric("20-Day SMA", f"₹{sma_20:.2f}")
k3.metric("5-Day Avg Volume", f"{avg_volume:.0f}")
k4.metric("Volatility (20D STD)", f"{volatility:.2f}")

# --- Visualizations ---
btn_cols = st.columns(5)
chart_flags = {"OHLC": False, "Bollinger": False, "Return": False, "Volume": False, "Candlestick": False}

if btn_cols[0].button("📈 OHLC"):
    chart_flags["OHLC"] = True
if btn_cols[1].button("📉 Bollinger"):
    chart_flags["Bollinger"] = True
if btn_cols[2].button("📈 Return"):
    chart_flags["Return"] = True
if btn_cols[3].button("📦 Volume"):
    chart_flags["Volume"] = True
if btn_cols[4].button("🕯️ Candle"):
    chart_flags["Candlestick"] = True

st.markdown("---")
if chart_flags["OHLC"]:
    show_ohlc_chart(df)
elif chart_flags["Bollinger"]:
    show_bollinger_bands(df)
elif chart_flags["Return"]:
    show_daily_return(df)
elif chart_flags["Volume"]:
    show_volume(df)
elif chart_flags["Candlestick"]:
    show_candlestick(df)

# --- Model Training & Saving ---
st.markdown("---")
st.subheader("🧠 Model Training and Feature Importance")

features = ['Open', 'ratio', 'High', 'Close', 'Daily Return',
            'Volume', 'Volume_Change', 'MA20',
            'Upper','Lower','Momentum','Momentum5',
            'MA10', 'MA5','STD5', 'STD10', 'STD20',
            'Momentum10', 'Momentum15', 'Momentum30',
            'Monthly_Volume_Avg','Open_Close_Diff','lagVolume_Change',
            'lagClose5']

model, acc, report, feat_names, cm_fig, f1, precision, recall = train_lightgbm(df, features)
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

st.success(f"Model Accuracy: {acc*100:.2f}%")

# --- Load Previous Model Metrics if Available ---
model_path = "best_model.pkl"
metrics_path = "best_metrics.pkl"

previous_f1 = previous_recall = -1  # Defaults if no model exists yet
if os.path.exists(metrics_path):
    previous_metrics = joblib.load(metrics_path)
    previous_f1 = previous_metrics.get("f1", -1)
    previous_recall = previous_metrics.get("recall", -1)

# --- Auto Save if Model Improved ---
if (f1 > previous_f1) or (recall > previous_recall):
    joblib.dump(model, model_path)
    joblib.dump({"f1": f1, "recall": recall}, metrics_path)
    st.success("✅ Model Improved and Saved!")
else:
    st.info("ℹ️ Current model did not outperform saved model. No overwrite.")

# --- Feature Importance ---
plot_feature_importance(importance_df)

st.subheader("📊 Confusion Matrix")
st.pyplot(cm_fig)

# --- Scores ---
st.subheader("F1 Score")
st.warning(f"{f1:.4f}")

st.subheader("Precision Score")
st.warning(f"{precision:.4f}")

st.subheader("Recall Score")
st.warning(f"{recall:.4f}")
