import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report


st.set_page_config(page_title="Reliance Stock Dashboard", layout="wide")
st.title("📈 Reliance Stock Dashboard")

# <--- DATE FILTERS --->
st.markdown("### 📅 Select Date Range")
col1, col2 = st.columns([1, 1])
with col1:
    from_date = st.date_input("From", date.today() - timedelta(days=365))
with col2:
    to_date = st.date_input("To", date.today())

# <--- DATA DOWNLOAD --->
@st.cache_data
def get_data(ticker):
    df = yf.download("RELIANCE.NS", period="2y", interval="1d")
    df.columns = df.columns.get_level_values(0)  # ✅ Flatten multi-index
    df.reset_index(inplace=True)
    return df

df = get_data("RELIANCE.NS")
df = df[(df['Date'] >= pd.to_datetime(from_date)) & (df['Date'] <= pd.to_datetime(to_date))]

if df.empty:
    st.error("⚠️ No data available for the selected date range.")
    st.stop()

# <--- CALCULATIONS --->
df['Daily Return'] = df['Close'].pct_change()
df['MA20'] = df['Close'].rolling(window=20).mean()
df['STD20'] = df['Close'].rolling(window=20).std()
df['Upper'] = df['MA20'] + 2 * df['STD20']
df['Lower'] = df['MA20'] - 2 * df['STD20']

# <--- KPIs --->
df = df.sort_values(by='Date').reset_index(drop=True)

if len(df) >= 2:
    latest_close = float(df['Close'].iloc[-1])
    prev_close = float(df['Close'].iloc[-2])
    price_change_pct = ((latest_close - prev_close) / prev_close) * 100
else:   
    latest_close = prev_close = price_change_pct = 0.0

avg_volume = float(df['Volume'].rolling(window=5).mean().iloc[-1])
sma_20 = float(df['MA20'].iloc[-1])
volatility = float(df['STD20'].iloc[-1])

st.markdown("## 📌 Key Performance Indicators")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Latest Close", f"₹{latest_close:.2f}", f"{price_change_pct:.2f}%")
k2.metric("20-Day SMA", f"₹{sma_20:.2f}")
k3.metric("5-Day Avg Volume", f"{avg_volume:.0f}")
k4.metric("Volatility (20D STD)", f"{volatility:.2f}")

# <--- SECTION SELECTORS AS BUTTONS --->
st.markdown("---")
st.markdown("### 📊 Select Visualization")
bt1, bt2, bt3, bt4, bt5, bt6 = st.columns(6)

show_ohlc = bt1.button("📊 OHLC Chart")
show_bb = bt2.button("📉 Bollinger Bands")
show_volume = bt3.button("📦 Volume")
show_d_return = bt4.button("📈 Daily Return")
show_candle = bt5.button("🕯️ Candlestick")
show_raw = bt6.button("🧾 Raw Data")

section_flags = {
    "ohlc": show_ohlc,
    "bb": show_bb,
    "d_return": show_d_return,
    "volume": show_volume,
    "candle": show_candle,
    "raw": show_raw
}

# <--- Enhanced Canvas Design --->
st.markdown("""
    <style>
    .canvas-box {
        background: #FFFFF0 !important;  /* Ivory background */
        border-radius: 20px !important;
        padding: 30px !important;
        margin-bottom: 35px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12) !important;
        border: 2px solid #F5F5DC !important;  /* Beige border for definition */
        position: relative !important;
    }
    
    /* Create canvas background specifically for charts */
    .chart-container {
        position: relative !important;
        background: #FFFFF0 !important;  /* Ivory background */
        border-radius: 15px !important;
        padding: 20px !important;
        margin: 10px 0 !important;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.05) !important;
        border: 1px solid #F5F5DC !important;
    }
    
    /* Ensure plotly charts render properly on the canvas */
    .chart-container .js-plotly-plot {
        background: transparent !important;
        position: relative !important;
        z-index: 2 !important;
    }
    
    /* Style the dataframe container */
    .canvas-box .dataframe {
        background: rgba(255, 255, 255, 0.9) !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }

    .stButton button {
        font-size: 16px !important;
        padding: 10px 22px !important;
        border-radius: 8px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15) !important;
    }
    </style>
""", unsafe_allow_html=True)

plot='rgba(255,229,204,1)',  # Transparent plot background
paper='rgba(1,100,100,1)'  # Transparent paper background
# <--- RAW DATA --->
if section_flags["raw"]:
    st.markdown('<div class="canvas-box">', unsafe_allow_html=True)
    st.subheader("🧾 Raw Data")
    st.dataframe(df, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# <--- OHLC CHART --->
if section_flags["ohlc"]:
    st.markdown('<div class="canvas-box">', unsafe_allow_html=True)
    st.subheader("📊 OHLC Line Chart")
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig_ohlc = go.Figure()
    fig_ohlc.add_trace(go.Scatter(x=df['Date'], y=df['Open'], mode='lines', name='Open', line=dict(width=2)))
    fig_ohlc.add_trace(go.Scatter(x=df['Date'], y=df['High'], mode='lines', name='High', line=dict(width=2)))
    fig_ohlc.add_trace(go.Scatter(x=df['Date'], y=df['Low'], mode='lines', name='Low', line=dict(width=2)))
    fig_ohlc.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close', line=dict(width=3)))
    fig_ohlc.update_layout(
        title='OHLC Prices Over Time', 
        xaxis_title='Date', 
        yaxis_title='Price (INR)', 
        hovermode='x unified',
        plot_bgcolor=plot,  # Transparent plot background
        paper_bgcolor=paper,  # Transparent paper background
    )
    st.plotly_chart(fig_ohlc, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# <--- BB CHART --->
if section_flags["bb"]:
    st.markdown('<div class="canvas-box">', unsafe_allow_html=True)
    st.subheader("📉 Bollinger Bands")
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig_bb = go.Figure()
    fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', mode='lines', line=dict(width=3)))
    fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['Upper'], name='Upper Band', line=dict(dash='dot', width=2)))
    fig_bb.add_trace(go.Scatter(x=df['Date'], y=df['Lower'], name='Lower Band', line=dict(dash='dot', width=2)))
    fig_bb.update_layout(
        title='Bollinger Bands (20-day)', 
        xaxis_title='Date', 
        yaxis_title='Price', 
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
        paper_bgcolor='rgba(0,0,0,0)'  # Transparent paper background
    )
    st.plotly_chart(fig_bb, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# <--- RETURN CHART --->
if section_flags["d_return"]:
    st.markdown('<div class="canvas-box">', unsafe_allow_html=True)
    st.subheader("📈 Daily Return (%)")
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig_return = px.line(df, x='Date', y='Daily Return', title='Daily Returns (%)')
    fig_return.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
        paper_bgcolor='rgba(0,0,0,0)'  # Transparent paper background
    )
    fig_return.update_traces(line=dict(width=2))
    st.plotly_chart(fig_return, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# <--- VOLUME CHART --->
if section_flags['volume']:
    st.markdown('<div class="canvas-box">', unsafe_allow_html=True)
    st.subheader("📦 Volume Over Time")
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    fig_vol = px.bar(df, x='Date', y='Volume', title='Volume')
    fig_vol.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
        paper_bgcolor='rgba(0,0,0,0)'  # Transparent paper background
    )
    st.plotly_chart(fig_vol, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# <--- CANDLESTICK CHART --->
if section_flags["candle"]:
    st.markdown('<div class="canvas-box">', unsafe_allow_html=True)
    st.subheader("🕯️ Candlestick Chart")
    fig_candle = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])
    fig_candle.update_layout(
        title="Candlestick Chart", 
        xaxis_title='Date', 
        yaxis_title='Price (INR)',
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background
        paper_bgcolor='rgba(0,0,0,0)'  # Transparent paper background
    )
    st.plotly_chart(fig_candle, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Feature engineering
import numpy as np
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
df['MA5'] = df['Close'].rolling(window=5).mean() #Moving Average 5 days
df['MA10'] = df['Close'].rolling(window=10).mean() #Moving Average 10 days
df['Momentum'] = df['Close'] - df['Close'].shift(5) #Speed of price movement - to check if it goes up or down 
df['STD5'] = df['Close'].rolling(window=5).std() #Standard deviation - volatility indicator for stability
df['Upper'] = df['MA5'] + 2 * df['STD5'] #Moving average + 2 * SD
df['Lower'] = df['MA5'] - 2 * df['STD5'] #Moving average - 2 * SD
df['Volume_Change'] = df['Volume'].pct_change().replace([np.inf, -np.inf], 0)

df.dropna(inplace=True)

features = ['Open', 'High', 'Low', 'Close']
X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

model = XGBClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=1,
    min_child_weight=3,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

print(accuracy)
print(report)

import matplotlib.pyplot as plt #Feature Importance
importances = model.feature_importances_
features = X.columns

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(importance_df)
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='teal')
plt.xlabel("Importance Score")
plt.title("Feature Importances from Random Forest")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.show()