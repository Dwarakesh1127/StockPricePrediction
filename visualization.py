import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import matplotlib.pyplot as plt

# def with_background(plot_fn):
#     def wrapper(*args, **kwargs):
#         st.markdown(
#             """
#             <div style="
#                 background-color: #f9f959;
#                 border-radius: 15px;
#                 padding: 25px;
#                 margin: 15px 0;
#                 box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
#             ">
#             """,
#             unsafe_allow_html=True
#         )
#         plot_fn(*args, **kwargs)
#         st.markdown("</div>", unsafe_allow_html=True)
#     return wrapper

bgc = "#ECE09C"
paper = "#435543"
#@with_background
def show_ohlc_chart(df):
    st.subheader("📊 OHLC Line Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Open'], mode='lines', name='Open'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['High'], mode='lines', name='High'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Low'], mode='lines', name='Low'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close'))
    fig.update_layout(plot_bgcolor=bgc, paper_bgcolor=paper,title="OHLC Prices", xaxis_title="Date", yaxis_title="Price (INR)")
    st.plotly_chart(fig, use_container_width=True)

#@with_background
def show_bollinger_bands(df):
    st.subheader("📉 Bollinger Bands")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', mode='lines'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Upper'], name='Upper Band', line=dict(dash='dot')))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Lower'], name='Lower Band', line=dict(dash='dot')))
    fig.update_layout(plot_bgcolor=bgc, paper_bgcolor=paper,title="Bollinger Bands", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

# @with_background
def show_daily_return(df):
    st.subheader("📈 Daily Return (%)")
    fig = px.line(df, x='Date', y='Daily Return', title='Daily Return Over Time')
    fig.update_layout(
        plot_bgcolor=bgc, 
        paper_bgcolor=paper,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Daily Return (%)"
    )
    fig.update_traces(line=dict(width=2))
    st.plotly_chart(fig, use_container_width=True)

# @with_background
def show_volume(df):
    st.subheader("📦 Volume")
    fig = px.bar(df, x='Date', y='Volume', title='Volume Over Time')
    fig.update_layout(
        plot_bgcolor=bgc,
        paper_bgcolor=paper,
        xaxis_title="Date",
        yaxis_title="Volume"
    )
    st.plotly_chart(fig, use_container_width=True)

#@with_background
def show_candlestick(df):
    st.subheader("🕯️ Candlestick Chart")
    fig = go.Figure(data=[go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])
    fig.update_layout(plot_bgcolor=bgc, paper_bgcolor=paper,title="Candlestick Chart", xaxis_title="Date", yaxis_title="Price (INR)")
    st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns([1, 1])
with col1:
    def plot_feature_importance(importance_df, width=6, height=3):
        st.subheader("📌 Feature Importances")
        st.dataframe(importance_df)

        fig, ax = plt.subplots(figsize=(width, height))
        ax.barh(importance_df['Feature'], importance_df['Importance'], color='teal')
        ax.set_xlabel("Importance Score", fontsize=10)
        ax.set_title("Feature Importances from XGBoost", fontsize=12)
        ax.invert_yaxis()
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        st.pyplot(fig)