import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import io

st.set_page_config(page_title="MarketLens", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #f94d4d;'>Market<span style='color:#4df98b;'>Lens</span></h1>
    <h4 style='text-align: center; color: white;'>AI-powered insights into stock market trends.</h4>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📈 Configuration")
symbol = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, GOOGL)", value="AAPL")
years = st.sidebar.slider("Forecast Years", 1, 5, 1)
future_days = years * 365

show_ma = st.sidebar.multiselect(
    "Moving Averages",
    options=["MA50", "MA100", "MA200"],
    default=["MA50", "MA100", "MA200"]
)

show_bb = st.sidebar.checkbox("Show Bollinger Bands (20-day)", value=True)

alert_price = st.sidebar.number_input(
    "🔔 Alert if Forecast ≥", min_value=0.0, value=200.0, step=1.0
)

@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start="2012-01-01")
    df.reset_index(inplace=True)
    return df

if symbol:
    with st.spinner("Loading data..."):
        data = load_data(symbol)
    st.success("Data loaded successfully!")

    st.subheader(f"{symbol} Stock Data")
    st.dataframe(data.tail())

    # Calculate indicators
    ma_50 = data.Close.rolling(50).mean()
    ma_100 = data.Close.rolling(100).mean()
    ma_200 = data.Close.rolling(200).mean()

    # Bollinger Bands
    bb_window = 20
    sma = data['Close'].rolling(window=bb_window).mean()
    std = data['Close'].rolling(window=bb_window).std()
    upper_bb = sma + (2 * std)
    lower_bb = sma - (2 * std)

    # Plot price with overlays
    st.subheader("Price Chart with Technical Indicators")
    fig_ma = plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label='Close Price', color='green')

    if "MA50" in show_ma:
        plt.plot(data['Date'], ma_50, label="MA50", color='red')
    if "MA100" in show_ma:
        plt.plot(data['Date'], ma_100, label="MA100", color='blue')
    if "MA200" in show_ma:
        plt.plot(data['Date'], ma_200, label="MA200", color='orange')

    if show_bb:
        plt.plot(data['Date'], upper_bb, label='Upper BB', color='gray', linestyle='--')
        plt.plot(data['Date'], lower_bb, label='Lower BB', color='gray', linestyle='--')

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig_ma)

    # Forecasting
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])

    try:
        model = load_model("stock_model.keras")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    past_100 = scaled_data[-100:]
    input_sequence = past_100.reshape(1, 100, 1)

    predicted = []
    for _ in range(future_days):
        pred = model.predict(input_sequence, verbose=0)[0][0]
        predicted.append(pred)
        input_sequence = np.append(input_sequence[:, 1:, :], [[[pred]]], axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predicted).reshape(-1, 1))
    forecast_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_days)
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecast": predicted_prices.flatten()})

    # Forecast Chart
    st.subheader("Forecasted Stock Price")
    fig_forecast = plt.figure(figsize=(12, 5))
    plt.plot(data['Date'], data['Close'], label='Historical')
    plt.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', color='red')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Stock Price Forecast")
    plt.legend()
    st.pyplot(fig_forecast)

    # Forecast Table
    st.subheader("Forecast Data Table")
    st.dataframe(forecast_df.tail())

    # CSV Download
    csv = forecast_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Forecast as CSV",
        data=csv,
        file_name=f"{symbol}_forecast.csv",
        mime='text/csv'
    )
