import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="MarketLens", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #f94d4d;'>Market<span style='color:#4df98b;'>Lens</span></h1>
    <h4 style='text-align: center; color: white;'>AI-powered insights into stock market trends.</h4>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Stock Symbol")
symbol = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, GOOGL)", value="AAPL")
years = st.sidebar.slider("Forecast Years", 1, 5, 1)
future_days = years * 365
show_ma = st.sidebar.multiselect(
    "Select Moving Averages to Display",
    options=["MA50", "MA100", "MA200"],
    default=["MA50", "MA100", "MA200"]
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

    # Moving Averages
    st.subheader("Price vs Moving Averages")
    ma_50 = data.Close.rolling(50).mean()
    ma_100 = data.Close.rolling(100).mean()
    ma_200 = data.Close.rolling(200).mean()

    fig_ma = plt.figure(figsize=(10, 5))
    plt.plot(data['Close'], label='Close Price', color='green')
    if "MA50" in show_ma:
        plt.plot(ma_50, label="MA50", color='red')
    if "MA100" in show_ma:
        plt.plot(ma_100, label="MA100", color='blue')
    if "MA200" in show_ma:
        plt.plot(ma_200, label="MA200", color='orange')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig_ma)

    # Bollinger Bands
    st.subheader("Price with Bollinger Bands")
    rolling_mean = data['Close'].rolling(window=20).mean()
    rolling_std = data['Close'].rolling(window=20).std()
    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std

    fig_boll = plt.figure(figsize=(10, 5))
    plt.plot(data['Close'], label='Close Price', color='green')
    plt.plot(rolling_mean, label='20-Day MA', color='blue')
    plt.fill_between(data.index, lower_band, upper_band, color='lightblue', alpha=0.4, label='Bollinger Bands')
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig_boll)

    # Prepare data for prediction
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
        pred = model.predict(input_sequence)[0][0]
        predicted.append(pred)
        input_sequence = np.append(input_sequence[:, 1:, :], [[[pred]]], axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predicted).reshape(-1, 1))

    # Forecast plot
    st.subheader("Forecasted Stock Price")
    forecast_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_days)
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Forecast": predicted_prices.flatten()
    })
    forecast_df["Upper"] = forecast_df["Forecast"] * 1.05
    forecast_df["Lower"] = forecast_df["Forecast"] * 0.95

    fig_forecast = plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data['Close'], label='Historical')
    plt.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', color='red')
    plt.fill_between(forecast_df['Date'], forecast_df['Lower'], forecast_df['Upper'],
                     color='red', alpha=0.2, label='Confidence Interval (±5%)')
    plt.title("Stock Price Forecast with Confidence Range")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig_forecast)

    st.subheader("Forecast Data Table")
    st.dataframe(forecast_df.tail())
