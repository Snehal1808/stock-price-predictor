import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from io import StringIO

st.set_page_config(page_title="MarketLens", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #f94d4d;'>Market<span style='color:#4df98b;'>Lens</span></h1>
    <h4 style='text-align: center; color: white;'>AI-powered insights into stock market trends.</h4>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Stock Settings")
symbol = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, GOOGL)", value="AAPL")
years = st.sidebar.slider("Forecast Years", 1, 5, 1)
future_days = years * 365

show_ma = st.sidebar.multiselect(
    "Select Moving Averages",
    options=["MA50", "MA100", "MA200"],
    default=["MA50", "MA100", "MA200"]
)

show_bb = st.sidebar.checkbox("Show Bollinger Bands", value=True)

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

    # Technical Indicators
    ma_50 = data['Close'].rolling(window=50).mean()
    ma_100 = data['Close'].rolling(window=100).mean()
    ma_200 = data['Close'].rolling(window=200).mean()

    # Bollinger Bands
    sma = data['Close'].rolling(window=20).mean()
    std = data['Close'].rolling(window=20).std()
    upper_bb = sma + 2 * std
    lower_bb = sma - 2 * std

    # Combine into DataFrame for plotting
    plot_df = data.copy()
    plot_df['MA50'] = ma_50
    plot_df['MA100'] = ma_100
    plot_df['MA200'] = ma_200
    plot_df['SMA20'] = sma
    plot_df['UpperBB'] = upper_bb
    plot_df['LowerBB'] = lower_bb
    plot_df = plot_df.dropna().tail(180)  # last 180 days

    st.subheader("Price Chart with Indicators")

    fig_ma = plt.figure(figsize=(12, 6))
    plt.plot(plot_df['Date'], plot_df['Close'], label='Close Price', color='green', linewidth=2)

    if "MA50" in show_ma:
        plt.plot(plot_df['Date'], plot_df['MA50'], label="MA50", color='red', linewidth=1)
    if "MA100" in show_ma:
        plt.plot(plot_df['Date'], plot_df['MA100'], label="MA100", color='blue', linewidth=1)
    if "MA200" in show_ma:
        plt.plot(plot_df['Date'], plot_df['MA200'], label="MA200", color='orange', linewidth=1)

    if show_bb:
        plt.plot(plot_df['Date'], plot_df['UpperBB'], label='Upper BB', color='gray', linestyle='--', alpha=0.6)
        plt.plot(plot_df['Date'], plot_df['LowerBB'], label='Lower BB', color='gray', linestyle='--', alpha=0.6)

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Recent Price Chart with Indicators")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig_ma)

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
        pred = model.predict(input_sequence, verbose=0)[0][0]
        predicted.append(pred)
        input_sequence = np.append(input_sequence[:, 1:, :], [[[pred]]], axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predicted).reshape(-1, 1))

    # Forecast Plot
    st.subheader("Forecasted Stock Price")
    forecast_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_days)
    forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecast": predicted_prices.flatten()})

    fig_forecast = plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data['Close'], label='Historical')
    plt.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', color='red')
    plt.title("Stock Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig_forecast)

    st.subheader("Forecast Data Table")
    st.dataframe(forecast_df.tail())

    # Download CSV
    csv = forecast_df.to_csv(index=False)
    st.download_button(
        label="Download Forecast CSV",
        data=csv,
        file_name=f"{symbol}_forecast.csv",
        mime='text/csv'
    )
