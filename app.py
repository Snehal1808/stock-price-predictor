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
st.sidebar.title("Stock Settings")
symbol = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, GOOGL)", value="AAPL")
years = st.sidebar.slider("Forecast Years", 1, 5, 1)
future_days = years * 365

show_ma = st.sidebar.multiselect(
    "Select Moving Averages",
    options=["MA50", "MA100", "MA200"],
    default=["MA50", "MA100", "MA200"]
)

show_rsi = st.sidebar.checkbox("Show RSI (14-day)", value=True)

@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start="2012-01-01")
    df.reset_index(inplace=True)
    return df

# RSI Calculation
def compute_rsi(data, period=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

if symbol:
    with st.spinner("Loading data..."):
        data = load_data(symbol)
    st.success("Data loaded successfully!")

    st.subheader(f"{symbol} Stock Data")
    st.dataframe(data.tail())

    # Technical Indicators
    data['MA50'] = data['Close'].rolling(window=50).mean()
    data['MA100'] = data['Close'].rolling(window=100).mean()
    data['MA200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = compute_rsi(data)

    plot_df = data.dropna().tail(180)

    # Price + MA Chart
    st.subheader("Price Chart with Indicators")
    fig_price, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(plot_df['Date'], plot_df['Close'], label='Close Price', color='green', linewidth=2)
    if "MA50" in show_ma:
        ax1.plot(plot_df['Date'], plot_df['MA50'], label="MA50", color='red', linewidth=1)
    if "MA100" in show_ma:
        ax1.plot(plot_df['Date'], plot_df['MA100'], label="MA100", color='blue', linewidth=1)
    if "MA200" in show_ma:
        ax1.plot(plot_df['Date'], plot_df['MA200'], label="MA200", color='orange', linewidth=1)

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price")
    ax1.set_title("Recent Price Chart")
    ax1.legend()
    ax1.grid(True)
    st.pyplot(fig_price)

        # RSI Chart
    if show_rsi:
        st.subheader("RSI (Relative Strength Index)")
        rsi_df = data[['Date', 'RSI']].dropna().tail(180)

        fig_rsi, ax_rsi = plt.subplots(figsize=(12, 3))
        ax_rsi.plot(rsi_df['Date'], rsi_df['RSI'], color='purple', label='RSI (14)')
        ax_rsi.axhline(70, color='red', linestyle='--', linewidth=1, label='Overbought (70)')
        ax_rsi.axhline(30, color='blue', linestyle='--', linewidth=1, label='Oversold (30)')
        ax_rsi.set_xlabel("Date")
        ax_rsi.set_ylabel("RSI")
        ax_rsi.set_title("14-Day RSI")
        ax_rsi.legend()
        ax_rsi.grid(True)
        fig_rsi.autofmt_xdate()
        st.pyplot(fig_rsi)


    # Prepare for Prediction
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
