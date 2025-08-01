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
symbol = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, GOOGL)", value="")
years = st.sidebar.slider("Forecast Years", 1, 5, 1)
future_days = years * 365

show_ma = st.sidebar.multiselect(
    "Select Moving Averages",
    options=["MA50", "MA100", "MA200"],
    default=["MA50", "MA100", "MA200"]
)

show_bb = st.sidebar.checkbox("Show Bollinger Bands", value=True)
optimism_range = st.sidebar.slider("Optimism/Pessimism Range (%)", 1, 20, 5)

@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start="2012-01-01")
    df.reset_index(inplace=True)
    return df

if symbol:
    with st.spinner("Loading data..."):
        data = load_data(symbol)
    st.success("Data loaded successfully!")

    st.subheader(f"{symbol.upper()} Stock Data")
    st.dataframe(data.tail())

    # Calculate MAs
    ma_50 = data['Close'].rolling(50).mean()
    ma_100 = data['Close'].rolling(100).mean()
    ma_200 = data['Close'].rolling(200).mean()

    # Bollinger Bands
    sma_20 = data['Close'].rolling(window=20).mean()
    std_20 = data['Close'].rolling(window=20).std()
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)

    # Plot
    st.subheader("Price Chart with Indicators")
    fig_ma = plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data['Close'], label='Close Price', color='green')
    if "MA50" in show_ma:
        plt.plot(data['Date'], ma_50, label="MA50", color='red')
    if "MA100" in show_ma:
        plt.plot(data['Date'], ma_100, label="MA100", color='blue')
    if "MA200" in show_ma:
        plt.plot(data['Date'], ma_200, label="MA200", color='orange')
    if show_bb:
        plt.plot(data['Date'], upper_bb, label='Upper Bollinger Band', linestyle='--', color='magenta')
        plt.plot(data['Date'], lower_bb, label='Lower Bollinger Band', linestyle='--', color='cyan')
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
        st.error(f"Model loading failed: {e}")
        st.stop()

    past_100 = scaled_data[-100:]
    input_seq = past_100.reshape(1, 100, 1)

    predictions = []
    for _ in range(future_days):
        pred = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    forecast_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    forecast_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_days)
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Forecast": forecast_prices.flatten()
    })

    # Optimistic & Pessimistic Range
    optimism = (optimism_range / 100) * forecast_df["Forecast"]
    forecast_df["Optimistic"] = forecast_df["Forecast"] + optimism
    forecast_df["Pessimistic"] = forecast_df["Forecast"] - optimism

    # Forecast Plot
    st.subheader("Forecasted Stock Price")
    fig_forecast = plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data['Close'], label='Historical')
    plt.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', color='red')
    plt.fill_between(
        forecast_df['Date'],
        forecast_df['Pessimistic'],
        forecast_df['Optimistic'],
        color='orange', alpha=0.3,
        label=f"±{optimism_range}% Range"
    )
    plt.title("AI Forecast with Confidence Range")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig_forecast)

    st.subheader("Forecast Data Table")
    st.dataframe(forecast_df.tail())

    # Optional: Download CSV
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Forecast Data (CSV)", csv, file_name=f"{symbol.upper()}_forecast.csv", mime="text/csv")

else:
    st.warning("Please enter a stock ticker symbol in the sidebar to get started.")
