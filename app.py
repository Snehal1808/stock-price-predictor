import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import date

st.set_page_config(page_title="MarketLens", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #f94d4d;'>Market<span style='color:#4df98b;'>Lens</span></h1>
    <h4 style='text-align: center; color: white;'>AI-powered insights into stock market trends.</h4>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Stock Controls")
symbol = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, GOOGL)", value="")
years = st.sidebar.slider("Forecast Years", 1, 5, 1)
future_days = years * 365
show_ma = st.sidebar.multiselect(
    "Select Moving Averages to Display",
    options=["MA50", "MA100", "MA200"],
    default=["MA50", "MA100", "MA200"]
)
show_bb = st.sidebar.checkbox("Show Bollinger Bands", value=True)

start_date = st.sidebar.date_input(
    "Select Start Date for Chart",
    value=pd.to_datetime("2022-01-01"),
    min_value=pd.to_datetime("2012-01-01"),
    max_value=date.today()
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

    # Filtered data for historical chart
    filtered_data = data[data['Date'] >= pd.to_datetime(start_date)].copy()

    # Indicators
    ma_50 = filtered_data['Close'].rolling(50).mean()
    ma_100 = filtered_data['Close'].rolling(100).mean()
    ma_200 = filtered_data['Close'].rolling(200).mean()
    sma_20 = filtered_data['Close'].rolling(20).mean()
    std_20 = filtered_data['Close'].rolling(20).std()
    upper_bb = sma_20 + (2 * std_20)
    lower_bb = sma_20 - (2 * std_20)

    # Plot Price vs Indicators
    st.subheader("Price with Indicators")
    fig_ma = plt.figure(figsize=(10, 5))
    plt.plot(filtered_data['Date'], filtered_data['Close'], label='Close Price', color='green')
    if "MA50" in show_ma:
        plt.plot(filtered_data['Date'], ma_50, label="MA50", color='red')
    if "MA100" in show_ma:
        plt.plot(filtered_data['Date'], ma_100, label="MA100", color='blue')
    if "MA200" in show_ma:
        plt.plot(filtered_data['Date'], ma_200, label="MA200", color='orange')
    if show_bb:
        plt.plot(filtered_data['Date'], upper_bb, label='Upper BB', linestyle='--', color='magenta')
        plt.plot(filtered_data['Date'], lower_bb, label='Lower BB', linestyle='--', color='cyan')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig_ma)

    # Prepare for Forecast
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

    # Add optimistic/pessimistic bands (±5%)
    optimistic = predicted_prices * 1.05
    pessimistic = predicted_prices * 0.95

    # Forecast plot
    st.subheader("Forecasted Stock Price")
    forecast_dates = pd.date_range(start=data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_days)
    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Forecast": predicted_prices.flatten(),
        "Optimistic": optimistic.flatten(),
        "Pessimistic": pessimistic.flatten()
    })

    fig_forecast = plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data['Close'], label='Historical')
    plt.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', color='red')
    plt.fill_between(forecast_df['Date'], forecast_df['Pessimistic'], forecast_df['Optimistic'],
                     color='red', alpha=0.2, label='Confidence Range')
    plt.title("Stock Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig_forecast)

    st.subheader("Forecast Data Table")
    st.dataframe(forecast_df.tail())

    # Download button
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Forecast as CSV",
        data=csv,
        file_name=f"{symbol}_forecast.csv",
        mime="text/csv"
    )
else:
    st.info("Please enter a ticker symbol to get started.")
