import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import date

# Page setup
st.set_page_config(page_title="MarketLens", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #f94d4d;'>Market<span style='color:#4df98b;'>Lens</span></h1>
    <h4 style='text-align: center; color: white;'>AI-powered insights into stock market trends.</h4>
""", unsafe_allow_html=True)

# Sidebar Inputs
st.sidebar.title("🔍 Stock Configuration")
symbol = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, GOOGL)", value="")
years = st.sidebar.slider("Forecast Years", 1, 5, 1)
future_days = years * 365
show_ma = st.sidebar.multiselect("Select Moving Averages", ["MA50", "MA100", "MA200"], default=["MA50", "MA100", "MA200"])
show_bb = st.sidebar.checkbox("Show Bollinger Bands", value=True)
scenario_price = st.sidebar.number_input("🔮 Simulate: Next-day Price", min_value=0.0, value=0.0, step=0.5)

# Load data
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start="2012-01-01")
    df.reset_index(inplace=True)
    return df

if symbol:
    with st.spinner("Fetching stock data..."):
        data = load_data(symbol)
    st.success("Data loaded!")

    # Date Filter
    min_date = data['Date'].min().date()
    max_date = data['Date'].max().date()
    date_range = st.slider("Select Date Range for Historical Plot", min_value=min_date, max_value=max_date,
                           value=(max_date.replace(year=max_date.year - 1), max_date))
    data = data[(data['Date'].dt.date >= date_range[0]) & (data['Date'].dt.date <= date_range[1])]

    st.subheader(f"{symbol} Stock Data")
    st.dataframe(data.tail())

    # --- Plotting MA + Bollinger Bands ---
    st.subheader("📊 Price & Technical Indicators")
    fig_ma = plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data['Close'], label='Close Price', color='green')

    if "MA50" in show_ma:
        plt.plot(data['Date'], data['Close'].rolling(50).mean(), label='MA50', color='red')
    if "MA100" in show_ma:
        plt.plot(data['Date'], data['Close'].rolling(100).mean(), label='MA100', color='blue')
    if "MA200" in show_ma:
        plt.plot(data['Date'], data['Close'].rolling(200).mean(), label='MA200', color='orange')

    if show_bb:
        sma_20 = data['Close'].rolling(window=20).mean()
        std_20 = data['Close'].rolling(window=20).std()
        upper_bb = sma_20 + (2 * std_20)
        lower_bb = sma_20 - (2 * std_20)
        plt.plot(data['Date'], upper_bb, label='Upper BB', linestyle='--', color='magenta')
        plt.plot(data['Date'], lower_bb, label='Lower BB', linestyle='--', color='cyan')

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig_ma)

    # --- LSTM Forecasting ---
    st.subheader("🤖 Forecasted Stock Price")

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

    # Plot Forecast
    fig_forecast = plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data['Close'], label='Historical')
    plt.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', color='red')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title("Stock Price Forecast")
    plt.legend()
    st.pyplot(fig_forecast)

    st.subheader("📁 Forecast Data Table")
    st.dataframe(forecast_df.tail())
    st.download_button("📥 Download Forecast CSV", forecast_df.to_csv(index=False), "forecast.csv")

    # --- 📈 Buy/Sell Window Suggestion ---
    st.subheader("🧠 Suggested Buy/Sell Strategy")
    min_index = forecast_df['Forecast'].idxmin()
    buy_date = forecast_df.loc[min_index, 'Date']
    buy_price = forecast_df.loc[min_index, 'Forecast']

    post_buy_df = forecast_df[min_index:]
    max_index = post_buy_df['Forecast'].idxmax()
    sell_date = forecast_df.loc[max_index, 'Date']
    sell_price = forecast_df.loc[max_index, 'Forecast']

    st.markdown(f"""
    - 💰 **Buy on:** `{buy_date.date()}` at **${buy_price:.2f}**
    - 💸 **Sell on:** `{sell_date.date()}` at **${sell_price:.2f}**
    - 📈 **Profit Potential:** `${sell_price - buy_price:.2f}`
    """)

    # --- 🔥 Volatility Forecast ---
    st.subheader("📊 Forecast Volatility")
    volatility = np.std(forecast_df['Forecast'])
    forecast_range = forecast_df['Forecast'].max() - forecast_df['Forecast'].min()

    if volatility < 5:
        label = "🔵 Low"
    elif volatility < 15:
        label = "🟡 Medium"
    else:
        label = "🔴 High"

    st.markdown(f"""
    - 📉 **Std Deviation:** {volatility:.2f}
    - ↕️ **Price Range:** ${forecast_range:.2f}
    - 🔍 **Volatility Level:** {label}
    """)

    # --- 🔮 Scenario Simulation ---
    st.subheader("🧪 Scenario Forecast (Custom Start)")
    if scenario_price > 0:
        custom_scaled = scaler.transform([[scenario_price]])
        input_seq = np.append(past_100[1:], [[custom_scaled[0]]], axis=0).reshape(1, 100, 1)

        scenario_pred = []
        for _ in range(future_days):
            pred = model.predict(input_seq, verbose=0)[0][0]
            scenario_pred.append(pred)
            input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

        scenario_prices = scaler.inverse_transform(np.array(scenario_pred).reshape(-1, 1))
        scenario_df = pd.DataFrame({"Date": forecast_dates, "Scenario Forecast": scenario_prices.flatten()})

        fig_scenario = plt.figure(figsize=(10, 5))
        plt.plot(data['Date'], data['Close'], label='Historical')
        plt.plot(scenario_df['Date'], scenario_df['Scenario Forecast'], label='Scenario Forecast', color='purple')
        plt.legend()
        st.pyplot(fig_scenario)
