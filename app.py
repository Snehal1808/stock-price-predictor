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
symbol = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, GOOGL)", value="")
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

    # Moving Averages Plot
    st.subheader("Price vs Moving Averages")
    ma_50 = data['Close'].rolling(50).mean()
    ma_100 = data['Close'].rolling(100).mean()
    ma_200 = data['Close'].rolling(200).mean()

    fig_ma = plt.figure(figsize=(10, 5))
    plt.plot(data['Date'], data['Close'], label='Close Price', color='green')
    if "MA50" in show_ma:
        plt.plot(data['Date'], ma_50, label="MA50", color='red')
    if "MA100" in show_ma:
        plt.plot(data['Date'], ma_100, label="MA100", color='blue')
    if "MA200" in show_ma:
        plt.plot(data['Date'], ma_200, label="MA200", color='orange')
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig_ma)

    # Forecast and LSTM Features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])

    if len(scaled_data) < 100:
        st.warning("⚠️ Not enough historical data to generate LSTM-based forecasts or features (need at least 100 closing prices).")
    else:
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
        st.subheader("📈 Forecasted Stock Price")
        fig_forecast = plt.figure(figsize=(10, 5))
        plt.plot(data['Date'], data['Close'], label='Historical')
        plt.plot(forecast_df['Date'], forecast_df['Forecast'], label='Forecast', color='red')
        plt.title("Stock Price Forecast")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(fig_forecast)

        st.subheader("📋 Forecast Data Table")
        st.dataframe(forecast_df.tail())

        # Buy/Sell Window Suggestion
        st.subheader("📍 Optimal Buy/Sell Window Suggestion")
        min_price_day = forecast_df.loc[forecast_df['Forecast'].idxmin()]
        max_price_day = forecast_df.loc[forecast_df['Forecast'].idxmax()]
        st.markdown(f"""
        - ✅ **Buy Recommendation:** {min_price_day['Date'].date()} at ${min_price_day['Forecast']:.2f}  
        - 📤 **Sell Recommendation:** {max_price_day['Date'].date()} at ${max_price_day['Forecast']:.2f}
        """)

        # Volatility Forecast
        st.subheader("📉 Volatility Forecast")
        std_dev = forecast_df['Forecast'].std()
        if std_dev < 5:
            volatility = "🔵 Low"
        elif std_dev < 15:
            volatility = "🟡 Medium"
        else:
            volatility = "🔴 High"
        st.markdown(f"Estimated forecast volatility: **{volatility}** (std: {std_dev:.2f})")

        # Scenario Simulation
        st.subheader("🧪 Scenario Simulation (What if...)")
        hypothetical_price = st.number_input("Enter a hypothetical price for tomorrow", min_value=0.0, value=float(data['Close'].iloc[-1]))
        if st.button("Simulate Future"):
            custom_input = np.append(scaled_data[-99:], [[scaler.transform([[hypothetical_price]])[0][0]]], axis=0)
            custom_sequence = custom_input.reshape(1, 100, 1)

            custom_predicted = []
            for _ in range(future_days):
                pred = model.predict(custom_sequence, verbose=0)[0][0]
                custom_predicted.append(pred)
                custom_sequence = np.append(custom_sequence[:, 1:, :], [[[pred]]], axis=1)

            custom_prices = scaler.inverse_transform(np.array(custom_predicted).reshape(-1, 1))
            custom_forecast_df = pd.DataFrame({
                "Date": forecast_dates,
                "Simulated Forecast": custom_prices.flatten()
            })

            fig_custom = plt.figure(figsize=(10, 5))
            plt.plot(data['Date'], data['Close'], label='Historical')
            plt.plot(custom_forecast_df['Date'], custom_forecast_df['Simulated Forecast'], label='Simulated Forecast', color='purple')
            plt.title("Simulated Forecast (Based on Hypothetical Price)")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            st.pyplot(fig_custom)
