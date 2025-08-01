import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from ta.volatility import BollingerBands

# Streamlit Page Setup
st.set_page_config(page_title="MarketLens", layout="wide")

st.markdown("""
    <h1 style='text-align: center; color: #f94d4d;'>Market<span style='color:#4df98b;'>Lens</span></h1>
    <h4 style='text-align: center; color: white;'>AI-powered insights into stock market trends.</h4>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")
stock = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, GOOGL)", value="AAPL")
years = st.sidebar.slider("Forecast Horizon (Years)", 1, 5, 1)
show_ma = st.sidebar.multiselect(
    "Select Moving Averages to Display",
    options=["MA50", "MA100", "MA200"],
    default=["MA50", "MA100", "MA200"]
)

# Load model
model = load_model("stock_model.keras")

# Download Data
start = '2012-01-01'
end = '2024-08-18'
data = yf.download(stock, start=start, end=end)
st.subheader(f'{stock} Historical Stock Data')
st.dataframe(data.tail())

# Moving Averages
ma_50 = data['Close'].rolling(window=50).mean()
ma_100 = data['Close'].rolling(window=100).mean()
ma_200 = data['Close'].rolling(window=200).mean()

# Plot Price with Selected MAs
st.subheader("Price with Selected Moving Averages")
fig_ma = plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Close Price', color='green')

if "MA50" in show_ma:
    plt.plot(ma_50, label="MA50", color='red')
if "MA100" in show_ma:
    plt.plot(ma_100, label="MA100", color='blue')
if "MA200" in show_ma:
    plt.plot(ma_200, label="MA200", color='orange')

plt.legend()
plt.xlabel("Date")
plt.ylabel("Price")
st.pyplot(fig_ma)

# Bollinger Bands
st.subheader("Bollinger Bands")
boll = BollingerBands(close=data['Close'], window=20, window_dev=2)
bb_upper = boll.bollinger_hband()
bb_middle = boll.bollinger_mavg()
bb_lower = boll.bollinger_lband()

fig_boll = plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Close Price', color='green')
plt.plot(bb_upper, label='Upper Band', linestyle='--', color='red')
plt.plot(bb_middle, label='Middle Band', linestyle='--', color='blue')
plt.plot(bb_lower, label='Lower Band', linestyle='--', color='orange')
plt.fill_between(data.index, bb_lower, bb_upper, color='gray', alpha=0.1)
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig_boll)

# Prepare Data for Prediction
data_train = pd.DataFrame(data['Close'][0: int(len(data)*0.80)])
data_test = pd.DataFrame(data['Close'][int(len(data)*0.80):])
scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = data_train.tail(100)
final_test_data = pd.concat([past_100_days, data_test], ignore_index=True)
final_test_scaled = scaler.fit_transform(final_test_data)

x_test = []
y_test = []

for i in range(100, final_test_scaled.shape[0]):
    x_test.append(final_test_scaled[i-100:i])
    y_test.append(final_test_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predict and Rescale
predictions = model.predict(x_test)
scale_factor = 1 / scaler.scale_[0]
predictions = predictions.flatten() * scale_factor
y_test = y_test * scale_factor

# Confidence Interval
errors = y_test - predictions
std_dev = np.std(errors)
upper_bound = predictions + 1.96 * std_dev
lower_bound = predictions - 1.96 * std_dev

# Plot Predicted vs Actual
st.subheader("Actual vs Predicted Prices with Confidence Range")
fig_pred = plt.figure(figsize=(12,6))
plt.plot(y_test, label='Actual Price', color='green')
plt.plot(predictions, label='Predicted Price', color='red')
plt.fill_between(range(len(predictions)), lower_bound, upper_bound, color='orange', alpha=0.3, label='Confidence Range')
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
st.pyplot(fig_pred)

# Forecast Summary
st.subheader("📊 Latest Forecast Summary")
st.write(f"📈 Optimistic Price: ${upper_bound[-1]:.2f}")
st.write(f"📉 Pessimistic Price: ${lower_bound[-1]:.2f}")
st.write(f"🎯 Predicted Price: ${predictions[-1]:.2f}")
