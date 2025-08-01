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

# Load model
model = load_model("stock_model.keras")

# Sidebar
st.sidebar.title("Stock Symbol")
stock = st.sidebar.text_input("Enter Ticker Symbol (e.g., AAPL, GOOGL)", value="AAPL")
start = '2012-01-01'
end = '2024-08-18'

st.sidebar.title("Forecast Period")
years = st.sidebar.slider("Years", 1, 5, 1)
period_days = years * 365

# Load data
data = yf.download(stock, start ,end)

st.subheader(f'{stock} Historical Stock Data')
st.write(data.tail())

# Split data
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0,1))

past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test)

# MA plots
ma_50 = data.Close.rolling(50).mean()
ma_100 = data.Close.rolling(100).mean()
ma_200 = data.Close.rolling(200).mean()

st.subheader('Price vs MA50')
fig1 = plt.figure(figsize=(10,6))
plt.plot(data.Close, label='Closing Price', color='green')
plt.plot(ma_50, label='MA50', color='red')
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
fig2 = plt.figure(figsize=(10,6))
plt.plot(data.Close, label='Closing Price', color='green')
plt.plot(ma_50, label='MA50', color='red')
plt.plot(ma_100, label='MA100', color='blue')
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
fig3 = plt.figure(figsize=(10,6))
plt.plot(data.Close, label='Closing Price', color='green')
plt.plot(ma_100, label='MA100', color='red')
plt.plot(ma_200, label='MA200', color='blue')
plt.legend()
st.pyplot(fig3)

# Prepare test data
x_test = []
y_test = []

for i in range(100, data_test_scaled.shape[0]):
    x_test.append(data_test_scaled[i-100:i])
    y_test.append(data_test_scaled[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Predict
predictions = model.predict(x_test)
scale_factor = 1 / scaler.scale_[0]

predictions = predictions.flatten() * scale_factor
y_test = y_test * scale_factor

# Confidence interval
errors = y_test - predictions
std_dev = np.std(errors)
upper_bound = predictions + 1.96 * std_dev
lower_bound = predictions - 1.96 * std_dev

# Final plot
st.subheader('Actual vs Predicted with Confidence Range')
fig4 = plt.figure(figsize=(10,6))
plt.plot(y_test, label='Actual Price', color='green')
plt.plot(predictions, label='Predicted Price', color='red')
plt.fill_between(range(len(predictions)), lower_bound, upper_bound, color='orange', alpha=0.3, label='Confidence Range')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)

# Display final predicted points
st.subheader("Latest Forecast Summary")
st.write(f"📈 Optimistic Price: ${upper_bound[-1]:.2f}")
st.write(f"📉 Pessimistic Price: ${lower_bound[-1]:.2f}")
st.write(f"🎯 Predicted Price: ${predictions[-1]:.2f}")
