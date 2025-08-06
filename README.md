# 📈 Stock Price Predictor  

**AI-powered stock price forecasts with real-time data—try it live!**  

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://marketlens18.streamlit.app/)   

This project predicts future stock prices using **LSTM** (Long Short-Term Memory) and other ML models. It’s designed for traders, investors, and ML enthusiasts who want data-driven market insights.  

## ✨ Features

### 🧠 AI Prediction Engine
- **4-layer LSTM model** with dropout regularization (94% accuracy)
- **10 years of historical data** from Yahoo Finance
- **Multi-day forecasting** (1-5 year predictions)
- **Technical indicators**: 50/100/200-day moving averages

### 📊 Interactive Dashboard
- Real-time stock data visualization
- Buy/sell recommendation engine
- Volatility risk assessment
- Customizable moving average displays

## 🚀 Try It Now!  
The app is live on Streamlit—no installation needed!  
👉 **[Launch App](https://marketlens18.streamlit.app/)**  

## 🖼️ Screenshots
<img width="1918" height="889" alt="Screenshot 2025-08-06 200319" src="https://github.com/user-attachments/assets/fbb1aa1f-5ce5-4e60-b9a3-05e7ab5c59a1" />
<img width="1907" height="891" alt="Screenshot 2025-08-06 200624" src="https://github.com/user-attachments/assets/f12c26ce-2e64-43c0-b929-d60118989700" />
<img width="1919" height="895" alt="Screenshot 2025-08-06 200646" src="https://github.com/user-attachments/assets/f295c137-7d40-49aa-a983-6672c82a825c" />
<img width="1917" height="875" alt="Screenshot 2025-08-06 200704" src="https://github.com/user-attachments/assets/c9bac703-132d-4d8e-b261-6c5e0f1e7a08" />
<img width="1919" height="874" alt="Screenshot 2025-08-06 200719" src="https://github.com/user-attachments/assets/3aaaf8f9-4478-4c6d-9a49-5b18bfa7018b" />

## 🛠 Tech Stack

| Component       | Technologies Used |
|----------------|-------------------|
| **Core ML**    | TensorFlow/Keras, Scikit-learn |
| **Data**       | yfinance, Pandas, NumPy |
| **Visualization** | Matplotlib, Plotly |
| **Frontend**   | Streamlit |
| **Deployment** | Streamlit Cloud |

## 📂 Project Structure
```
marketlens/
├── app.py                  # Streamlit application
├── stock_model.keras       # Trained LSTM model
├── train_model.ipynb       # Jupyter notebook for model training
├── requirements.txt        # Dependencies
├── LICENSE
└── README.md
```

## 🧠 Model Architecture
```python
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(100, 1)),
    Dropout(0.2),
    LSTM(60, activation='relu', return_sequences=True),
    Dropout(0.3),
    LSTM(80, activation='relu', return_sequences=True),
    Dropout(0.4),
    LSTM(120, activation='relu'),
    Dropout(0.5),
    Dense(1)
])
```

## 📊 Performance Metrics
| Metric | Value |
|--------|-------|
| RMSE   | 1.89  |
| MAE    | 1.52  |
| R²     | 0.94  |

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

## 📜 License
Distributed under the MIT License. See [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) for more information.

## ✉️ Contact
Snehal Kumar Subudhi - snehalsubu18@gmail.com
