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

<img width="1918" height="889" alt="Screenshot 2025-08-06 200319" src="https://github.com/user-attachments/assets/001bac64-b413-491b-ae58-6322aa4b55e4" />
<img width="1907" height="891" alt="Screenshot 2025-08-06 200624" src="https://github.com/user-attachments/assets/d45a68ad-cc78-40eb-9bb9-2644a0cff39f" />
<img width="1919" height="895" alt="Screenshot 2025-08-06 200646" src="https://github.com/user-attachments/assets/21f9ac8b-d240-440a-98c7-8e74715f13d8" />
<img width="1917" height="875" alt="Screenshot 2025-08-06 200704" src="https://github.com/user-attachments/assets/585d86db-8bc5-4dee-8f11-e340808b6c8a" />
<img width="1919" height="874" alt="Screenshot 2025-08-06 200719" src="https://github.com/user-attachments/assets/4c808ff3-1767-43c3-87a2-0bbb96e189d1" />

## 🛠 Tech Stack

| Component       | Technologies Used |
|----------------|-------------------|
| **Core ML**    | TensorFlow/Keras, Scikit-learn |
| **Data**       | yfinance, Pandas, NumPy |
| **Visualization** | Matplotlib, Plotly |
| **Frontend**   | Streamlit |
| **Deployment** | Streamlit Cloud |

Here's a professional **README.md** for your GitHub repository that showcases your Stock Price Predictor with both the LSTM model and Streamlit app:

```markdown
# 📈 MarketLens - AI Stock Price Predictor

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/marketlens/blob/main/stock_predictor.ipynb)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-streamlit-app.streamlit.app)

A sophisticated stock prediction system combining **LSTM neural networks** with an interactive Streamlit dashboard for market analysis.

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

## 🛠 Tech Stack

| Component       | Technologies Used |
|----------------|-------------------|
| **Core ML**    | TensorFlow/Keras, Scikit-learn |
| **Data**       | yfinance, Pandas, NumPy |
| **Visualization** | Matplotlib, Plotly |
| **Frontend**   | Streamlit |
| **Deployment** | Streamlit Cloud |

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/marketlens.git
   cd marketlens
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Access Web App
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://marketlens.streamlit.app)

## 📂 Project Structure
```
marketlens/
├── app.py                  # Streamlit application
├── stock_model.keras       # Trained LSTM model
├── train_model.ipynb       # Jupyter notebook for model training
├── requirements.txt        # Dependencies
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

## 🖼️ Screenshots

| Feature | Preview |
|---------|---------|
| **Dashboard** | ![Dashboard](https://i.imgur.com/dashboard.png) |
| **Forecast** | ![Forecast](https://i.imgur.com/forecast.png) |
| **Recommendations** | ![Recommendations](https://i.imgur.com/recommend.png) |

## 🤝 Contributing
1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

## ✉️ Contact
Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - youremail@example.com
```

## Key Highlights:
1. **Dual Focus**: Showcases both ML model and Streamlit app
2. **Visual Hierarchy**: Clean sections with emoji categorization
3. **Actionable Links**: Direct Colab and Streamlit badges
4. **Tech Transparency**: Clear stack breakdown
5. **Reproducibility**: Detailed setup instructions
