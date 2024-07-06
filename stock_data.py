import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
print("scikit-learn is installed and imported successfully.")
from sklearn.model_selection import train_test_split
import numpy as np
import pmdarima as pm
import logging
import matplotlib.pyplot as plt


def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        if hist.empty:
            logging.error(f"No data returned for ticker {ticker}")
            return None
        logging.info(f"Data fetched for {ticker}: {hist.head()}")
        return hist
    except Exception as e:
        logging.error(f"Error fetching data for ticker {ticker}: {e}")
        return None

def calculate_sma(data, window):
    try:
        sma = data['Close'].rolling(window=window).mean()
        if sma.isnull().all():
            logging.error(f"SMA calculation failed for window {window}")
            return None
        return sma
    except Exception as e:
        logging.error(f"Error calculating SMA: {e}")
        return None

def preprocess_data(data):
    # Handle missing values and outliers if any
    data = data.dropna()
    if data.empty:
        logging.error("Data is empty after preprocessing")
    return data

def predict_stock_price(data, future_days):
    try:
        close_prices = preprocess_data(data['Close'])
        if close_prices.empty:
            logging.error("Close prices data is empty after preprocessing")
            return []

        logging.info(f"Data for prediction after preprocessing: {close_prices.head()}")

        # Fit the Auto ARIMA model
        model = pm.auto_arima(close_prices, seasonal=False, stepwise=True, suppress_warnings=True, error_action="ignore", max_p=5, max_q=5)
        logging.info(f"Fitted ARIMA model: {model.summary()}")

        # Model diagnostics
        residuals = pd.DataFrame(model.resid())
        if not residuals.empty:
            plt.figure(figsize=(10, 5))
            plt.plot(residuals)
            plt.title("Model Residuals")
            plt.show()

            residuals.plot(kind='kde', title="Residuals Density Plot")
            plt.show()

        # Make predictions
        forecast = model.predict(n_periods=future_days)
        logging.info(f"Forecast for next {future_days} days: {forecast}")
        return forecast
    except Exception as e:
        logging.error(f"Error predicting stock price: {e}")
        return []

if __name__ == "__main__":
    ticker = "AAPL"
    data = get_stock_data(ticker)
    if data is not None:
        sma = calculate_sma(data, 20)
        predictions = predict_stock_price(data, 30)  # Predict the next 30 days
        print(data.head())  # Print the first few rows of the dataframe
