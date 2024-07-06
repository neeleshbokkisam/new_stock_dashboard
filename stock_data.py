import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
print("scikit-learn is installed and imported successfully.")
from sklearn.model_selection import train_test_split
import numpy as np
import pmdarima as pm
import logging

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    return hist

def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

def preprocess_data(data):
    # Handle missing values and outliers if any
    data = data.dropna()
    return data

def predict_stock_price(data, future_days):
    close_prices = preprocess_data(data['Close'])
    logging.info(f"Data for prediction after preprocessing: {close_prices.head()}")

    # Fit the Auto ARIMA model
    model = pm.auto_arima(close_prices, seasonal=False, stepwise=True, suppress_warnings=True, error_action="ignore")
    logging.info(f"Fitted ARIMA model: {model.summary()}")

    # Make predictions
    forecast = model.predict(n_periods=future_days)
    logging.info(f"Forecast for next {future_days} days: {forecast}")
    return forecast

if __name__ == "__main__":
    ticker = "AAPL"
    data = get_stock_data(ticker)
    sma = calculate_sma(data, 20)
    predictions, actual_prices = predict_stock_price(data, 30)  # Predict the next 30 days
    print(data.head())  # Print the first few rows of the dataframe
