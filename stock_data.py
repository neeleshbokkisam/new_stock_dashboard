import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression
print("scikit-learn is installed and imported successfully.")
from sklearn.model_selection import train_test_split
import numpy as np

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    return hist

def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

def predict_stock_price(data, future_days):
    data['Prediction'] = data['Close'].shift(-future_days)

    X = np.array(data[['Close']])
    X = X[:-future_days]

    y = np.array(data['Prediction'])
    y = y[:-future_days]

    model = LinearRegression()
    model.fit(X, y)

    future_X = np.array(data[['Close']])[-future_days:]
    predicted_prices = model.predict(future_X)

    return {i+1: price for i, price in enumerate(predicted_prices)}


if __name__ == "__main__":
    ticker = "AAPL"
    data = get_stock_data(ticker)
    sma = calculate_sma(data, 20)
    predictions, actual_prices = predict_stock_price(data, 30)  # Predict the next 30 days
    print(data.head())  # Print the first few rows of the dataframe
