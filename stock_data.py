import pandas as pd
import yfinance as yf

def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y")
    return hist

def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()



if __name__ == "__main__":
    ticker = "AAPL"
    data = get_stock_data(ticker)
    sma = calculate_sma(data, 20)

    print(data.head())  # Print the first few rows of the data
    print(sma.head())   # Print the first few rows of the SMA
