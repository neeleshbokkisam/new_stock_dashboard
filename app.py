import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

from flask import Flask, render_template, request
from apscheduler.schedulers.background import BackgroundScheduler

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from stock_data import get_stock_data, calculate_sma, predict_stock_price
from io import BytesIO
import base64
import logging
import atexit

app = Flask(__name__)
scheduler = BackgroundScheduler()
data_store = {}
def fetch_latest_data():
    tickers = ['AAPL', 'GOOG', 'MSFT']  # Add any other tickers you want to track
    for ticker in tickers:
        data = get_stock_data(ticker)
        data_store[ticker] = data
        logging.info(f"Updated data for {ticker}")

# Schedule the task to run every day at a specific time
scheduler.add_job(func=fetch_latest_data, trigger="interval", hours=24)
scheduler.start()
# Ensure the scheduler is shut down when the app exits
atexit.register(lambda: scheduler.shutdown())
# Set up logging

logging.basicConfig(level=logging.INFO)

def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")
        return hist
    except Exception as e:
        logging.error(f"Error fetching stock data for {ticker}: {e}")
        return None

def calculate_sma(data, window):
    try:
        return data['Close'].rolling(window=window).mean()
    except Exception as e:
        logging.error(f"Error calculating SMA: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    prediction = None
    if request.method == 'POST':
        try:
            ticker = request.form['ticker']
            window = int(request.form['window'])
            future_days = int(request.form['future_days'])

            if ticker in data_store:
                data = data_store[ticker]
            else:
                data = get_stock_data(ticker)
                data_store[ticker] = data

            if data is not None:
                data['SMA'] = calculate_sma(data, window)

                plt.figure(figsize=(10, 5))
                plt.plot(data['Close'], label='Close Price')
                plt.plot(data['SMA'], label='SMA')
                plt.title(f'{ticker} Stock Price and SMA')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()

                img = BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode('utf8')

                predictions = predict_stock_price(data, future_days)
                prediction = {i+1: pred for i, pred in enumerate(predictions)}
            else:
                logging.error(f"No data returned for {ticker}")
        except Exception as e:
            logging.error(f"Error processing request: {e}")

    return render_template('index.html', plot_url=plot_url, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)