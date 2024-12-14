import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# Function to predict future stock prices using linear regression
def predict_future_prices(data, days_to_predict):
    data = data.reset_index()
    data['Day'] = np.arange(len(data))

    # Train the model
    X = data[['Day']]
    y = data['Close']
    model = LinearRegression()
    model.fit(X, y)

    # Predict future prices
    future_days = np.arange(len(data), len(data) + days_to_predict).reshape(-1, 1)
    future_prices = model.predict(future_days)

    return future_prices

# Function to calculate buy/sell/hold signals based on moving averages
def calculate_signals(data, short_window, long_window):
    data['Short_MA'] = data['Close'].rolling(window=short_window).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window).mean()

    data['Signal'] = 'Hold'
    data.loc[data['Short_MA'] > data['Long_MA'], 'Signal'] = 'Buy'
    data.loc[data['Short_MA'] < data['Long_MA'], 'Signal'] = 'Sell'

    return data

# Main function to display the app
def show():
    st.title("Stock Buy/Sell/Hold Predictor and Future Price Prediction")

    # Main input fields
    stock_symbol = st.text_input("Stock Symbol (e.g., AAPL, TSLA)", value="AAPL")
    end_date = st.date_input("End Date", value=pd.to_datetime("2024-12-13"))
    start_date = end_date - pd.DateOffset(years=1)  # Auto-fill start date as 1 year before end date
    st.text(f"Start Date: {start_date.date()}")  # Display start date as read-only
    short_window = st.slider("Short Moving Average Window", min_value=5, max_value=50, value=20)
    long_window = st.slider("Long Moving Average Window", min_value=20, max_value=200, value=50)
    days_to_predict = st.slider("Days to Predict", min_value=1, max_value=30, value=5)

    # Fetch stock data
    st.header(f"Stock Data for {stock_symbol}")
    try:
        data = yf.download(stock_symbol, start=start_date, end=end_date)
        if data.empty:
            st.error("No data found for the selected stock and date range.")
        else:
            # Calculate signals
            data = calculate_signals(data, short_window, long_window)

            # Fetch current price and calculate percent change
            try:
                current_data = yf.Ticker(stock_symbol).history(period="1d", interval="1m")
                if not current_data.empty:
                    current_price = current_data['Close'].iloc[-1]
                    previous_close = current_data['Close'].iloc[0]
                    percent_change = ((current_price - previous_close) / previous_close) * 100
                else:
                    current_price = None
                    percent_change = None

                st.metric(
                    label=f"Current Price of {stock_symbol}",
                    value=f"${current_price:.2f}" if isinstance(current_price, (int, float)) else "N/A",
                    delta=f"{percent_change:.2f}%" if isinstance(percent_change, (int, float)) else "N/A"
                )

            except Exception as e:
                st.metric(
                    label=f"Current Price of {stock_symbol}",
                    value="N/A",
                    delta="N/A"
                )

            # Plot the data
            st.subheader("Stock Price and Moving Averages")
            plt.figure(figsize=(12, 6))
            plt.plot(data['Close'], label="Close Price", alpha=0.5)
            plt.plot(data['Short_MA'], label=f"{short_window}-Day MA", alpha=0.75)
            plt.plot(data['Long_MA'], label=f"{long_window}-Day MA", alpha=0.75)
            plt.title(f"{stock_symbol} Price and Moving Averages")
            plt.legend()
            st.pyplot(plt)

            # Show signals
            st.subheader("Signals")
            st.write(data[['Close', 'Short_MA', 'Long_MA', 'Signal']].tail(10))

            # Predict future prices
            st.subheader("Future Price Prediction")
            future_prices = predict_future_prices(data[['Close']], days_to_predict)
            st.write(f"Predicted prices for the next {days_to_predict} days:")
            st.write(future_prices)

            # Plot future predictions
            plt.figure(figsize=(12, 6))
            plt.plot(data['Close'], label="Historical Prices", alpha=0.5)
            future_days = np.arange(len(data), len(data) + days_to_predict)
            plt.plot(future_days, future_prices, label="Predicted Prices", linestyle="--", color="red")
            plt.title(f"{stock_symbol} Price Prediction")
            plt.legend()
            st.pyplot(plt)

            # Current day signal prediction
            current_signal = data['Signal'].iloc[-1]
            st.subheader(f"Prediction for Today ({data.index[-1].date()})")
            st.write(f"Signal: {current_signal}")

            # Next day prediction based on linear regression
            if len(future_prices) > 0:
                next_day_price = future_prices[0]  # Get the first predicted price
                st.subheader(f"Prediction for Next Day")
                st.write(f"Predicted Price: ${next_day_price.item():.2f}")

                if len(data) > 1 and len(future_prices) > 1:
                    # Get the last two actual prices as a NumPy array
                    actual_prices = data['Close'].iloc[-2:].values

                    # Get the first two predicted prices as a NumPy array
                    predicted_prices = future_prices[:2]

                    # Calculate the absolute error for each day
                    prediction_error = np.abs(actual_prices - predicted_prices)

                    # Calculate the Mean Absolute Error (MAE)
                    mae = np.mean(prediction_error)

                    # Calculate the Mean Absolute Percentage Error (MAPE)
                    percentage_error = (prediction_error / actual_prices) * 100
                    mape = np.mean(percentage_error)

                    # Calculate Accuracy
                    accuracy = 100 - mape

                    st.subheader("Prediction Accuracy")
                    st.write(f"Actual Prices: {actual_prices}")
                    st.write(f"Predicted Prices: {predicted_prices}")
                    st.write(f"Prediction Error for Last 2 Days: {prediction_error}")
                    st.write(f"Mean Absolute Error (MAE): ${mae:.2f}")
                    st.write(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
                    st.write(f"Prediction Accuracy: {accuracy:.2f}%")

            else:
                st.write("No future predictions available.")

    except Exception as e:
        st.error(f"An error occurred: {e}")


