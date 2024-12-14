import requests
import streamlit as st

# Replace with your actual Finnhub API key
FINNHUB_API_KEY = 'cte3169r01qt478kvaa0cte3169r01qt478kvaag'
BASE_URL = 'https://finnhub.io/api/v1'
CONVERSION_API_URL = 'https://api.exchangerate-api.com/v4/latest/USD'  # Example currency conversion API


def get_stock_data(symbol):
    """Fetch stock data from Finnhub."""
    url = f"{BASE_URL}/quote?symbol={symbol}&token={FINNHUB_API_KEY}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad HTTP responses
        data = response.json()
        if 'c' in data and data['c'] != 0:  # Ensure valid price is returned
            return {
                'current_price': data['c'],
                'high_price': data['h'],
                'low_price': data['l'],
                'open_price': data['o'],
                'previous_close': data['pc']
            }
        else:
            st.error("Invalid stock data received from Finnhub API.")
            return None
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None


def get_usd_to_inr():
    """Fetch USD to INR conversion rate."""
    try:
        response = requests.get(CONVERSION_API_URL)
        response.raise_for_status()  # Raise an error for bad HTTP responses
        data = response.json()
        if 'rates' in data and 'INR' in data['rates']:
            return data['rates']['INR']
        else:
            st.error("Invalid conversion rate data received.")
            return None
    except Exception as e:
        st.error(f"Error fetching conversion rate: {e}")
        return None


def main():
    st.title("Stock Market Chatbot")

    symbol = st.text_input("Enter a stock ticker symbol (e.g., AAPL, TSLA):")

    if symbol:
        stock_data = get_stock_data(symbol)
        conversion_rate = get_usd_to_inr() or 83.0  # Use fallback rate if API fails

        if stock_data and conversion_rate:
            st.subheader("Stock Prices in INR:")
            st.write(f"Current Price: ₹{stock_data['current_price'] * conversion_rate:.2f}")
            st.write(f"High Price: ₹{stock_data['high_price'] * conversion_rate:.2f}")
            st.write(f"Low Price: ₹{stock_data['low_price'] * conversion_rate:.2f}")
            st.write(f"Open Price: ₹{stock_data['open_price'] * conversion_rate:.2f}")
            st.write(f"Previous Close Price: ₹{stock_data['previous_close'] * conversion_rate:.2f}")
        else:
            st.write("No data found or an error occurred.")


