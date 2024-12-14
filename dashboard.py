import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import ta


# Fetch stock data based on the ticker, period, and interval
def fetch_stock_data(ticker, period, interval):
    end_date = datetime.now()
    if period == '1wk':
        start_date = end_date - timedelta(days=7)
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    else:
        data = yf.download(ticker, period=period, interval=interval)
    return data


def process_data(data):
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize('UTC')
    data.index = data.index.tz_convert('US/Eastern')
    data.reset_index(inplace=True)
    data.rename(columns={'Date': 'Datetime'}, inplace=True)
    return data


def calculate_metrics(data):
    # Ensure scalar values are extracted
    last_close = float(data['Close'].iloc[-1])  # Last closing price
    prev_close = float(data['Close'].iloc[0])  # First closing price
    change = last_close - prev_close
    pct_change = (change / prev_close) * 100
    high = float(data['High'].max())
    low = float(data['Low'].min())
    volume = int(data['Volume'].sum())
    return last_close, change, pct_change, high, low, volume


# Add simple moving average (SMA) and exponential moving average (EMA) indicators
def add_technical_indicators(data):
    # Ensure that 'Close' is a pandas Series and not a DataFrame
    close_series = data['Close'].squeeze()  # This ensures it is a 1D Series

    # Apply the technical indicators
    data['SMA_20'] = ta.trend.sma_indicator(close_series, window=20, fillna=True)
    data['EMA_20'] = ta.trend.ema_indicator(close_series, window=20, fillna=True)
    return data


def show():
    st.title('Real-Time Stock Dashboard')

    # Ticker input field
    ticker = st.text_input('Enter Ticker (NSE: TICKER.NS, BSE: TICKER.BO)', '')

    if ticker:  # Only show the rest of the inputs if a ticker is provided
        # Time period selection
        time_period = st.selectbox('Select Time Period', ['1d', '1wk', '1mo', '1y', 'max'])

        # Chart type selection
        chart_type = st.selectbox('Select Chart Type', ['Candlestick', 'Line'])

        # Technical indicators selection
        indicators = st.multiselect('Select Technical Indicators', ['SMA 20', 'EMA 20'])

        # Mapping of time periods to data intervals
        interval_mapping = {
            '1d': '1m',
            '1wk': '30m',
            '1mo': '1d',
            '1y': '1wk',
            'max': '1wk'
        }

        # Update the dashboard based on user input
        if st.button('Update'):
            try:
                data = fetch_stock_data(ticker, time_period, interval_mapping[time_period])
                data = process_data(data)
                data = add_technical_indicators(data)

                last_close, change, pct_change, high, low, volume = calculate_metrics(data)

                # Display main metrics (ensure scalars are used for formatting)
                st.metric(label=f"{ticker} Last Price", value=f"{last_close:.2f} USD",
                          delta=f"{change:.2f} ({pct_change:.2f}%)")

                col1, col2, col3 = st.columns(3)
                col1.metric("High", f"{high:.2f} USD")
                col2.metric("Low", f"{low:.2f} USD")
                col3.metric("Volume", f"{volume:,}")

                # Plot the stock price chart
                fig = go.Figure()
                if chart_type == 'Candlestick':
                    fig.add_trace(go.Candlestick(x=data['Datetime'],
                                                 open=data['Open'],
                                                 high=data['High'],
                                                 low=data['Low'],
                                                 close=data['Close']))
                else:
                    fig = px.line(data, x='Datetime', y='Close')

                # Add selected technical indicators to the chart
                for indicator in indicators:
                    if indicator == 'SMA 20':
                        fig.add_trace(go.Scatter(x=data['Datetime'], y=data['SMA_20'], name='SMA 20'))
                    elif indicator == 'EMA 20':
                        fig.add_trace(go.Scatter(x=data['Datetime'], y=data['EMA_20'], name='EMA 20'))

                # Format graph
                fig.update_layout(title=f'{ticker} {time_period.upper()} Chart',
                                  xaxis_title='Time',
                                  yaxis_title='Price (USD)',
                                  height=600)
                st.plotly_chart(fig, use_container_width=True)

                # Display historical data and technical indicators
                st.subheader('Historical Data')
                st.dataframe(data[['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume']])

                st.subheader('Technical Indicators')
                st.dataframe(data[['Datetime', 'SMA_20', 'EMA_20']])
            except Exception as e:
                st.error(f"Error fetching data: {e}")
    else:
        st.warning("Please enter a ticker symbol to proceed.")
