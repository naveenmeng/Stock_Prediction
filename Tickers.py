import requests
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
from alpha_vantage.fundamentaldata import FundamentalData
from stocknews import StockNews
import datetime  # Added for date restriction logic
from swot_analysis import show_swot_analysis


# Inject custom CSS for tab font size
st.markdown("""
    <style>
    div[data-testid="stTabs"] button {
        font-size: 18px;  /* Adjust font size as needed */
    }
    </style>
    """, unsafe_allow_html=True)

FINNHUB_API_KEY = 'cte3169r01qt478kvaa0cte3169r01qt478kvaag'
BASE_URL = 'https://finnhub.io/api/v1'
CONVERSION_API_URL = 'https://api.exchangerate-api.com/v4/latest/USD'  # Example currency conversion API
def show():

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

    # Title of the app
    st.title('Stock Board')
    ticker = st.text_input('Ticker (e.g., AAPL, MSFT, GOOG)')
    data_type = st.radio("Data Type", ["Historical", "Intraday"])
    interval = None
    selected_date = None
    start_date = None

    if data_type == "Intraday":
        interval = st.selectbox("Interval", ["1m", "5m", "15m", "30m", "1h", "1d"], index=4)
        # Allow users to pick a single date for intraday data
        today = datetime.date.today()
        selected_date = st.date_input("Select Date", value=today, min_value=today - datetime.timedelta(days=7), max_value=today)
        st.markdown(f"Fetching intraday data for: {selected_date}")

    if data_type == "Historical":
        start_date = st.date_input('Start Date')
        end_date = st.date_input('End Date')



    def fetch_intraday_data(ticker, interval, date):
        try:
            # Calculate start and end time for the selected date
            start_time = datetime.datetime.combine(date, datetime.time.min)  # Start of the day
            end_time = datetime.datetime.combine(date, datetime.time.max)  # End of the day

            # Fetch intraday data for the selected date
            data = yf.download(tickers=ticker, start=start_time, end=end_time, interval=interval)
            if data.empty:
                st.error(f"No intraday data found for ticker '{ticker}' on {date} with interval '{interval}'.")
            return data
        except Exception as e:
            st.error(f"An error occurred while fetching intraday data: {e}")
            return pd.DataFrame()

    # Function to calculate RSI
    def calculate_rsi(data, window=14):
        delta = data['Adj Close'].diff(1)
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    # Function to calculate MACD
    def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
        short_ema = data['Adj Close'].ewm(span=short_window, adjust=False).mean()
        long_ema = data['Adj Close'].ewm(span=long_window, adjust=False).mean()
        macd = short_ema - long_ema
        signal = macd.ewm(span=signal_window, adjust=False).mean()
        return macd, signal

    # Function to calculate Bollinger Bands
    def calculate_bollinger_bands(data, window=20):
        rolling_mean = data['Adj Close'].rolling(window=window).mean()
        rolling_std = data['Adj Close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * 2)
        lower_band = rolling_mean - (rolling_std * 2)
        return rolling_mean, upper_band, lower_band

    # Function to calculate Moving Averages
    def calculate_moving_averages(data, short_window=20, long_window=50):
        short_ma = data['Adj Close'].rolling(window=short_window).mean()
        long_ma = data['Adj Close'].rolling(window=long_window).mean()
        return short_ma, long_ma

    # Function to generate Buy/Sell signals based on Moving Averages
    def generate_signals(data):
        signals = []
        for i in range(1, len(data)):
            if pd.notnull(data['20-day MA'][i]) and pd.notnull(data['50-day MA'][i]):
                if data['20-day MA'][i] > data['50-day MA'][i] and data['20-day MA'][i - 1] <= data['50-day MA'][i - 1]:
                    signals.append("Buy")
                elif data['20-day MA'][i] < data['50-day MA'][i] and data['20-day MA'][i - 1] >= data['50-day MA'][i - 1]:
                    signals.append("Sell")
                else:
                    signals.append("Hold")
            else:
                signals.append("Hold")  # Default if moving averages are not available
        signals.insert(0, "Hold")  # The first data point has no prior comparison
        return signals

    def get_stock_data_from_yfinance(ticker):
        stock = yf.Ticker(ticker)
        stock_data = stock.history(period="1d")  # Get data for the last day
        return stock_data.iloc[0]  # Return the most recent data (first row)


    # Validate user inputs
    if not ticker or not ticker.isalpha():
        st.error("Please enter a valid stock ticker (e.g., AAPL, MSFT).")
    elif data_type == "Intraday" and not selected_date:
        st.error("Please select a valid date for intraday data.")
    elif data_type == "Historical" and start_date >= end_date:
        st.error("Start date must be earlier than end date.")
    else:
        try:
            st.write(f"Fetching data for {ticker}...")
            # Fetch stock data
            if data_type == "Intraday" and interval:
                data = fetch_intraday_data(ticker, interval, selected_date)
            else:
                data = yf.download(ticker, start=start_date, end=end_date)

            if data.empty:
                st.error(f"No data found for ticker '{ticker}' with the given date range. Please try again.")
            else:
                # Add new calculations for technical indicators
                data.reset_index(inplace=True)

                # Adjust column names dynamically for intraday or historical data
                date_column = 'Datetime' if data_type == "Intraday" else 'Date'

                # Ensure the index column exists for visualizations
                if date_column not in data.columns:
                    data[date_column] = data.index

                data['RSI'] = calculate_rsi(data)
                data['MACD'], data['Signal Line'] = calculate_macd(data)
                data['20-day MA'], data['50-day MA'] = calculate_moving_averages(data)
                data['Rolling Mean'], data['Upper Band'], data['Lower Band'] = calculate_bollinger_bands(data)

                x_axis = f"{date_column}:T"

                # Altair: Line Chart of Stock Prices with Bollinger Bands and Moving Averages
                base_chart = alt.Chart(data).encode(x=x_axis)
                try:
                    current_data = yf.Ticker(ticker).history(period="1d", interval="1m")
                    if not current_data.empty:
                        current_price = current_data['Close'].iloc[-1]
                        previous_close = current_data['Close'].iloc[0]
                        percent_change = ((current_price - previous_close) / previous_close) * 100
                    else:
                        current_price = "N/A"
                        percent_change = "N/A"
                except Exception as e:
                    current_price = "N/A"
                    percent_change = "N/A"

                st.metric(label=f"Current Price of {ticker}",
                          value=f"${current_price:.2f}" if isinstance(current_price, (int, float)) else "N/A",
                          delta=f"{percent_change:.2f}%" if isinstance(percent_change, (int, float)) else "N/A")

                price_chart = base_chart.mark_line(color='black').encode(
                    y=alt.Y('Adj Close:Q', title='Price'),
                    tooltip=[x_axis, 'Adj Close:Q']
                ).properties(
                    title=f'{ticker} Adjusted Closing Prices with Bollinger Bands and Moving Averages',
                    width=800,
                    height=400
                ).interactive()

                ma_chart = base_chart.mark_line(color='orange').encode(
                    y='20-day MA:Q',
                    tooltip=[x_axis, '20-day MA:Q']
                )

                long_ma_chart = base_chart.mark_line(color='green').encode(
                    y='50-day MA:Q',
                    tooltip=[x_axis, '50-day MA:Q']
                )

                upper_band_chart = base_chart.mark_line(color='red').encode(
                    y='Upper Band:Q',
                    tooltip=[x_axis, 'Upper Band:Q']
                )

                lower_band_chart = base_chart.mark_line(color='blue').encode(
                    y='Lower Band:Q',
                    tooltip=[x_axis, 'Lower Band:Q']
                )

                # Combine all charts
                combined_chart = (
                    price_chart
                    + ma_chart
                    + long_ma_chart
                    + upper_band_chart
                    + lower_band_chart
                ).resolve_scale(
                    y='shared'
                )
                st.altair_chart(combined_chart)

                # RSI Chart
                rsi_chart = (
                    alt.Chart(data)
                    .mark_line(color='orange')
                    .encode(
                        x=x_axis,
                        y='RSI:Q',
                        tooltip=[x_axis, 'RSI:Q']
                    )
                    .properties(
                        title=f'{ticker} RSI (Relative Strength Index)',
                        width=800,
                        height=300
                    )
                    .interactive()
                )
                st.altair_chart(rsi_chart)

                # MACD Chart
                macd_chart = (
                    alt.Chart(data)
                    .mark_line()
                    .encode(
                        x=x_axis,
                        y='MACD:Q',
                        tooltip=[x_axis, 'MACD:Q']
                    )
                    .properties(
                        title=f'{ticker} MACD (Moving Average Convergence Divergence)',
                        width=800,
                        height=300
                    )
                )
                signal_chart = (
                    alt.Chart(data)
                    .mark_line(color='red')
                    .encode(
                        x=x_axis,
                        y='Signal Line:Q',
                        tooltip=[x_axis, 'Signal Line:Q']
                    )
                )
                st.altair_chart(macd_chart + signal_chart)

                pricing_data, fundamental_data, news, finnhub_tab, SwotAnalysis = st.tabs(
                    ['Pricing data', 'Fundamental data', 'Top 10 news', 'finnhub_tab','Swot Analysis'])

                # Pricing Data Tab
                with pricing_data:
                    st.write('## Price Movements')
                    data['% Change'] = data['Adj Close'].pct_change()
                    data['Signal'] = generate_signals(data)

                    # Display the data with signals
                    st.write(data)

                    # Highlight Buy/Sell signals in a separate section
                    st.write("### Buy/Sell Signals")
                    buy_signals = data[data['Signal'] == "Buy"].reset_index()
                    sell_signals = data[data['Signal'] == "Sell"].reset_index()

                    # Candlestick chart with Volume
                    candlestick = alt.Chart(data).mark_rule().encode(
                        x=alt.X('Date:T', axis=alt.Axis(title='Date')),
                        y='Low:Q',
                        y2='High:Q',
                        color=alt.condition(
                            "datum.Open <= datum.Close", alt.value("green"), alt.value("red")
                        ),
                        tooltip=['Date:T', 'Open:Q', 'High:Q', 'Low:Q', 'Close:Q']
                    )

                    bars = alt.Chart(data).mark_bar(size=8).encode(
                        x='Date:T',
                        y='Open:Q',
                        y2='Close:Q',
                        color=alt.condition(
                            "datum.Open <= datum.Close", alt.value("green"), alt.value("red")
                        ),
                        tooltip=['Date:T', 'Open:Q', 'Close:Q']
                    )

                    # Add volume bars
                    volume_chart = alt.Chart(data).mark_bar(opacity=0.5).encode(
                        x='Date:T',
                        y=alt.Y('Volume:Q', axis=alt.Axis(title='Volume')),
                        color=alt.condition(
                            "datum.Open <= datum.Close", alt.value("green"), alt.value("red")
                        )
                    ).properties(
                        height=100
                    )

                    # Combine candlestick and volume charts
                    combined_chart = alt.vconcat(
                        (candlestick + bars).properties(
                            title=f"{ticker} Candlestick Chart with Volume",
                            width=800,
                            height=400
                        ),
                        volume_chart
                    ).resolve_scale(
                        x='shared'
                    )

                    st.altair_chart(combined_chart, use_container_width=True)

                    # Highlight Buy/Sell signals in a table
                    st.write("### Buy/Sell Signals")

                    # Debugging: Check if 'buy_signals' and 'sell_signals' are being created properly
                    st.write("#### Debugging Signal Data")
                    date_column = 'Datetime' if data_type == "Intraday" else 'Date'  # Handle intraday vs historical data
                    st.write("All Signals:")
                    st.write(data[[date_column, '20-day MA', '50-day MA', 'Signal']])

                    # Filter for Buy and Sell signals
                    buy_signals = data[data['Signal'] == "Buy"].reset_index()
                    sell_signals = data[data['Signal'] == "Sell"].reset_index()

                    # Display Buy Signals
                    st.write("#### Buy Signals")
                    if not buy_signals.empty:
                        st.write(buy_signals[[date_column, '20-day MA', '50-day MA', 'Signal']])
                    else:
                        st.write("No Buy signals generated.")

                    # Display Sell Signals
                    st.write("#### Sell Signals")
                    if not sell_signals.empty:
                        st.write(sell_signals[[date_column, '20-day MA', '50-day MA', 'Signal']])
                    else:
                        st.write("No Sell signals generated.")

                    # Calculate and display annual return, standard deviation, and risk-adjusted return
                    annual_return = data['% Change'].mean() * 252 * 100
                    stdev = np.std(data['% Change']) * np.sqrt(252)
                    risk_adj_return = annual_return / stdev if stdev != 0 else None

                    st.write(f"#### Annual Return: {annual_return:.2f}%")
                    st.write(f"#### Standard Deviation: {stdev * 100:.2f}%")
                    st.write(f"#### Risk-Adjusted Return: {risk_adj_return:.2f} (if applicable)")

                # Fundamental Data Tab
                with fundamental_data:
                    try:
                        # Fetching data using yfinance
                        stock = yf.Ticker(ticker)

                        # Balance Sheet
                        st.subheader("Balance Sheet")
                        balance_sheet = stock.balance_sheet
                        if not balance_sheet.empty:
                            st.write(balance_sheet)
                        else:
                            st.write("Balance sheet data not available.")

                        # Income Statement
                        st.subheader("Income Statement")
                        income_statement = stock.financials
                        if not income_statement.empty:
                            st.write(income_statement)
                        else:
                            st.write("Income statement data not available.")

                        # Cash Flow Statement
                        st.subheader("Cash Flow Statement")
                        cash_flow = stock.cashflow
                        if not cash_flow.empty:
                            st.write(cash_flow)
                        else:
                            st.write("Cash flow statement data not available.")

                    except Exception as e:
                        st.error(f"Error fetching fundamental data: {e}")

                # News Tab
                with news:
                    st.write(f"## News for {ticker}")
                    try:
                        sn = StockNews(ticker, save_news=False)
                        df_news = sn.read_rss()

                        # Display details for top 10 news articles
                        for i in range(min(10, len(df_news))):  # Limit to 10 articles
                            st.subheader(f'News {i + 1}')
                            st.write(f"Published: {df_news['published'][i]}")
                            st.write(f"Title: {df_news['title'][i]}")
                            st.write(f"Summary: {df_news['summary'][i]}")

                            # Display sentiment analysis
                            st.write(f"Title Sentiment: {df_news['sentiment_title'][i]}")
                            st.write(f"Summary Sentiment: {df_news['sentiment_summary'][i]}")
                    except Exception as e:
                        st.error(f"Error fetching news: {e}")

                with finnhub_tab:
                    st.write("## Finnhub Chatbot Insights")
                    st.info("This tab uses Finnhub's API to provide stock-related insights.")

                    # Get stock data for the ticker entered by the user
                    if ticker:
                        stock_data = get_stock_data(ticker)
                        conversion_rate = get_usd_to_inr() or 83.0  # Use fallback rate if API fails

                        if stock_data and conversion_rate:
                            st.subheader(f"Stock Prices for {ticker} in INR:")
                            st.write(f"Current Price: ₹{stock_data['current_price'] * conversion_rate:.2f}")
                            st.write(f"High Price: ₹{stock_data['high_price'] * conversion_rate:.2f}")
                            st.write(f"Low Price: ₹{stock_data['low_price'] * conversion_rate:.2f}")
                            st.write(f"Open Price: ₹{stock_data['open_price'] * conversion_rate:.2f}")
                            st.write(f"Previous Close Price: ₹{stock_data['previous_close'] * conversion_rate:.2f}")

                        else:
                            st.write("No data found or an error occurred.")
                    else:
                        st.write("Please enter a stock ticker to get insights.")

                    # Create a simple interface for the chatbot-like functionality
                    chat_input = st.text_input("You can check for different Tickers:")

                    if chat_input:
                        stock_data = get_stock_data(chat_input)
                        conversion_rate = get_usd_to_inr() or 83.0  # Use fallback rate if API fails

                        if stock_data:
                            # Static response based on the stock data
                            response = f"Here is the stock information for {chat_input}: \n" \
                                       f"- Current Price: ₹{stock_data['current_price'] * conversion_rate:.2f} \n" \
                                       f"- High Price: ₹{stock_data['high_price'] * conversion_rate:.2f} \n" \
                                       f"- Low Price: ₹{stock_data['low_price'] * conversion_rate:.2f} \n" \
                                       f"- Open Price: ₹{stock_data['open_price'] * conversion_rate:.2f} \n" \
                                       f"- Previous Close Price: ₹{stock_data['previous_close'] * conversion_rate:.2f}"

                            st.write(response)
                        else:
                            st.write(f"Sorry, no data available for {chat_input}.")

                with SwotAnalysis:
                    show_swot_analysis(ticker)

        except Exception as e:
            st.error(f"An error occurred: {e}")