import streamlit as st
import requests
from serpapi import GoogleSearch
from textblob import TextBlob
from datetime import datetime, timedelta

# Your API key for SERP API
serp_api_key = 'a556dd24b36b68f5166928370678f2079210409e1b7c61a6e9e9e12fdc241b79'


# Function to fetch stock news using SERP API
def get_stock_news(symbol, date_range="1d"):
    params = {
        "q": f"{symbol} stock news",
        "api_key": serp_api_key,
        "tbs": f"cdr:1,cd_min:{(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')}"
        # Filter news from the last day
    }
    search = GoogleSearch(params)
    results = search.get_dict()

    # Get the top news titles and their snippets
    news = []
    for result in results.get('organic_results', []):
        title = result.get('title')
        snippet = result.get('snippet')
        news.append(f"{title}: {snippet}")

    return news


# Function to analyze sentiment of the news articles
def analyze_sentiment(news):
    positive, negative = 0, 0
    for article in news:
        # Perform sentiment analysis on the article snippet
        blob = TextBlob(article)
        sentiment = blob.sentiment.polarity
        if sentiment > 0:
            positive += 1
        elif sentiment < 0:
            negative += 1
    return positive, negative


# Function to get top trending stocks based on sentiment analysis for a given date
def get_trending_stocks(symbols, date_range="1d"):
    trending_up = []
    trending_down = []

    for symbol in symbols:
        news = get_stock_news(symbol, date_range)
        positive, negative = analyze_sentiment(news)

        if positive > negative:
            trending_up.append(symbol)
        else:
            trending_down.append(symbol)

    return trending_up, trending_down


# Function to fetch the top trending stocks (dynamically)
def get_top_trending_stocks():
    # You can use APIs like Yahoo Finance, Google News, or other sources to get trending stocks
    # For now, we can simulate this by fetching news for a set of popular stocks

    # Example: Fetch top trending stocks from a financial news source (for demo, we use static symbols)
    trending_symbols = ["TSLA", "AAPL", "AMZN", "GOOG", "MSFT", "NFLX", "META",
                        "NVDA"]  # This can be dynamically fetched
    return trending_symbols


# Streamlit UI
st.title("Top Trending Stocks Based on News Sentiment")

# Fetch top trending stocks dynamically (could be expanded to use actual API to fetch trending tickers)
stock_symbols = get_top_trending_stocks()

# Fetch trending stocks for the current day
trending_up_today, trending_down_today = get_trending_stocks(stock_symbols, date_range="1d")

# Display trending stocks for today
st.write("### Trending Stocks Today")
st.write("#### Trending Up")
for stock in trending_up_today:
    st.write(f"- {stock}")

st.write("#### Trending Down")
for stock in trending_down_today:
    st.write(f"- {stock}")

# Fetch trending stocks for the previous day
trending_up_yesterday, trending_down_yesterday = get_trending_stocks(stock_symbols, date_range="1d")

# Display trending stocks for the previous day
st.write("### Trending Stocks Yesterday")
st.write("#### Trending Up")
for stock in trending_up_yesterday:
    st.write(f"- {stock}")

st.write("#### Trending Down")
for stock in trending_down_yesterday:
    st.write(f"- {stock}")
