import streamlit as st
from Tickers import show as Ticker
from dashboard import show as dash
from Stockbot import show as chat
from prediction import show as prd



# Sidebar for navigation
st.sidebar.title("Stock Analysis")
option = st.sidebar.radio("Choose an option", ["Dashboard", "Ticker","Future Prediction", "Stock-bot"])

if option == "Dashboard":
    dash()

elif option == "Ticker":
    st.title("Ticker Functions")
    Ticker()

elif option == "Stock-bot":
    st.title("Stock-bot Functions")
    chat()

elif option == "Future Prediction":
    prd()


