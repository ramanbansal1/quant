import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# App Title
st.title("📈 Stock Price Viewer")

# Sidebar Inputs
st.sidebar.header("Customize your query")

# Ticker Input
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL", max_chars=10)

# Time Period Options (for yfinance)
time_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
time_period = st.sidebar.selectbox("Select Time Period", time_periods, index=2)

# Interval Options
intervals = ["1m", "5m", "15m", "30m", "60m", "90m", "1d", "5d", "1wk", "1mo", "3mo"]
interval = st.sidebar.selectbox("Select Interval", intervals, index=6)

# Load Data
@st.cache_data
def load_data(ticker, period, interval):
    data = yf.download(ticker, period=period, interval=interval)
    return data

if ticker:
    try:
        data = load_data(ticker, time_period, interval)
        if data.empty:
            st.warning("No data found. Try a different combination.")
        else:
            st.write(f"### Showing data for: {ticker.upper()} ({time_period}, {interval})")
            st.line_chart(data['Close'])
            st.dataframe(data.tail())
    except Exception as e:
        st.error(f"Failed to load data: {e}")
else:
    st.info("Please enter a stock ticker symbol to begin.")
