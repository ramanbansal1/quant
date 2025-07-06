import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd # Import pandas to check for Series type

# Set the title of the Streamlit application
st.title("Geometric Brownian Motion Simulation")

# --- Sidebar for User Inputs ---
st.sidebar.header("Stock Data Parameters")
# Input for stock ticker symbol
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL", max_chars=10)

# Dropdown for selecting the time period for historical data download
time_periods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
time_period = st.sidebar.selectbox("Select Historical Data Period", time_periods, index=2)

# Dropdown for selecting the interval for historical data download
intervals = ["1m", "5m", "15m", "30m", "60m", "90m", "1d", "5d", "1wk", "1mo", "3mo"]
interval = st.sidebar.selectbox("Select Historical Data Interval", intervals, index=6)

st.sidebar.header("GBM Simulation Parameters")
# Input for simulation duration in years
T = st.sidebar.slider("Simulation Duration (Years)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
# Input for number of steps (trading days) in the simulation
n_steps = st.sidebar.number_input("Number of Simulation Steps (Days)", min_value=50, max_value=1000, value=252, step=10)
# Input for number of Monte Carlo paths to simulate
n_paths = st.sidebar.number_input("Number of Simulation Paths", min_value=10, max_value=1000, value=100, step=10)

# --- Data Loading and GBM Simulation Logic ---

@st.cache_data
def load_and_simulate_data(ticker_symbol, period, interval, sim_T, sim_n_steps, sim_n_paths):
    """
    Loads historical stock data, calculates log returns, estimates GBM parameters,
    and performs Geometric Brownian Motion simulation.
    """
    try:
        # Load historical stock data
        data = yf.download(ticker_symbol, period=period, interval=interval, auto_adjust=True)

        if data.empty:
            st.warning("No historical data found for the selected parameters. Cannot run simulation.")
            return None, None, None, None, None

        prices = data['Close']
        # Calculate log returns, dropping the first NaN value
        log_returns = np.log(prices / prices.shift(1)).dropna()

        # Estimate mean (mu) and standard deviation (sigma) of log returns
        # These are the parameters for the GBM model
        mu = log_returns.mean()
        sigma = log_returns.std()

        # --- FIX: Ensure mu and sigma are scalar floats ---
        # If mean() or std() return a pandas Series (e.g., if log_returns was a DataFrame
        # or if pandas version behavior changes), extract the scalar value.
        if isinstance(mu, pd.Series):
            mu = mu.iloc[0]
        if isinstance(sigma, pd.Series):
            sigma = sigma.iloc[0]
        # --- END FIX ---

        # Get the last available closing price as the starting point for simulation
        S0 = prices.iloc[-1]
        
        # Calculate time step for simulation
        dt = sim_T / sim_n_steps

        # Initialize an array to store all simulation paths
        # The first row is initialized with the starting price S0
        simulations = np.zeros((sim_n_steps, sim_n_paths))
        simulations[0] = S0

        # Perform the GBM simulation
        for t in range(1, sim_n_steps):
            # Generate random numbers from a standard normal distribution
            Z = np.random.standard_normal(sim_n_paths)
            # Apply the GBM formula to calculate the next price step for all paths
            simulations[t] = simulations[t - 1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        
        return data, simulations, S0, mu, sigma

    except Exception as e:
        st.error(f"Error during data loading or simulation: {e}")
        return None, None, None, None, None

# --- Main Application Logic ---
if ticker:
    historical_data, gbm_simulations, S0_val, mu_val, sigma_val = load_and_simulate_data(
        ticker, time_period, interval, T, n_steps, n_paths
    )

    if historical_data is not None:
        # Display Historical Data Section
        st.write(f"### Historical Data for: {ticker.upper()}")
        # Create a Plotly figure for the historical closing price
        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(x=historical_data.index, y=historical_data['Close'], mode='lines', name='Close Price'))

        fig_hist.update_layout(
            title=f'{ticker.upper()} Historical Close Price',
            xaxis_title='Date',
            yaxis_title='Close Price (USD)',
            hovermode='x unified',
            template="plotly_white",
            margin=dict(l=20, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.write("### Raw Historical Data (Last 5 Rows)")
        st.dataframe(historical_data.tail())

        # Display GBM Simulation Section
        if gbm_simulations is not None:
            st.write(f"### Geometric Brownian Motion Simulation for: {ticker.upper()}")
            st.write(f"**Starting Price (S0):** ${S0_val:.2f}")
            st.write(f"**Estimated Mean Log Return (μ):** {mu_val:.6f}")
            st.write(f"**Estimated Volatility (σ):** {sigma_val:.6f}")

            # Create a Plotly figure for the GBM simulation paths
            fig_gbm = go.Figure()
            for i in range(n_paths):
                fig_gbm.add_trace(go.Scatter(
                    y=gbm_simulations[:, i],
                    mode='lines',
                    name=f'Path {i+1}',
                    opacity=0.5,
                    line=dict(width=1)
                ))

            fig_gbm.update_layout(
                title=f'Simulated GBM Paths for {ticker.upper()} ({n_paths} paths, {T} years)',
                xaxis_title='Time Steps (Days)',
                yaxis_title='Simulated Stock Price (USD)',
                hovermode='x unified',
                template="plotly_white",
                showlegend=False, # Hide legend for too many paths
                margin=dict(l=20, r=20, t=40, b=20),
            )
            st.plotly_chart(fig_gbm, use_container_width=True)
else:
    st.info("Please enter a stock ticker symbol to begin.")
