import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm
import pandas as pd


# utils
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

st.set_page_config(page_title='Black Scholes Model' ,layout="wide")
st.title("Black Scholes Model")

with st.sidebar:
    S = st.number_input(
        label='Current Asset Price',
        min_value=0, value=20
    )
    
    K = st.number_input(
        label='Strike Price',
        min_value=0, value=30
    )
    
    T = st.number_input(
        label='Time to Maturity (Years)',
        min_value=0, value=1
    )
    
    sigma = st.number_input(
        label='Volatility ($\sigma$)',
        min_value=0.0, value=.25
    )

    r = st.number_input(
        label='Risk free interest Rate $\r$',
        min_value=0.0, value=.10
    )
    
    st.markdown("`Created by:` Raman Bansal")
    st.sidebar.markdown("""
    <a href="https://www.linkedin.com/in/-raman-bansal" target="_blank">
        <img src="https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin&style=for-the-badge" alt="LinkedIn Badge">
    </a>
    """, unsafe_allow_html=True)

call_option = round(black_scholes(S, K, T, r, sigma), 2)
put_option = round(black_scholes(S, K, T, r, sigma, option_type='put'), 2)

## Inputs display
columns = st.columns(5)
inputs = ['Current Stock Price', 'Strike Price', 'Time to Maturity (Years)', 'Risk free interest Rate $\r$', 'Volatility ($\sigma$)']
input_vals = [S, K, T, r, sigma]
for i, col in enumerate(columns):
    col.metric(label=inputs[i], value=input_vals[i])
    

## Results display
st.subheader(":rocket: Results")
col1, col2 = st.columns(2)
with col1:
    st.container(border=True).metric(label='Call option', value=call_option)

with col2:
    st.container(border=True).metric(label='Put option', value=put_option)

st.subheader("Option Price Heatmap")
col1, col2 = st.columns(2)

# Create heatmap data
spot_range = np.linspace(S - 30, S + 30, 50)
vol_range = np.linspace(0.1, 1.0, 50)

call_prices = np.array([[black_scholes(s, K, T, r, v, 'call') for v in vol_range] for s in spot_range])
put_prices = np.array([[black_scholes(s, K, T, r, v, 'put') for v in vol_range] for s in spot_range])
with col1:
    fig_call, ax1 = plt.subplots()
    c1 = ax1.imshow(call_prices, cmap='viridis', aspect='auto', 
                    extent=[vol_range[0], vol_range[-1], spot_range[0], spot_range[-1]],
                    origin='lower')
    fig_call.colorbar(c1, ax=ax1, label="Call Price")
    ax1.set_title("Call Option Heatmap")
    ax1.set_xlabel("Volatility (σ)")
    ax1.set_ylabel("Spot Price (S)")
    ax1.grid(linewidth=2, linestyle='-', alpha=.5, color='black')
    st.pyplot(fig_call)

with col2:
    fig_put, ax2 = plt.subplots()
    c2 = ax2.imshow(put_prices, cmap='plasma', aspect='auto', 
                    extent=[vol_range[0], vol_range[-1], spot_range[0], spot_range[-1]],
                    origin='lower')
    fig_put.colorbar(c2, ax=ax2, label="Put Price")
    ax2.set_title("Put Option Heatmap")
    ax2.set_xlabel("Volatility (σ)")
    ax2.set_ylabel("Spot Price (S)")
    st.pyplot(fig_put)
