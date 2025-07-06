import streamlit as st


st.title("Black Scholes Model")

with st.sidebar:
    current_asset_price = st.number_input(
        label='Current Asset Price',
        min_value=0, value=20
    )
    
    strike_price = st.number_input(
        label='Strike Price',
        min_value=0, value=30
    )
    
    time_to_expiry = st.number_input(
        label='Time to Maturity (Years)',
        min_value=0, value=1
    )
    
    volatility = st.number_input(
        label='Volatility ($\sigma$)',
        min_value=0.0, value=.25
    )

    interest_rate = st.number_input(
        label='Risk free interest Rate $\r$',
        min_value=0.0, value=.10
    )
    
    
    
# ...existing code...

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# --- Heatmap ---
st.subheader("Option Price Heatmap")
S_range = np.linspace(current_asset_price * 0.5, current_asset_price * 1.5, 50)
V_range = np.linspace(0.05, 0.5, 50)
S_grid, V_grid = np.meshgrid(S_range, V_range)
prices = black_scholes_call(S_grid, strike_price, time_to_expiry, interest_rate, V_grid)

fig1, ax1 = plt.subplots()
c = ax1.pcolormesh(S_range, V_range, prices, shading='auto', cmap='viridis')
fig1.colorbar(c, ax=ax1, label='Call Price')
ax1.set_xlabel('Asset Price')
ax1.set_ylabel('Volatility')
st.pyplot(fig1)

# --- Special Plot ---
st.subheader("Special Plot: Option Price vs Asset Price")

fig2, ax2 = plt.subplots()

# Main line (blue)
S_plot = np.linspace(current_asset_price * 0.5, current_asset_price * 1.5, 100)
main_price = black_scholes_call(S_plot, strike_price, time_to_expiry, interest_rate, volatility)
ax2.plot(S_plot, main_price, color='blue', label='Current Volatility')

# Faded lines for other volatilities
fade_vols = [volatility * 0.5, volatility * 0.75, volatility * 1.25, volatility * 1.5]
for v in fade_vols:
    alpha = max(0.2, 1 - abs(v - volatility) / volatility)  # More faded if further from main
    faded_price = black_scholes_call(S_plot, strike_price, time_to_expiry, interest_rate, v)
    ax2.plot(S_plot, faded_price, color='blue', alpha=alpha, linestyle='--')

# Horizontal line for strike price
ax2.axhline(strike_price, color='red', linestyle=':', label='Strike Price')

ax2.set_xlabel('Asset Price')
ax2.set_ylabel('Call Option Price')
ax2.legend()
st.pyplot(fig2)
# ...existing code...
    