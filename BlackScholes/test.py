import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import norm

# Black-Scholes Formula
def black_scholes(S, K, T, r, sigma, option_type='call'):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

st.set_page_config(layout="wide")
st.title("📊 Black-Scholes Option Heatmap and Special Plots")

# Sidebar Inputs
S = st.sidebar.slider("Current Asset Price (S)", 50, 150, 100)
K = st.sidebar.slider("Strike Price (K)", 50, 150, 100)
T = st.sidebar.slider("Time to Maturity (Years)", 1, 3, 1) * 0.5
sigma = st.sidebar.slider("Volatility (σ)", 0.1, 1.0, 0.25)
r = st.sidebar.slider("Risk-Free Rate (r)", 0.01, 0.2, 0.05)

st.header("🔥 Options Price Heatmap")

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

# Special Plot
st.header("🔷 Special Plot with Fading Lines and Strike Price")

# Create fading lines
base_sigma = sigma
sigmas = [base_sigma + i * 0.05 for i in range(-3, 4)]
colors = ['rgba(0,0,255,{})'.format(1 - abs(i)/3.5) for i in range(-3, 4)]

x_vals = np.linspace(S - 30, S + 30, 300)

fig = go.Figure()

for i, (s, c) in enumerate(zip(sigmas, colors)):
    y_vals = [black_scholes(x, K, T, r, s, option_type='call') for x in x_vals]
    name = "Main (σ={:.2f})".format(s) if i == 3 else "σ={:.2f}".format(s)
    fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color=c), name=name))

# Strike Price horizontal line
fig.add_trace(go.Scatter(x=[x_vals[0], x_vals[-1]],
                         y=[black_scholes(K, K, T, r, sigma, 'call')] * 2,
                         mode="lines", line=dict(dash='dash', color='red'),
                         name="Strike Price Value"))

fig.update_layout(title="Call Option Price vs Spot Price",
                  xaxis_title="Spot Price",
                  yaxis_title="Option Price",
                  template="plotly_dark",
                  height=600)

st.plotly_chart(fig, use_container_width=True)