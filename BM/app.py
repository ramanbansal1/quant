import streamlit as st
import plotly.graph_objects as go
import numpy as np

st.title("Brownian Motion Simulation")

with st.sidebar:
    st.header("Simulation Settings")
    num_steps = st.slider("Number of Steps", min_value=100, max_value=10000, value=1000, step=100)

    
steps = np.random.normal(loc=0, scale=1, size=num_steps)

path = np.cumsum(steps)

# Create Plotly figure
fig = go.Figure()
fig.add_trace(go.Scatter(
    y=path, mode='lines', 
    name='Brownian Path'
))
fig.update_layout(title="Simulated Brownian Motion", xaxis_title="Step", yaxis_title="Position")

# Display Plotly chart in Streamlit
st.plotly_chart(fig)

st.header("Important Formulation")

st.markdown("""
The Brownian motion can be mathematically described as:
""")