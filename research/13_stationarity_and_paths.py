import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller, pacf
import refinitiv.data as rd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tnic_universe import TNICUniverse
from src.models_midterm import MidTermGovernor

def simulate_forward_paths(z_0, mu, theta, sigma, horizon, dt, num_paths):
    """Simulates parallel Jump-Diffusion paths starting from z_0"""
    Z = np.zeros((num_paths, horizon))
    Z[:, 0] = z_0
    dW = np.random.normal(0, np.sqrt(dt), size=(num_paths, horizon - 1))
    
    # Poisson Jumps (5 major events per year)
    jump_prob = 5.0 * dt
    dN = np.random.binomial(1, jump_prob, size=(num_paths, horizon - 1))
    xi = np.random.normal(0, sigma * 3.0, size=(num_paths, horizon - 1))
    
    for t in range(1, horizon):
        Z[:, t] = Z[:, t-1] + theta * (mu - Z[:, t-1]) * dt + sigma * dW[:, t-1] + xi[:, t-1] * dN[:, t-1]
    return Z

def run_path_diagnostics():
    target_ric = "XOM.N" 
    top_n = 4 
    
    universe = TNICUniverse(tnic_data_path="data/raw/tnic2_data.txt", target_year=2021)
    # Using hardcoded GVKEY mapping for the script bypass (Ensure your dict is updated)
    RIC_TO_GVKEY = {"XOM.N": "4503", "CVX.N": "5903", "COP.N": "61971", "OXY.N": "25964", "MPC.N": "294524"}
    GVKEY_TO_RIC = {v: k for k, v in RIC_TO_GVKEY.items()}
    
    print("Fetching topology and executing Kalman Filter...")
    peer_rics, weights = universe.get_top_n_peers(target_ric, RIC_TO_GVKEY, GVKEY_TO_RIC, n_peers=top_n)
    
    rd.open_session()
    try:
        df = rd.get_history(universe=[target_ric] + peer_rics, fields=["TRDPRC_1"], 
                            start="2022-01-01", end="2023-01-01", interval="1D")
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.ffill().bfill().dropna().astype(float)
        
        target_prices = df[target_ric].values
        peer_prices = df[peer_rics].values
        
        synthetic_index = universe.construct_peer_index(weights, peer_prices)
        governor = MidTermGovernor(dt=1/252)
        spread = governor.fit_kalman_cointegration(target_prices, synthetic_index)
        
        # 1. Tumbling Window Parameters (Calibrate on Q1, Predict Q2)
        q1_length = 63
        q1_spread = spread[:q1_length]
        q2_spread = spread[q1_length:q1_length*2]
        
        # Stationarity Check (ADF)
        adf_stat = adfuller(q1_spread)
        p_value = adf_stat[1]
        
        # PACF Check
        pacf_vals = pacf(q1_spread, nlags=15)
        
        # Parameter Extraction
        mu, sigma, half_life = governor.fit_rolling_ou_process(q1_spread)
        theta = np.log(2) / half_life if half_life > 0 else 0.0
        
        print(f"\n--- Q1 Calibration ---")
        print(f"ADF P-Value: {p_value:.4f} (Must be < 0.05 for stationarity)")
        print(f"Mean (mu): {mu:.4f} | Speed (theta): {theta:.2f} | Sigma (MAD): {sigma:.4f}")
        
        # 2. Monte Carlo Forward Simulation
        paths = simulate_forward_paths(z_0=q1_spread[-1], mu=mu, theta=theta, sigma=sigma, 
                                       horizon=q1_length, dt=1/252, num_paths=1000)
        
        # 3. Visualizations
        fig = make_subplots(rows=1, cols=3, subplot_titles=("Q1 Stationarity (ADF)", "PACF (Lag Check)", "Q2 Forward Simulation vs. Realized Path"), column_widths=[0.25, 0.25, 0.5])
        
        # Panel 1: Spread & Mean
        fig.add_trace(go.Scatter(y=q1_spread, name="Q1 Spread"), row=1, col=1)
        fig.add_trace(go.Scatter(y=[mu]*q1_length, line=dict(dash='dash', color='black'), name="Q1 Mean"), row=1, col=1)
        
        # Panel 2: PACF
        fig.add_trace(go.Bar(x=np.arange(15), y=pacf_vals, name="PACF"), row=1, col=2)
        fig.add_trace(go.Scatter(x=[-1, 15], y=[0.05, 0.05], line=dict(dash='dot', color='red'), showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[-1, 15], y=[-0.05, -0.05], line=dict(dash='dot', color='red'), showlegend=False), row=1, col=2)
        
        # Panel 3: Quant Guild Simulation overlay
        time_axis = np.arange(q1_length)
        for i in range(200): # Plot 200 paths
            fig.add_trace(go.Scatter(x=time_axis, y=paths[i, :], mode='lines', line=dict(color='lightgrey', width=1), opacity=0.3, showlegend=False), row=1, col=3)
            
        fig.add_trace(go.Scatter(x=time_axis, y=np.mean(paths, axis=0), line=dict(color='blue', width=2, dash='dash'), name="E[Spread]"), row=1, col=3)
        fig.add_trace(go.Scatter(x=time_axis, y=q2_spread, line=dict(color='red', width=3), name="Realized Q2 Path"), row=1, col=3)
        
        fig.update_layout(title="Ergodic Assumption Testing & Forward Path Analysis", height=600, template="plotly_white")
        fig.show()

    finally:
        rd.close_session()

if __name__ == "__main__":
    run_path_diagnostics()