import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller
import refinitiv.data as rd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tnic_universe import TNICUniverse
from src.models_midterm import MidTermGovernor

def run_ergodicity_diagnostic():
    target_ric = "XOM.N" 
    peer_mapping = {"5903": "CVX.N", "61971": "COP.N", "25964": "OXY.N", "294524": "MPC.N"}
    similarity_scores = np.array([0.0629, 0.0417, 0.0171, 0.0108])
    peer_rics = list(peer_mapping.values())
    
    rd.open_session()
    try:
        # 1. Fetch & Align
        df_merged = rd.get_history(universe=[target_ric] + peer_rics, fields=["TRDPRC_1"], 
                                   start="2021-01-01", end="2024-01-01", interval="1D")
        if isinstance(df_merged.columns, pd.MultiIndex):
            df_merged.columns = df_merged.columns.get_level_values(0)
        df_merged = df_merged.ffill().bfill().dropna().astype(float)
        
        target_prices = df_merged[target_ric].values
        peer_prices = df_merged[peer_rics].values
        dates = df_merged.index
        T = len(target_prices)
        
        # 2. Extract Spread via Adaptive Kalman
        universe = TNICUniverse(tnic_data_path="data/raw/tnic2_data.txt", target_year=2021)
        synthetic_peer_index = universe.construct_peer_index(similarity_scores, peer_prices)
        governor = MidTermGovernor(dt=1/252)
        spread = governor.fit_kalman_cointegration(target_prices, synthetic_peer_index)
        
        # 3. Rolling Ergodicity Testing
        window = 126 
        valid_length = T - window
        
        adf_pvalues = np.zeros(valid_length)
        l2_std = np.zeros(valid_length)
        l1_robust_std = np.zeros(valid_length)
        pit_u = np.zeros(valid_length)
        
        for i, t in enumerate(range(window, T)):
            local_spread = spread[t-window:t]
            
            # A. Strict Stationarity Test (ADF)
            try:
                # We use AIC to dynamically find the optimal lag structure of the microstructure noise
                adf_stat = adfuller(local_spread, autolag='AIC')
                adf_pvalues[i] = adf_stat[1]
            except:
                adf_pvalues[i] = 1.0 # Assume non-stationary if collinearity breaks the test
                
            # B. Variance Calibration (L2 vs L1 Norm)
            mu, sigma_l2, half_life = governor.fit_rolling_ou_process(local_spread)
            
            Z_t = local_spread[1:]
            Z_t_minus_1 = local_spread[:-1]
            
            # Reconstruct the regression to extract raw residuals
            X = np.vstack([np.ones(len(Z_t_minus_1)), Z_t_minus_1]).T
            try:
                params = np.linalg.inv(X.T @ X) @ X.T @ Z_t
                residuals = Z_t - (params[0] + params[1] * Z_t_minus_1)
            except:
                residuals = np.zeros_like(Z_t)
                
            # Standard L2 Norm (Euclidean)
            l2_std[i] = np.std(residuals) / np.sqrt(1/252)
            
            # Robust L1 Norm (MAD) projected to asymptotic normality
            mad = np.median(np.abs(residuals - np.median(residuals)))
            l1_robust_std[i] = (1.4826 * mad + 1e-8) / np.sqrt(1/252)
            
            # C. Probability Integral Transform (PIT) for the current step
            # Does the realized spread fall within the expected stationary measure?
            if half_life > 0 and half_life < 252:
                theta = np.log(2) / half_life
                asymptotic_var = (l1_robust_std[i]**2) / (2 * theta)
                expected_std = np.sqrt(asymptotic_var)
                
                # CDF of the realized point under the assumed normal invariant measure
                from scipy.stats import norm
                pit_u[i] = norm.cdf(spread[t], loc=mu, scale=expected_std)
            else:
                pit_u[i] = np.nan # Undefined measure

        

        # 4. Render the Diagnostic Dashboard
        valid_dates = dates[window:]
        
        fig = make_subplots(rows=3, cols=1, shared_xaxes=False, 
                            vertical_spacing=0.08,
                            row_heights=[0.3, 0.4, 0.3],
                            subplot_titles=("Rolling ADF Stationarity P-Value (Is there an invariant measure?)", 
                                            "Variance Calibration: L2 (Standard) vs L1 (MAD)",
                                            "Calibrated PIT Histogram (Ergodicity Proof)"))

        # Panel 1: ADF P-Values
        fig.add_trace(go.Scatter(x=valid_dates, y=adf_pvalues, mode='lines', name='ADF P-Value', line=dict(color='purple')), row=1, col=1)
        fig.add_trace(go.Scatter(x=valid_dates, y=np.full(valid_length, 0.05), mode='lines', name='0.05 Significance Gate', line=dict(color='red', dash='dash')), row=1, col=1)
        
        # Panel 2: Variance
        fig.add_trace(go.Scatter(x=valid_dates, y=l2_std, mode='lines', name='L2 Std Dev (Jump Contaminated)', line=dict(color='red', width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=valid_dates, y=l1_robust_std, mode='lines', name='L1 Robust MAD (Structural)', line=dict(color='navy', width=2)), row=2, col=1)

        # Panel 3: PIT Histogram
        valid_pit = pit_u[~np.isnan(pit_u)]
        fig.add_trace(go.Histogram(x=valid_pit, nbinsx=20, histnorm='probability density', marker_color='steelblue', name='PIT'), row=3, col=1)
        fig.add_trace(go.Scatter(x=[0, 1], y=[1, 1], mode='lines', line=dict(color='red', dash='dash', width=2), name='Theoretical Uniform'), row=3, col=1)

        fig.update_layout(title_text="Strict Measure-Theoretic Ergodicity Diagnostics", hovermode="x unified", height=900, template="plotly_white")
        fig.show()

    finally:
        rd.close_session()

if __name__ == "__main__":
    run_ergodicity_diagnostic()