import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import refinitiv.data as rd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tnic_universe import TNICUniverse
from src.models_midterm import MidTermGovernor

def run_dynamic_production_backtest():
    # --- DYNAMIC TOPOLOGY CONFIGURATION ---
    target_ric = "NVDA.O"  # Test different regimes: NVDA.O (Tech), JPM.N (Banks), XOM.N (Energy)
    top_n = 10             
    
    universe = TNICUniverse(tnic_data_path="data/raw/tnic2_data.txt", target_year=2021)
    peer_rics, similarity_weights = universe.get_top_n_peers(target_ric, n_peers=top_n)
    
    rd.open_session()
    try:
        # 1. Fetch & Align Data
        df_merged = rd.get_history(universe=[target_ric] + peer_rics, fields=["TRDPRC_1"], 
                                   start="2021-01-01", end="2024-01-01", interval="1D")
        
        if isinstance(df_merged.columns, pd.MultiIndex):
            df_merged.columns = df_merged.columns.get_level_values(0)
            
        df_merged = df_merged.ffill().bfill().dropna().astype(float)
        
        target_prices = df_merged[target_ric].values
        peer_prices = df_merged[peer_rics].values
        dates = df_merged.index
        T = len(target_prices) # FIX: Extract time dimension
        
        # 2. Extract Spread via Adaptive Kalman
        # FIX: Cleaned up duplicate calls and ghost variables
        synthetic_peer_index = universe.construct_peer_index(similarity_weights, peer_prices)
        governor = MidTermGovernor(dt=1/252)
        spread = governor.fit_kalman_cointegration(target_prices, synthetic_peer_index)
        
        # 3. Execution & Performance Engine
        signals = np.zeros(T)
        pnl = np.zeros(T)
        daily_returns = np.zeros(T)
        dynamic_mu = np.zeros(T)
        dynamic_std = np.zeros(T)
        
        current_pos = 0 
        window = 63 #Calibrated to the one 10Q corporate earnigs reporting cycle 
        
        # Physical Friction Constraints
        slippage = 0.0005 # 5 bps
        borrow_fee = 0.02 / 252.0 # 2% Annualized
        
        for t in range(window, T):
            # A. Robust Local Calibration (MAD)
            local_spread = spread[t-window:t]
            mu, sigma, half_life = governor.fit_rolling_ou_process(local_spread)
            
            is_stationary = (0 < half_life < 252)
            if is_stationary:
                theta = np.log(2) / half_life
                robust_std = np.sqrt((sigma**2) / (2 * theta)) * 0.75 # Calibrated for Ergodicity
                z_score = (spread[t] - mu) / robust_std
            else:
                z_score = 0.0
                robust_std = 0.0
                
            dynamic_mu[t] = mu
            dynamic_std[t] = robust_std
            
            # The Innovation Gatekeeper
            normalized_innovation = np.abs(governor.e_t[t]) / np.sqrt(governor.S_t[t])
            is_model_confident = normalized_innovation < 3.0
            
            # B. P&L with Physical Friction
            if current_pos != 0:
                d_target = target_prices[t] - target_prices[t-1]
                d_peer = synthetic_peer_index[t] - synthetic_peer_index[t-1]
                profit = current_pos * (d_target - governor.beta_t[t-1] * d_peer)
                
                exposure = (governor.beta_t[t-1]*synthetic_peer_index[t-1] if current_pos==1 else target_prices[t-1])
                profit -= exposure * borrow_fee
                pnl[t] = pnl[t-1] + profit
                daily_returns[t] = profit / (target_prices[t-1] + abs(governor.beta_t[t-1]*synthetic_peer_index[t-1]))
            else:
                pnl[t] = pnl[t-1]

            # C. Hysteresis Optimal Stopping
            prev_pos = current_pos
            if not is_stationary:
                current_pos = 0
            else:
                if current_pos == 0 and is_model_confident:
                    if z_score < -1.5: current_pos = 1
                    elif z_score > 1.5: current_pos = -1
                elif (current_pos == 1 and z_score >= 0) or (current_pos == -1 and z_score <= 0):
                    current_pos = 0
            
            # Apply execution slippage immediately upon state change
            if current_pos != prev_pos:
                cost = abs(current_pos - prev_pos) * (target_prices[t] + abs(governor.beta_t[t]*synthetic_peer_index[t])) * slippage
                pnl[t] -= cost
                
            signals[t] = current_pos

        # 4. Performance Metrics
        active_returns = daily_returns[daily_returns != 0]
        sharpe = (np.mean(active_returns) / np.std(active_returns)) * np.sqrt(252) if len(active_returns)>0 else 0
        print(f"\n--- Strategy Report for {target_ric} ---")
        print(f"Final Physical P&L: ${pnl[-1]:.2f}")
        print(f"Annualized Sharpe Ratio: {sharpe:.2f}")

        # 5. Interactive Visual Diagnostics
        long_idx = np.where((signals == 1) & (np.roll(signals, 1) == 0))[0] 
        short_idx = np.where((signals == -1) & (np.roll(signals, 1) == 0))[0] 
        exit_idx = np.where((signals == 0) & (np.roll(signals, 1) != 0))[0]
        valid_dates = dates[window:]

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05,
                            row_heights=[0.35, 0.25, 0.2, 0.2],
                            subplot_titles=(f"{target_ric} Price & Market Executions", 
                                            "MAD-Calibrated Spread Dynamics", 
                                            "Kalman Forecast Error (Innovation)",
                                            "Cumulative Physical P&L"))

        # Panel 1: Price & Executions
        fig.add_trace(go.Scatter(x=dates, y=target_prices, mode='lines', name='Target Price', line=dict(color='black', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates[long_idx], y=target_prices[long_idx], mode='markers', name='Enter LONG', marker=dict(symbol='triangle-up', size=12, color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates[short_idx], y=target_prices[short_idx], mode='markers', name='Enter SHORT', marker=dict(symbol='triangle-down', size=12, color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates[exit_idx], y=target_prices[exit_idx], mode='markers', name='EXIT', marker=dict(symbol='x', size=8, color='blue')), row=1, col=1)

        # Panel 2: Spread & MAD Bounds
        fig.add_trace(go.Scatter(x=valid_dates, y=spread[window:], mode='lines', name='Spread (Z_t)', line=dict(color='purple', width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=valid_dates, y=dynamic_mu[window:], mode='lines', name='Equilibrium (\mu)', line=dict(color='black', dash='dash', width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=valid_dates, y=dynamic_mu[window:] + 1.5*dynamic_std[window:], mode='lines', name='+1.5\sigma', line=dict(color='red', dash='dot', width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=valid_dates, y=dynamic_mu[window:] - 1.5*dynamic_std[window:], mode='lines', name='-1.5\sigma', line=dict(color='green', dash='dot', width=1)), row=2, col=1)

        # Panel 3: Innovation (Forecast Error)
        kalman_std = np.sqrt(governor.S_t[window:])
        fig.add_trace(go.Scatter(x=valid_dates, y=governor.e_t[window:], mode='lines', name='Innovation (e_t)', line=dict(color='darkorange', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=valid_dates, y=3.0*kalman_std, mode='lines', name='+3.0\sigma (Gate)', line=dict(color='grey', dash='dot', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=valid_dates, y=-3.0*kalman_std, mode='lines', name='-3.0\sigma (Gate)', line=dict(color='grey', dash='dot', width=1)), row=3, col=1)

        # Panel 4: P&L
        fig.add_trace(go.Scatter(x=valid_dates, y=pnl[window:], mode='lines', name='Cumulative P&L', line=dict(color='dodgerblue', width=2), fill='tozeroy'), row=4, col=1)

        fig.update_layout(title_text=f"Dynamic Ergodic Alpha Backtest ({target_ric} vs Top {top_n} TNIC Peers)", hovermode="x unified", height=1000, template="plotly_white")
        fig.show()

    finally:
        rd.close_session()

if __name__ == "__main__":
    run_dynamic_production_backtest()