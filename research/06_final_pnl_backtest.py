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

def run_pnl_backtest():
    target_ric = "XOM.N" 
    peer_mapping = {"5903": "CVX.N", "61971": "COP.N", "25964": "OXY.N", "294524": "MPC.N"}
    similarity_scores = np.array([0.0629, 0.0417, 0.0171, 0.0108])
    peer_rics = list(peer_mapping.values())
    
    rd.open_session()
    try:
        # 1. Fetch and Align
        all_rics = [target_ric] + peer_rics
        df_raw = rd.get_history(universe=all_rics, fields=["TRDPRC_1"], 
                                start="2021-01-01", end="2024-01-01", interval="1D")
        
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)

        with pd.option_context('future.no_silent_downcasting', True):
            df_merged = df_raw.ffill().bfill().infer_objects(copy=False)
            
        df_merged = df_merged.astype(float).dropna()
        target_prices = df_merged[target_ric].values
        peer_prices = df_merged[peer_rics].values
        dates = df_merged.index
        T = len(target_prices)
        
        # 2. Kalman Governor
        universe = TNICUniverse(tnic_data_path="data/raw/tnic2_data.txt", target_year=2021)
        synthetic_peer_index = universe.construct_peer_index(similarity_scores, peer_prices)
        
        governor = MidTermGovernor(dt=1/252)
        spread = governor.fit_kalman_cointegration(target_prices, synthetic_peer_index)
        
        # 3. Rolling Execution Engine & P&L
        signals = np.zeros(T)
        pnl = np.zeros(T)
        dynamic_mu = np.zeros(T)
        dynamic_std = np.zeros(T)
        current_position = 0 
        
        window = 126 
        
        # Market Friction Parameters
        bps_slippage = 0.0005        # 5 bps physical spread crossing cost
        annual_borrow_rate = 0.02    # 2% annual fee to short shares
        daily_borrow_rate = annual_borrow_rate / 252.0
        
        for t in range(window, T):
            
            # A. Extract local physical parameters strictly from F_{t-1} first
            local_spread = spread[t-window:t]
            mu, sigma, half_life = governor.fit_rolling_ou_process(local_spread)
            
            is_stationary = (half_life > 0) and (half_life < 252)
            
            if is_stationary:
                theta = np.log(2) / half_life
                asymptotic_var = (sigma**2) / (2 * theta)
                std_dev = np.sqrt(asymptotic_var)
                z_score = (spread[t] - mu) / std_dev
            else:
                z_score = 0.0 
                std_dev = 0.0
                
            dynamic_mu[t] = mu
            dynamic_std[t] = std_dev
            
            # The Innovation Gatekeeper
            normalized_innovation = np.abs(governor.e_t[t]) / np.sqrt(governor.S_t[t])
            is_model_confident = normalized_innovation < 3.0 
            
            # B. P&L Tracker (Including Borrow Costs)
            if current_position != 0:
                delta_target = target_prices[t] - target_prices[t-1]
                delta_peer = synthetic_peer_index[t] - synthetic_peer_index[t-1]
                daily_profit = current_position * (delta_target - governor.beta_t[t-1] * delta_peer)
                
                # Deduct Borrow Fees based on the short leg
                if current_position == 1:
                    # Long Target, Short Peer
                    borrow_cost = (governor.beta_t[t-1] * synthetic_peer_index[t-1]) * daily_borrow_rate
                else:
                    # Short Target, Long Peer
                    borrow_cost = target_prices[t-1] * daily_borrow_rate
                    
                daily_profit -= borrow_cost
                pnl[t] = pnl[t-1] + daily_profit
            else:
                pnl[t] = pnl[t-1]

            # C. Hysteresis Optimal Stopping Logic
            previous_position = current_position
            
            if not is_stationary:
                current_position = 0 
            else:
                if current_position == 0:
                    if z_score < -1.5 and is_model_confident: current_position = 1
                    elif z_score > 1.5 and is_model_confident: current_position = -1
                elif current_position == 1 and z_score >= 0:
                    current_position = 0
                elif current_position == -1 and z_score <= 0:
                    current_position = 0
                    
            # D. Market Impact Penalty (Slippage applied when state changes)
            if current_position != previous_position:
                # Gross exposure traded: Price of Target + (Beta * Price of Peer)
                gross_exposure = target_prices[t] + (governor.beta_t[t] * synthetic_peer_index[t])
                
                # Moving from Cash to Long is a change of 1.
                # Moving from Long to Short directly (if it happened) would be a change of 2.
                state_change = abs(current_position - previous_position)
                slippage_cost = state_change * gross_exposure * bps_slippage
                
                # Immediately subtract the execution friction from the equity curve
                pnl[t] -= slippage_cost
                
            signals[t] = current_position

        # 4. Interactive Visualization Dashboard
        long_idx = np.where((signals == 1) & (np.roll(signals, 1) == 0))[0] 
        short_idx = np.where((signals == -1) & (np.roll(signals, 1) == 0))[0] 
        exit_idx = np.where((signals == 0) & (np.roll(signals, 1) != 0))[0]

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05,
                            row_heights=[0.35, 0.25, 0.2, 0.2],
                            subplot_titles=("Target Price & Executions", 
                                            "Out-of-Sample Rolling Spread (Stationarity Gated)", 
                                            "Kalman Forecast Error (Innovation & Uncertainty)",
                                            "Cumulative P&L (Net of 5bps Slippage & 2% Borrow Cost)"))

        # Panel 1: Price
        fig.add_trace(go.Scatter(x=dates, y=target_prices, mode='lines', name='Target Price', line=dict(color='black', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates[long_idx], y=target_prices[long_idx], mode='markers', name='Enter LONG', marker=dict(symbol='triangle-up', size=12, color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates[short_idx], y=target_prices[short_idx], mode='markers', name='Enter SHORT', marker=dict(symbol='triangle-down', size=12, color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates[exit_idx], y=target_prices[exit_idx], mode='markers', name='EXIT', marker=dict(symbol='x', size=8, color='blue')), row=1, col=1)

        # Panel 2: Rolling Spread
        valid_dates = dates[window:]
        fig.add_trace(go.Scatter(x=valid_dates, y=spread[window:], mode='lines', name='Spread (Z_t)', line=dict(color='purple', width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=valid_dates, y=dynamic_mu[window:], mode='lines', name='Rolling Eq. (\mu)', line=dict(color='black', dash='dash', width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=valid_dates, y=dynamic_mu[window:] + 1.5*dynamic_std[window:], mode='lines', name='+1.5\sigma', line=dict(color='red', dash='dot', width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=valid_dates, y=dynamic_mu[window:] - 1.5*dynamic_std[window:], mode='lines', name='-1.5\sigma', line=dict(color='green', dash='dot', width=1)), row=2, col=1)

        # Panel 3: Kalman Innovation
        kalman_error = governor.e_t[window:]
        kalman_std = np.sqrt(governor.S_t[window:])
        fig.add_trace(go.Scatter(x=valid_dates, y=kalman_error, mode='lines', name='Innovation (e_t)', line=dict(color='darkorange', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=valid_dates, y=3.0*kalman_std, mode='lines', name='+3.0\sigma (System)', line=dict(color='grey', dash='dot', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=valid_dates, y=-3.0*kalman_std, mode='lines', name='-3.0\sigma (System)', line=dict(color='grey', dash='dot', width=1)), row=3, col=1)

        # Panel 4: P&L
        fig.add_trace(go.Scatter(x=valid_dates, y=pnl[window:], mode='lines', name='Cumulative P&L', line=dict(color='dodgerblue', width=2), fill='tozeroy'), row=4, col=1)

        fig.update_layout(title_text="Structural Alpha Backtest (Stationarity Gated & Friction Applied)", hovermode="x unified", height=1000, template="plotly_white")
        fig.show()

    finally:
        rd.close_session()

if __name__ == "__main__":
    run_pnl_backtest()