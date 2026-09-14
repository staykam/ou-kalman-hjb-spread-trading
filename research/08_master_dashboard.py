import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import refinitiv.data as rd
from joblib import Parallel, delayed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tnic_universe import TNICUniverse
from src.models_midterm import MidTermGovernor

def generate_parallel_universes(batch_size: int, z_0: float, mu: float, theta: float, 
                                sigma: float, horizon_steps: int, dt: float) -> np.ndarray:
    Z = np.zeros((batch_size, horizon_steps))
    Z[:, 0] = z_0
    dW = np.random.normal(0, np.sqrt(dt), size=(batch_size, horizon_steps - 1))
    for t in range(1, horizon_steps):
        drift = theta * (mu - Z[:, t-1]) * dt
        diffusion = sigma * dW[:, t-1]
        Z[:, t] = Z[:, t-1] + drift + diffusion
    return Z

def run_master_dashboard():
    target_ric = "XOM.N" 
    peer_mapping = {"5903": "CVX.N", "61971": "COP.N", "25964": "OXY.N", "294524": "MPC.N"}
    similarity_scores = np.array([0.0629, 0.0417, 0.0171, 0.0108])
    peer_rics = list(peer_mapping.values())
    
    rd.open_session()
    try:
        # --- 1. Fetch & Align ---
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
        
        # --- 2. Kalman Governor ---
        universe = TNICUniverse(tnic_data_path="data/raw/tnic2_data.txt", target_year=2021)
        synthetic_peer_index = universe.construct_peer_index(similarity_scores, peer_prices)
        
        governor = MidTermGovernor(dt=1/252)
        spread = governor.fit_kalman_cointegration(target_prices, synthetic_peer_index)
        
        # --- 3. Execution Engine & P&L ---
        signals = np.zeros(T)
        pnl = np.zeros(T)
        dynamic_mu = np.zeros(T)
        dynamic_std = np.zeros(T)
        current_position = 0 
        window = 126 
        
        bps_slippage = 0.0005        
        daily_borrow_rate = 0.02 / 252.0
        
        for t in range(window, T):
            local_spread = spread[t-window:t]
            mu, sigma, half_life = governor.fit_rolling_ou_process(local_spread)
            is_stationary = (half_life > 0) and (half_life < 252)
            
            if is_stationary:
                theta_val = np.log(2) / half_life
                asymptotic_var = (sigma**2) / (2 * theta_val)
                std_dev = np.sqrt(asymptotic_var)
                z_score = (spread[t] - mu) / std_dev
            else:
                z_score, std_dev = 0.0, 0.0
                
            dynamic_mu[t], dynamic_std[t] = mu, std_dev
            normalized_innovation = np.abs(governor.e_t[t]) / np.sqrt(governor.S_t[t])
            is_model_confident = normalized_innovation < 3.0 
            
            if current_position != 0:
                delta_target = target_prices[t] - target_prices[t-1]
                delta_peer = synthetic_peer_index[t] - synthetic_peer_index[t-1]
                daily_profit = current_position * (delta_target - governor.beta_t[t-1] * delta_peer)
                borrow_cost = (governor.beta_t[t-1] * synthetic_peer_index[t-1] if current_position == 1 else target_prices[t-1]) * daily_borrow_rate
                pnl[t] = pnl[t-1] + daily_profit - borrow_cost
            else:
                pnl[t] = pnl[t-1]

            prev_position = current_position
            if not is_stationary:
                current_position = 0 
            else:
                if current_position == 0:
                    if z_score < -1.5 and is_model_confident: current_position = 1
                    elif z_score > 1.5 and is_model_confident: current_position = -1
                elif (current_position == 1 and z_score >= 0) or (current_position == -1 and z_score <= 0):
                    current_position = 0
                    
            if current_position != prev_position:
                gross_exposure = target_prices[t] + (governor.beta_t[t] * synthetic_peer_index[t])
                pnl[t] -= abs(current_position - prev_position) * gross_exposure * bps_slippage
            signals[t] = current_position

        # --- 4. Ergodicity Simulation (joblib) ---
        print("Simulating 16,000 universes for Ergodicity test...")
        valid_spread = spread[window:]
        global_mu, global_sigma, global_hl = governor.fit_rolling_ou_process(valid_spread)
        global_theta = np.log(2) / global_hl if global_hl > 0 else 0.0
        
        T_steps = len(valid_spread)
        M_paths = 16000
        n_batches = 16
        
        results = Parallel(n_jobs=-1)(
            delayed(generate_parallel_universes)(M_paths // n_batches, valid_spread[0], global_mu, global_theta, global_sigma, T_steps, 1/252.0) 
            for _ in range(n_batches)
        )
        simulated_ensemble = np.vstack(results)
        
        # Calculate percentiles for the density bands
        p5 = np.percentile(simulated_ensemble, 5, axis=0)
        p25 = np.percentile(simulated_ensemble, 25, axis=0)
        p50 = np.percentile(simulated_ensemble, 50, axis=0)
        p75 = np.percentile(simulated_ensemble, 75, axis=0)
        p95 = np.percentile(simulated_ensemble, 95, axis=0)
        
        # Calculate PIT
        U_t = np.array([np.mean(simulated_ensemble[:, t] <= valid_spread[t]) for t in range(T_steps)])
        terminal_dist = simulated_ensemble[:, -1]

        # --- 5. Interactive Plotly Dashboard ---
        print("Rendering WebGL Dashboard...")
        valid_dates = dates[window:]
        long_idx = np.where((signals == 1) & (np.roll(signals, 1) == 0))[0] 
        short_idx = np.where((signals == -1) & (np.roll(signals, 1) == 0))[0] 
        exit_idx = np.where((signals == 0) & (np.roll(signals, 1) != 0))[0]

        fig = make_subplots(
            rows=6, cols=2, 
            shared_xaxes=False,
            column_widths=[0.85, 0.15],
            row_heights=[0.2, 0.2, 0.15, 0.15, 0.2, 0.1],
            vertical_spacing=0.04, horizontal_spacing=0.02,
            specs=[
                [{"colspan": 2}, None], # 1. Price
                [{"colspan": 2}, None], # 2. Spread
                [{"colspan": 2}, None], # 3. Innovation
                [{"colspan": 2}, None], # 4. P&L
                [{"type": "xy"}, {"type": "xy"}], # 5. Simulation Density | Terminal Dist
                [{"colspan": 2}, None]  # 6. PIT
            ],
            subplot_titles=("Target Price & Executions", "Rolling Spread", "Kalman Innovation", "Cumulative P&L", 
                            "Probability Measure Evolution (50% & 90% Bands)", "Terminal", "PIT Ergodicity Test")
        )

        # 1. Price
        fig.add_trace(go.Scatter(x=dates, y=target_prices, mode='lines', name='Price', line=dict(color='black', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates[long_idx], y=target_prices[long_idx], mode='markers', marker=dict(symbol='triangle-up', size=10, color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates[short_idx], y=target_prices[short_idx], mode='markers', marker=dict(symbol='triangle-down', size=10, color='red')), row=1, col=1)

        # 2. Spread
        fig.add_trace(go.Scatter(x=valid_dates, y=spread[window:], mode='lines', name='Spread', line=dict(color='purple', width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=valid_dates, y=dynamic_mu[window:] + 1.5*dynamic_std[window:], mode='lines', line=dict(color='red', dash='dot', width=1)), row=2, col=1)
        fig.add_trace(go.Scatter(x=valid_dates, y=dynamic_mu[window:] - 1.5*dynamic_std[window:], mode='lines', line=dict(color='green', dash='dot', width=1)), row=2, col=1)

        # 3. Innovation
        kalman_std = np.sqrt(governor.S_t[window:])
        fig.add_trace(go.Scatter(x=valid_dates, y=governor.e_t[window:], mode='lines', name='Innovation', line=dict(color='darkorange', width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=valid_dates, y=3.0*kalman_std, mode='lines', line=dict(color='grey', dash='dot', width=1)), row=3, col=1)

        # 4. P&L
        fig.add_trace(go.Scatter(x=valid_dates, y=pnl[window:], mode='lines', name='P&L', fill='tozeroy', line=dict(color='dodgerblue')), row=4, col=1)

        # 5a. Probability Density Evolution
        fig.add_trace(go.Scatter(x=valid_dates, y=p95, mode='lines', line=dict(width=0), showlegend=False), row=5, col=1)
        fig.add_trace(go.Scatter(x=valid_dates, y=p5, mode='lines', fill='tonexty', fillcolor='rgba(173, 216, 230, 0.4)', line=dict(width=0), name='90% Conf'), row=5, col=1)
        fig.add_trace(go.Scatter(x=valid_dates, y=p75, mode='lines', line=dict(width=0), showlegend=False), row=5, col=1)
        fig.add_trace(go.Scatter(x=valid_dates, y=p25, mode='lines', fill='tonexty', fillcolor='rgba(70, 130, 180, 0.6)', line=dict(width=0), name='50% Conf'), row=5, col=1)
        fig.add_trace(go.Scatter(x=valid_dates, y=p50, mode='lines', line=dict(color='navy', width=1), name='Median Expected'), row=5, col=1)
        fig.add_trace(go.Scatter(x=valid_dates, y=valid_spread, mode='lines', line=dict(color='black', width=1.5), name='Realized Path'), row=5, col=1)

        # 5b. Terminal Distribution
        fig.add_trace(go.Histogram(y=terminal_dist, histnorm='probability density', marker_color='navy', orientation='h', showlegend=False), row=5, col=2)

        # 6. PIT
        fig.add_trace(go.Histogram(x=U_t, nbinsx=20, histnorm='probability density', marker_color='steelblue', name='PIT'), row=6, col=1)
        fig.add_trace(go.Scatter(x=[0, 1], y=[1, 1], mode='lines', line=dict(color='red', dash='dash', width=2), name='Uniform'), row=6, col=1)

        fig.update_layout(height=1400, hovermode="x unified", template="plotly_white", showlegend=False)
        
        # Link x-axes for the time-series panels
        for i in range(1, 6):
            fig.update_xaxes(matches='x', row=i, col=1)
            
        fig.show()

    finally:
        rd.close_session()

if __name__ == "__main__":
    run_master_dashboard()