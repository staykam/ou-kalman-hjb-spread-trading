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
from src.jb_solver import HJBOptimalStopping

def calibrate_empirical_jumps(spread, dt=1/252):
    """Scans the historical spread to empirically measure jump intensity and frequency."""
    deltas = np.diff(spread)
    mad_sigma = 1.4826 * np.median(np.abs(deltas - np.median(deltas))) / np.sqrt(dt)
    
    # Define a jump as a 3-sigma deviation from the continuous diffusion
    jump_threshold = 3.0 * mad_sigma * np.sqrt(dt)
    jump_indices = np.where(np.abs(deltas) > jump_threshold)[0]
    
    total_years = len(spread) / 252.0
    jumps_per_year = len(jump_indices) / total_years if total_years > 0 else 0
    
    if len(jump_indices) > 0:
        median_jump_size = np.median(np.abs(deltas[jump_indices]))
    else:
        median_jump_size = mad_sigma * 1.5 # Fallback if no jumps found
        
    print(f"[Empirical Calibration] Jumps per Year: {jumps_per_year:.1f} | Median Jump Volatility (\u03BE): {median_jump_size:.4f}")
    return jumps_per_year, median_jump_size

def run_hjb_production():
    target_ric = "XOM.N" 
    top_n = 4 
    
    universe = TNICUniverse(tnic_data_path="data/raw/tnic2_data.txt", target_year=2021)
    RIC_TO_GVKEY = {"XOM.N": "4503", "CVX.N": "5903", "COP.N": "61971", "OXY.N": "25964", "MPC.N": "294524"}
    GVKEY_TO_RIC = {v: k for k, v in RIC_TO_GVKEY.items()}
    
    print("Fetching topology and executing Kalman Filter...")
    raw_peer_rics, raw_weights = universe.get_top_n_peers(target_ric, RIC_TO_GVKEY, GVKEY_TO_RIC, n_peers=top_n)
    
    # Filter target out of peers
    valid_indices = [i for i, ric in enumerate(raw_peer_rics) if ric != target_ric]
    peer_rics = [raw_peer_rics[i] for i in valid_indices]
    weights = np.array([raw_weights[i] for i in valid_indices])
    weights = weights / np.sum(weights) 
    
    rd.open_session()
    try:
        df = rd.get_history(universe=[target_ric] + peer_rics, fields=["TRDPRC_1"], 
                            start="2021-01-01", end="2024-01-01", interval="1D")
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.ffill().bfill().dropna().astype(float)
        
        target_prices = df[target_ric].values
        peer_prices = df[peer_rics].values
        dates = df.index
        T = len(target_prices)
        
        synthetic_index = universe.construct_peer_index(weights, peer_prices)
        governor = MidTermGovernor(dt=1/252)
        spread = governor.fit_kalman_cointegration(target_prices, synthetic_index)
        
        # 1. Empirical Jump Calibration (Look at the whole sample to define the physics)
        jumps_per_year, median_jump_size = calibrate_empirical_jumps(spread)
        
        # 2. Execution Engine Initialization
        hjb_solver = HJBOptimalStopping(z_min=-10.0, z_max=10.0, grid_points=1000)
        pnl = np.zeros(T)
        signals = np.zeros(T)
        hjb_upper = np.zeros(T)
        hjb_lower = np.zeros(T)
        
        current_pos = 0 
        window = 63 # Strict Quarterly Tumbling Lookback
        slippage = 0.0005 
        borrow_fee = 0.02 / 252.0 
        
        print("Solving HJB Free-Boundaries and Routing Executions...")
        for t in range(window, T):
            # Extract local physics
            local_spread = spread[t-window:t]
            mu, sigma, half_life = governor.fit_rolling_ou_process(local_spread)
            
            is_stationary = (0 < half_life < 252)
            if is_stationary:
                theta = np.log(2) / half_life
                
                # Dynamic Friction Costing
                expected_holding_days = half_life
                round_trip_slippage = slippage * 2
                expected_borrow_cost = borrow_fee * expected_holding_days
                total_friction_bps = round_trip_slippage + expected_borrow_cost
                
                # PDE Grid Solver (Solve for Upper and Lower bounds dynamically)
                try:
                    # HJB returns the absolute distance from the mean to execute
                    optimal_z_dist = hjb_solver.solve_entry_boundary(mu=0.0, theta=theta, sigma=sigma, friction=total_friction_bps)
                    
                    if optimal_z_dist == -np.inf: # Friction is too high
                        bound_dist = np.inf
                    else:
                        bound_dist = abs(optimal_z_dist)
                        
                    hjb_upper[t] = mu + bound_dist
                    hjb_lower[t] = mu - bound_dist
                except Exception:
                    # Fallback if PDE fails to converge
                    hjb_upper[t] = mu + (1.5 * sigma)
                    hjb_lower[t] = mu - (1.5 * sigma)
            else:
                hjb_upper[t] = np.inf
                hjb_lower[t] = -np.inf

            # P&L Calculation
            if current_pos != 0:
                d_target = target_prices[t] - target_prices[t-1]
                d_peer = synthetic_index[t] - synthetic_index[t-1]
                profit = current_pos * (d_target - governor.beta_t[t-1] * d_peer)
                
                exposure = (governor.beta_t[t-1]*synthetic_index[t-1] if current_pos==1 else target_prices[t-1])
                profit -= exposure * borrow_fee
                pnl[t] = pnl[t-1] + profit
            else:
                pnl[t] = pnl[t-1]

            # HJB State Machine Execution
            prev_pos = current_pos
            if not is_stationary:
                current_pos = 0
            else:
                if current_pos == 0:
                    if spread[t] < hjb_lower[t]: current_pos = 1
                    elif spread[t] > hjb_upper[t]: current_pos = -1
                elif (current_pos == 1 and spread[t] >= mu) or (current_pos == -1 and spread[t] <= mu):
                    current_pos = 0
            
            if current_pos != prev_pos:
                cost = abs(current_pos - prev_pos) * (target_prices[t] + abs(governor.beta_t[t]*synthetic_index[t])) * slippage
                pnl[t] -= cost
                
            signals[t] = current_pos

        # 3. Master Visualization
        long_idx = np.where((signals == 1) & (np.roll(signals, 1) == 0))[0] 
        short_idx = np.where((signals == -1) & (np.roll(signals, 1) == 0))[0] 
        exit_idx = np.where((signals == 0) & (np.roll(signals, 1) != 0))[0]
        valid_dates = dates[window:]

        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            row_heights=[0.4, 0.3, 0.3],
                            subplot_titles=(f"{target_ric} Price with Execution Markers", 
                                            "Kalman Spread & HJB Dynamic Optimal Boundaries", 
                                            "Cumulative P&L (Net of HJB Friction)"))

        # Panel 1: Price & Executions
        fig.add_trace(go.Scatter(x=dates, y=target_prices, mode='lines', name='Target Price', line=dict(color='black', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates[long_idx], y=target_prices[long_idx] * 0.98, mode='markers', name='Enter LONG', marker=dict(symbol='triangle-up', size=12, color='green')), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates[short_idx], y=target_prices[short_idx] * 1.02, mode='markers', name='Enter SHORT', marker=dict(symbol='triangle-down', size=12, color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates[exit_idx], y=target_prices[exit_idx], mode='markers', name='EXIT', marker=dict(symbol='x', size=8, color='blue')), row=1, col=1)

        # Panel 2: Spread & HJB Bounds
        fig.add_trace(go.Scatter(x=valid_dates, y=spread[window:], mode='lines', name='Spread (Z_t)', line=dict(color='purple', width=1)), row=2, col=1)
        
        # Clean infinite bounds for plotting
        hjb_u_plot = np.where(hjb_upper[window:] > 5.0, np.nan, hjb_upper[window:])
        hjb_l_plot = np.where(hjb_lower[window:] < -5.0, np.nan, hjb_lower[window:])
        
        fig.add_trace(go.Scatter(x=valid_dates, y=hjb_u_plot, mode='lines', name='HJB Upper Entry', line=dict(color='red', dash='dot')), row=2, col=1)
        fig.add_trace(go.Scatter(x=valid_dates, y=hjb_l_plot, mode='lines', name='HJB Lower Entry', line=dict(color='green', dash='dot')), row=2, col=1)

        # Panel 3: P&L
        fig.add_trace(go.Scatter(x=valid_dates, y=pnl[window:], mode='lines', name='Cumulative P&L', line=dict(color='dodgerblue', width=2), fill='tozeroy'), row=3, col=1)

        fig.update_layout(title_text="Empirical HJB PDE Optimal Execution Backtest", hovermode="x unified", height=900, template="plotly_white")
        fig.show()

    finally:
        rd.close_session()

if __name__ == "__main__":
    run_hjb_production()