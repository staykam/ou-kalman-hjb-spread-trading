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

def simulate_jump_diffusion_paths(z_0, mu, theta, sigma, horizon, dt, num_paths):
    r"""
    Generates M parallel paths of the Jump-Diffusion process.
    dZ_t = \theta(\mu - Z_t)dt + \sigma dW_t + \xi dN_t
    This introduces 'Fat Tails' and structural shocks to the simulation.
    """
    Z = np.zeros((num_paths, horizon))
    Z[:, 0] = z_0
    
    # Continuous Diffusion
    dW = np.random.normal(0, np.sqrt(dt), size=(num_paths, horizon - 1))
    
    # Poisson Jumps (Assume roughly 5 major liquidity shocks per year)
    lambda_j = 5.0
    jump_prob = lambda_j * dt
    dN = np.random.binomial(1, jump_prob, size=(num_paths, horizon - 1))
    
    # Jump Size (Can be violent: 3x the normal daily standard deviation)
    xi = np.random.normal(0, sigma * 3.0, size=(num_paths, horizon - 1))
    
    for t in range(1, horizon):
        drift = theta * (mu - Z[:, t-1]) * dt
        diffusion = sigma * dW[:, t-1]
        jump = xi[:, t-1] * dN[:, t-1]
        
        Z[:, t] = Z[:, t-1] + drift + diffusion + jump
        
    return Z

def run_realistic_monte_carlo():
    target_ric = "XOM.N" 
    peer_mapping = {"5903": "CVX.N", "61971": "COP.N", "25964": "OXY.N", "294524": "MPC.N"}
    similarity_scores = np.array([0.0629, 0.0417, 0.0171, 0.0108])
    peer_rics = list(peer_mapping.values())
    
    print("Extracting physical parameters from LSEG...")
    rd.open_session()
    try:
        df_raw = rd.get_history(universe=[target_ric] + peer_rics, fields=["TRDPRC_1"], 
                                start="2021-01-01", end="2024-01-01", interval="1D")
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)
        df_merged = df_raw.ffill().bfill().dropna().astype(float)
        
        target_prices = df_merged[target_ric].values
        peer_prices = df_merged[peer_rics].values
        
        universe = TNICUniverse(tnic_data_path="data/raw/tnic2_data.txt", target_year=2021)
        synthetic_peer_index = universe.construct_peer_index(similarity_scores, peer_prices)
        
        governor = MidTermGovernor(dt=1/252)
        spread = governor.fit_kalman_cointegration(target_prices, synthetic_peer_index)
        
        # Extract Robust parameters
        mu, sigma, half_life = governor.fit_rolling_ou_process(spread[-252:])
        theta = np.log(2) / half_life if half_life > 0 else 0.0
        
        if theta <= 0:
            print("Spread is currently non-stationary. Cannot simulate ergodic P&L.")
            return

        # 2. Setup the Jump-Diffusion Universes
        horizon_days = 252 
        num_paths = 1000
        dt = 1/252.0
        
        print(f"Simulating {num_paths} hostile parallel universes over {horizon_days} days...")
        simulated_spreads = simulate_jump_diffusion_paths(spread[-1], mu, theta, sigma, horizon_days, dt, num_paths)
        
        # 3. Vectorized Backtest (Corrected Leverage & Friction)
        # We assume 1 unit of spread correlates to roughly 1 physical share for simplicity of P&L scale
        shares_traded = 100 
        bps_slippage = 0.0005
        daily_borrow = 0.02 / 252.0
        target_price_proxy = target_prices[-1] # Proxy for exposure
        
        asymptotic_var = (sigma**2) / (2 * theta)
        std_dev = np.sqrt(asymptotic_var)
        entry_upper = mu + 1.5 * std_dev
        entry_lower = mu - 1.5 * std_dev
        
        pnl_matrix = np.zeros((num_paths, horizon_days))
        positions = np.zeros(num_paths) 
        
        for t in range(1, horizon_days):
            current_spreads = simulated_spreads[:, t]
            prev_spreads = simulated_spreads[:, t-1]
            
            # Pure price difference * physical shares (No infinite std_dev leverage)
            daily_profit = positions * (current_spreads - prev_spreads) * shares_traded
            
            # Borrow costs based on proxy physical exposure
            borrow_costs = np.abs(positions) * (target_price_proxy * shares_traded) * daily_borrow
            
            pnl_matrix[:, t] = pnl_matrix[:, t-1] + daily_profit - borrow_costs
            
            prev_positions = positions.copy()
            
            # State Machine transitions
            positions[(positions == 0) & (current_spreads < entry_lower)] = 1
            positions[(positions == 0) & (current_spreads > entry_upper)] = -1
            positions[(positions == 1) & (current_spreads >= mu)] = 0
            positions[(positions == -1) & (current_spreads <= mu)] = 0
            
            # Slippage Penalty
            state_changes = np.abs(positions - prev_positions)
            slippage_costs = state_changes * (target_price_proxy * shares_traded * 2) * bps_slippage
            pnl_matrix[:, t] -= slippage_costs

        # 4. Render the QuantGuild-Style Visual
        terminal_pnls = pnl_matrix[:, -1]
        expected_value = np.mean(terminal_pnls)
        win_rate = np.mean(terminal_pnls > 0) * 100
        
        print(f"Expected Ensemble P&L: ${expected_value:.2f}")
        print(f"Probability of Profit (Win Rate): {win_rate:.1f}%")

        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.8, 0.2],
                            subplot_titles=("Jump-Diffusion P&L Trajectories (1000 Paths)", "Terminal Distribution"))
        
        time_axis = np.arange(horizon_days)
        for i in range(250): # Plot 250 paths to keep browser light
            color = 'green' if pnl_matrix[i, -1] > 0 else 'red'
            fig.add_trace(go.Scatter(x=time_axis, y=pnl_matrix[i, :], mode='lines', line=dict(color=color, width=0.5), opacity=0.3), row=1, col=1)
            
        fig.add_trace(go.Scatter(x=time_axis, y=np.mean(pnl_matrix, axis=0), mode='lines', name='Expected Value', line=dict(color='black', width=3)), row=1, col=1)

        fig.add_trace(go.Histogram(y=terminal_pnls, orientation='h', marker_color='steelblue', name='Terminal Density'), row=1, col=2)
        fig.add_trace(go.Scatter(x=[0, max(np.histogram(terminal_pnls)[0])], y=[0, 0], mode='lines', line=dict(color='black', dash='dash', width=2), name='Breakeven'), row=1, col=2)

        fig.update_layout(title_text=f"Stochastic Control Output: E[P&L] = ${expected_value:.2f} | Win Rate: {win_rate:.1f}%", 
                          showlegend=False, template="plotly_white", height=700)
        fig.update_yaxes(title_text="Cumulative Dollar P&L", row=1, col=1)
        fig.show()

    finally:
        rd.close_session()

if __name__ == "__main__":
    run_realistic_monte_carlo()