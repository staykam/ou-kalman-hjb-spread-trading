import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import refinitiv.data as rd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tnic_universe import TNICUniverse
from src.models_midterm import MidTermGovernor

def run_visual_backtest():
    target_ric = "XOM.N" 
    peer_mapping = {"5903": "CVX.N", "61971": "COP.N", "25964": "OXY.N", "294524": "MPC.N"}
    similarity_scores = np.array([0.0629, 0.0417, 0.0171, 0.0108])
    peer_rics = list(peer_mapping.values())
    
    rd.open_session()
    try:
        # 1. Fetch and Align Data
        all_rics = [target_ric] + peer_rics
        df_raw = rd.get_history(universe=all_rics, fields=["TRDPRC_1"], 
                                start="2021-01-01", end="2024-01-01", interval="1D")
        
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)

        with pd.option_context('future.no_silent_downcasting', True):
            df_merged = df_raw.ffill().bfill().infer_objects(copy=False)
            
        df_merged = df_merged.astype(float).dropna()
        target_prices = df_merged[target_ric].values
        peer_prices_matrix = df_merged[peer_rics].values
        dates = df_merged.index
        
        # 2. Structural Governor (Kalman Filter)
        universe = TNICUniverse(tnic_data_path="data/raw/tnic2_data.txt", target_year=2021)
        synthetic_peer_index = universe.construct_peer_index(similarity_scores, peer_prices_matrix)
        
        governor = MidTermGovernor(dt=1/252)
        spread = governor.fit_kalman_cointegration(target_prices, synthetic_peer_index)
        governor.fit_ou_process(spread)
        
        # 3. Simulate Signal Generation (With Hysteresis / Optimal Stopping)
        T = len(target_prices)
        signals = np.zeros(T)
        current_position = 0 # 0 = Cash, 1 = Long Target, -1 = Short Target
        
        burn_in = 20 
        entry_threshold = 1.5
        exit_threshold = 0.0 # Reversion to the mean
        
        for t in range(burn_in, T):
            # We now request the raw z-score from the governor
            z_score = governor.generate_signal(
                current_target=target_prices[t], 
                current_peer=synthetic_peer_index[t], 
                current_alpha=governor.alpha_t[t],
                current_beta=governor.beta_t[t]
            )
            
            # State Machine Logic
            if current_position == 0:
                # We are in Cash. Look for entry signals.
                if z_score < -entry_threshold:
                    current_position = 1   # Go Long Target
                elif z_score > entry_threshold:
                    current_position = -1  # Go Short Target
            
            elif current_position == 1:
                # We are Long. Look for exit signal.
                if z_score >= exit_threshold:
                    current_position = 0   # Close position
                    
            elif current_position == -1:
                # We are Short. Look for exit signal.
                if z_score <= exit_threshold:
                    current_position = 0   # Close position
                    
            signals[t] = current_position
            
        # Extract indices for plotting markers
        long_indices = np.where(signals == 1)[0]   # Buy Target, Short Peer
        short_indices = np.where(signals == -1)[0] # Short Target, Buy Peer
        
        # 4. Render the Backtest Visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # --- Top Plot: Raw Prices with Execution Markers ---
        ax1.plot(dates, target_prices, color='black', alpha=0.7, label=f'Target ({target_ric})')
        
        # Plot execution markers on the price line
        ax1.scatter(dates[long_indices], target_prices[long_indices], 
                    marker='^', color='green', s=100, zorder=5, label='Signal: LONG Target')
        ax1.scatter(dates[short_indices], target_prices[short_indices], 
                    marker='v', color='red', s=100, zorder=5, label='Signal: SHORT Target')
        
        ax1.set_title(r'Price Trajectory and Model Executions', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Price (USD)')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # --- Bottom Plot: The Cointegrated Spread ---
        ax2.plot(dates[burn_in:], spread[burn_in:], color='purple', label=r'Spread ($Z_t$)')
        ax2.axhline(governor.mu, color='black', linestyle='--', label=r'Equilibrium ($\mu$)')
        
        std_dev = np.sqrt((governor.sigma**2) / (2 * governor.theta_params))
        upper_bound = governor.mu + 1.5 * std_dev
        lower_bound = governor.mu - 1.5 * std_dev
        
        ax2.axhline(upper_bound, color='red', linestyle=':', label=r'+1.5$\sigma$ Threshold')
        ax2.axhline(lower_bound, color='green', linestyle=':', label=r'-1.5$\sigma$ Threshold')
        
        # Highlight periods where we are actively holding a position
        ax2.fill_between(dates[burn_in:], upper_bound, spread[burn_in:], 
                         where=(spread[burn_in:] > upper_bound), color='red', alpha=0.3)
        ax2.fill_between(dates[burn_in:], lower_bound, spread[burn_in:], 
                         where=(spread[burn_in:] < lower_bound), color='green', alpha=0.3)
                         
        ax2.set_title(r'Time-Varying Kalman Spread Dynamics', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Spread Value')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Backtest Visualizer Error: {e}")
    finally:
        rd.close_session()

if __name__ == "__main__":
    run_visual_backtest()