import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import refinitiv.data as rd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tnic_universe import TNICUniverse
from src.models_midterm import MidTermGovernor

def visualize_structural_alpha():
    target_ric = "XOM.N" 
    peer_mapping = {"5903": "CVX.N", "61971": "COP.N", "25964": "OXY.N", "294524": "MPC.N"}
    similarity_scores = np.array([0.0629, 0.0417, 0.0171, 0.0108])
    peer_rics = list(peer_mapping.values())
    
    rd.open_session()
    
    try:
        df_raw = rd.get_history(
            universe=[target_ric] + peer_rics, 
            fields=["TRDPRC_1"], start="2021-01-01", end="2024-01-01", interval="1D"
        )
        
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(df_raw.columns.names.index('Instrument'))

        with pd.option_context('future.no_silent_downcasting', True):
            df_merged = df_raw.ffill().bfill().infer_objects(copy=False)
        df_merged = df_merged.astype(float).dropna()
        
        target_prices = df_merged[target_ric].values
        peer_prices_matrix = df_merged[peer_rics].values
        
        universe = TNICUniverse(tnic_data_path="data/raw/tnic2_data.txt", target_year=2021)
        synthetic_peer_index = universe.construct_peer_index(similarity_scores, peer_prices_matrix)
        
        governor = MidTermGovernor(dt=1/252)
        spread = governor.fit_cointegration(target_prices, synthetic_peer_index)
        governor.fit_ou_process(spread)
        
        # --- THE PLOTTING LOGIC ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
        
        # Top Plot: Normalized Prices
        ax1.plot(df_merged.index, target_prices / target_prices[0], label=f'Target ({target_ric})', color='blue')
        ax1.plot(df_merged.index, synthetic_peer_index / synthetic_peer_index[0], label='Synthetic Peer Index', color='orange', alpha=0.8)
        ax1.set_title('Normalized Trajectories: Target vs. Synthetic Peers', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Bottom Plot: The Cointegrated Spread & OU Bounds
        z_score_threshold = 1.5
        asymptotic_var = (governor.sigma**2) / (2 * governor.theta)
        std_dev = np.sqrt(asymptotic_var)
        
        upper_bound = governor.mu + (z_score_threshold * std_dev)
        lower_bound = governor.mu - (z_score_threshold * std_dev)
        
        ax2.plot(df_merged.index, spread, label='Spread (Z_t)', color='purple')
        ax2.axhline(governor.mu, color='black', linestyle='--', label='Equilibrium (\mu)')
        ax2.axhline(upper_bound, color='red', linestyle=':', label='+1.5\sigma (Short Target)')
        ax2.axhline(lower_bound, color='green', linestyle=':', label='-1.5\sigma (Long Target)')
        
        ax2.set_title(f'Ornstein-Uhlenbeck Spread Dynamics (Half-Life: {governor.half_life*252:.1f} Days)', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Visualization Error: {e}")
    finally:
        rd.close_session()

if __name__ == "__main__":
    visualize_structural_alpha()