import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import refinitiv.data as rd
from joblib import Parallel, delayed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tnic_universe import TNICUniverse
from src.models_midterm import MidTermGovernor

def generate_parallel_universes(batch_size: int, z_0: float, mu: float, theta: float, 
                                sigma: float, horizon_steps: int, dt: float) -> np.ndarray:
    """
    Worker function to generate a batch of stochastic paths via Euler-Maruyama.
    Returns a matrix of shape (batch_size, horizon_steps).
    """
    Z = np.zeros((batch_size, horizon_steps))
    Z[:, 0] = z_0
    
    # Pre-compute Brownian increments: dW ~ N(0, dt)
    dW = np.random.normal(0, np.sqrt(dt), size=(batch_size, horizon_steps - 1))
    
    for t in range(1, horizon_steps):
        drift = theta * (mu - Z[:, t-1]) * dt
        diffusion = sigma * dW[:, t-1]
        Z[:, t] = Z[:, t-1] + drift + diffusion
        
    return Z

def run_ergodicity_pit_test():
    target_ric = "XOM.N" 
    peer_mapping = {"5903": "CVX.N", "61971": "COP.N", "25964": "OXY.N", "294524": "MPC.N"}
    similarity_scores = np.array([0.0629, 0.0417, 0.0171, 0.0108])
    peer_rics = list(peer_mapping.values())
    
    print("Opening LSEG Session...")
    rd.open_session()
    
    try:
        # 1. Fetch Data
        df_raw = rd.get_history(universe=[target_ric] + peer_rics, fields=["TRDPRC_1"], 
                                start="2021-01-01", end="2024-01-01", interval="1D")
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)

        with pd.option_context('future.no_silent_downcasting', True):
            df_merged = df_raw.ffill().bfill().infer_objects(copy=False)
            
        df_merged = df_merged.astype(float).dropna()
        target_prices = df_merged[target_ric].values
        peer_prices = df_merged[peer_rics].values
        
        # 2. Extract the Realized Physical Path (Z_t)
        universe = TNICUniverse(tnic_data_path="data/raw/tnic2_data.txt", target_year=2021)
        synthetic_peer_index = universe.construct_peer_index(similarity_scores, peer_prices)
        
        governor = MidTermGovernor(dt=1/252)
        realized_spread = governor.fit_kalman_cointegration(target_prices, synthetic_peer_index)
        
        # Fit global OU parameters to define the "Model Universe"
        # Using a stationary subset to calibrate the simulation laws of physics
        valid_spread = realized_spread[20:] 
        mu, sigma, half_life = governor.fit_rolling_ou_process(valid_spread)
        theta = np.log(2) / half_life if half_life > 0 else 0.0
        
        # 3. Simulate the Ensemble (M = 16,000 paths) using joblib
        T_steps = len(valid_spread)
        M_paths = 16000
        n_batches = 16
        paths_per_batch = M_paths // n_batches
        
        print(f"\nSimulating {M_paths} parallel universes across 16 CPU threads...")
        
        # The physical time step is daily
        dt = 1/252.0 
        
        results = Parallel(n_jobs=-1)(
            delayed(generate_parallel_universes)(
                paths_per_batch, valid_spread[0], mu, theta, sigma, T_steps, dt
            ) for _ in range(n_batches)
        )
        
        # Stack the list of batch matrices into one massive (M x T) matrix
        simulated_ensemble = np.vstack(results)
        
        # 4. The Probability Integral Transform (PIT)
        print("Projecting realized physical path onto the simulated measure space...")
        U_t = np.zeros(T_steps)
        for t in range(T_steps):
            ensemble_t = simulated_ensemble[:, t]
            # Empirical CDF: What percentage of parallel universes are below the real world?
            U_t[t] = np.mean(ensemble_t <= valid_spread[t])
            
        # 5. Render the Diagnostic
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(U_t, bins=20, range=(0, 1), density=True, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Theoretical Uniform Density')
        
        ax.set_title('Probability Integral Transform (PIT) of Realized Path\nTesting Ergodicity and Model Specification', fontweight='bold')
        ax.set_xlabel('Empirical Quantile $U_t$')
        ax.set_ylabel('Density')
        ax.legend()
        plt.grid(alpha=0.3)
        plt.show()

    finally:
        rd.close_session()

if __name__ == "__main__":
    run_ergodicity_pit_test()