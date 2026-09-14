import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import refinitiv.data as rd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Ensure the src folder is accessible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tnic_universe import TNICUniverse
from src.models_midterm import MidTermGovernor
from src.models_intraday import IntradayMicrostructure

"""
MODULE: End-to-End Master Diagnostic
------------------------------------
Executes the full structural pipeline:
1. Data Fetch & Martingale Alignment
2. Kalman Filter Cointegration
3. Rigorous Econometric Stationarity Testing (ADF, ACF, PACF)
4. Empirical Copula Tail Dependence
5. Jump-Diffusion PIDE Parallel Simulation
"""

def test_and_visualize_pipeline():
    target_ric = "XOM.N" 
    peer_mapping = {"5903": "CVX.N", "61971": "COP.N", "25964": "OXY.N", "294524": "MPC.N"}
    similarity_scores = np.array([0.0629, 0.0417, 0.0171, 0.0108])
    peer_rics = list(peer_mapping.values())
    
    print("Opening LSEG Session...")
    rd.open_session()
    
    try:
        # --- 1. DATA INGESTION & ALIGNMENT ---
        print("\n[1/5] Fetching and Aligning Market Data...")
        all_rics = [target_ric] + peer_rics
        df_raw = rd.get_history(universe=all_rics, fields=["TRDPRC_1"], 
                                start="2021-01-01", end="2024-01-01", interval="1D")
        
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(
                df_raw.columns.names.index('Instrument') if 'Instrument' in df_raw.columns.names else 0
            )

        with pd.option_context('future.no_silent_downcasting', True):
            df_merged = df_raw.ffill().bfill().infer_objects(copy=False)
            
        df_merged = df_merged.astype(float).dropna()
        target_prices = df_merged[target_ric].values
        peer_prices_matrix = df_merged[peer_rics].values
        
        # --- 2. STRUCTURAL GOVERNOR (KALMAN FILTER) ---
        print("[2/5] Running Structural Kalman Filter...")
        universe = TNICUniverse(tnic_data_path="data/raw/tnic2_data.txt", target_year=2021)
        synthetic_peer_index = universe.construct_peer_index(similarity_scores, peer_prices_matrix)
        
        governor = MidTermGovernor(dt=1/252)
        spread = governor.fit_kalman_cointegration(target_prices, synthetic_peer_index)
        governor.fit_ou_process(spread)
        
        # --- 3. ECONOMETRIC STATIONARITY TESTING ---
        print("[3/5] Executing Augmented Dickey-Fuller Test...")
        # We test the spread after the Kalman burn-in period (first 20 days)
        valid_spread = spread[20:]
        
        # ADF Test minimizes AIC to find the optimal lag structure
        adf_result = adfuller(valid_spread, autolag='AIC')
        print(f"\n--- ADF Test Results ---")
        print(f"Test Statistic: {adf_result[0]:.4f}")
        print(f"P-Value: {adf_result[1]:.4e}")
        for key, value in adf_result[4].items():
            print(f"Critical Value ({key}): {value:.4f}")
            
        if adf_result[1] < 0.05:
            print("CONCLUSION: Spread is strictly stationary I(0). Mean-reversion confirmed.")
        else:
            print("CONCLUSION: Spread contains a unit root I(1). DO NOT TRADE.")

        # --- 4. INTRADAY MICROSTRUCTURE (COPULA & JUMP-DIFFUSION) ---
        print("\n[4/5] Running Intraday Microstructure Analysis...")
        target_log_returns = np.diff(np.log(target_prices))
        peer_index_log_returns = np.diff(np.log(synthetic_peer_index))
        
        intraday_engine = IntradayMicrostructure(dt=1.0/(252*390))
        tau_lower = intraday_engine.estimate_empirical_tail_dependence(target_log_returns, peer_index_log_returns)
        print(f"Empirical Lower Tail Dependence: {tau_lower:.4f}")
        
        # Run the parallelized 100,000 path simulation via joblib
        survival_prob = intraday_engine.simulate_jump_diffusion_survival(
            z_0=valid_spread[-1], mu=governor.mu, theta=governor.theta_params, 
            sigma=governor.sigma, lambda_j=5.0, mu_j=0.0, sigma_j=0.02, 
            horizon_steps=390, num_paths=100000, n_jobs=-1
        )
        print(f"Probability of Survival (Mean-Reversion before Jump): {survival_prob*100:.2f}%")

        # --- 5. VISUALIZATION DASHBOARD ---
        print("\n[5/5] Rendering Diagnostic Plots...")
        fig = plt.figure(figsize=(16, 12))
        grid = plt.GridSpec(3, 2, hspace=0.4, wspace=0.2)

        # Plot A: Historical Spread
        ax_spread = fig.add_subplot(grid[0, :])
        ax_spread.plot(df_merged.index[20:], valid_spread, color='purple', label='Kalman Spread (Z_t)')
        ax_spread.axhline(governor.mu, color='black', linestyle='--', label='Equilibrium ($\mu$)')
        
        std_dev = np.sqrt((governor.sigma**2) / (2 * governor.theta_params))
        ax_spread.axhline(governor.mu + 1.5*std_dev, color='red', linestyle=':', label='+1.5$\sigma$')
        ax_spread.axhline(governor.mu - 1.5*std_dev, color='green', linestyle=':', label='-1.5$\sigma$')
        ax_spread.set_title(f'Kalman Filtered Cointegration Spread (Half-Life: {governor.half_life*252:.1f} Days)')
        ax_spread.legend(loc='upper right')

        # Plot B & C: ACF and PACF
        ax_acf = fig.add_subplot(grid[1, 0])
        plot_acf(valid_spread, ax=ax_acf, zero=False, lags=40, title='Autocorrelation (ACF)')
        
        ax_pacf = fig.add_subplot(grid[1, 1])
        plot_pacf(valid_spread, ax=ax_pacf, zero=False, lags=40, title='Partial Autocorrelation (PACF)', method='ywm')

        # Plot D: Jump-Diffusion Simulation (Plotting 100 sample paths)
        ax_sim = fig.add_subplot(grid[2, :])
        
        # Generate a small local batch purely for visual rendering
        horizon = 390
        Z_sim = np.zeros((100, horizon))
        Z_sim[:, 0] = valid_spread[-1]
        dt = 1.0/(252*390)
        dW = np.random.normal(0, np.sqrt(dt), size=(100, horizon - 1))
        dN = np.random.binomial(1, 5.0 * dt, size=(100, horizon - 1))
        xi = np.random.normal(0.0, 0.02, size=(100, horizon - 1))
        
        for t in range(1, horizon):
            drift = governor.theta_params * (governor.mu - Z_sim[:, t-1]) * dt
            diffusion = governor.sigma * dW[:, t-1]
            jump = xi[:, t-1] * dN[:, t-1]
            Z_sim[:, t] = Z_sim[:, t-1] + drift + diffusion + jump
            
        
            
        time_axis = np.arange(horizon)
        for i in range(100):
            ax_sim.plot(time_axis, Z_sim[i, :], color='blue', alpha=0.1)
            
        ax_sim.axhline(governor.mu, color='black', linestyle='--', linewidth=2, label='Absorbing Mean ($\mu$)')
        ax_sim.set_title(f'Jump-Diffusion Microstructure Simulation (1 Trading Day, P(Survival) = {survival_prob*100:.1f}%)')
        ax_sim.set_xlabel('Intraday Minutes')
        ax_sim.set_ylabel('Spread (Z)')
        ax_sim.legend()

        plt.show()

    except Exception as e:
        print(f"\n[FATAL ERROR] {repr(e)}")
    finally:
        rd.close_session()

if __name__ == "__main__":
    test_and_visualize_pipeline()