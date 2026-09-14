import sys
import os
import numpy as np
import pandas as pd
import refinitiv.data as rd

# Add the src folder to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tnic_universe import TNICUniverse
from src.models_midterm import MidTermGovernor

def run_midterm_alpha():
    """Currently hardcoded for ExxonMobil (XOM) and its supply-chain peers because we need a relational database to map the GVKEYs from hoberg into RICs for the LSEG API. 
    This will be fixed in the future with a more robust mapping layer and allow more peers and targets to be analyzed. The main point of this script is to demonstrate the
    cointegration test and OU process fitting on a real-world example."""
    target_ric = "XOM.N" 
    
    peer_mapping = {
        "5903": "CVX.N",    
        "61971": "COP.N",   
        "25964": "OXY.N",   
        "294524": "MPC.N",     
    }
    
    similarity_scores = np.array([0.0629, 0.0417, 0.0171, 0.0108])
    peer_rics = list(peer_mapping.values())
    
    print("Opening LSEG Session...")
    rd.open_session()
    
    try:
        all_rics = [target_ric] + peer_rics
        print(f"Fetching matrix for: {all_rics}")
        
        # A single API call
        df_raw = rd.get_history(
            universe=all_rics, 
            fields=["TRDPRC_1"], 
            start="2021-01-01", 
            end="2024-01-01", 
            interval="1D"
        )
        
        # 1. Handle LSEG MultiIndex return safely
        if isinstance(df_raw.columns, pd.MultiIndex):
            if target_ric in df_raw.columns.get_level_values(0):
                df_raw.columns = df_raw.columns.get_level_values(0)
            elif target_ric in df_raw.columns.get_level_values(1):
                df_raw.columns = df_raw.columns.get_level_values(1)

        # 2. The Martingale Data Fill (Fixing the <NA> issue)
        with pd.option_context('future.no_silent_downcasting', True):
            df_merged = df_raw.ffill().bfill()
            
        # --- THE FIREWALL ---
        # Force the dataframe into pure numpy floats. This strips out all `pd.NA` objects.
        df_merged = df_merged.astype(float)
        # Look inside the box to see how many missing days exist BEFORE dropping
        print(f"[DEBUG] Missing data points per stock before drop:\n{df_merged.isna().sum()}")
        
        # Drop any rows that still contain np.nan (e.g., if a stock didn't exist at the start of 2021)
        df_merged = df_merged.dropna()
        # --------------------
        
        # 3. Aggressively verify columns
        missing_rics = [ric for ric in all_rics if ric not in df_merged.columns]
        if missing_rics:
            raise KeyError(f"Missing RICs from LSEG: {missing_rics}. Available: {list(df_merged.columns)}")
            
        # Extract pure numpy arrays for the math engine
        target_prices = df_merged[target_ric].values
        peer_prices_matrix = df_merged[peer_rics].values
        
        print(f"Data aligned. Shape: {peer_prices_matrix.shape} (Days, Peers)")
        
        # 4. Construct the Synthetic Index
        universe = TNICUniverse(tnic_data_path="data/raw/tnic2_data.txt", target_year=2021)
        synthetic_peer_index = universe.construct_peer_index(similarity_scores, peer_prices_matrix)
        
        # [PREVIOUS CODE REMAINS THE SAME UNTIL INITIALIZING THE GOVERNOR]
        
        # 5. Initialize the Governor and run the physics
        governor = MidTermGovernor(dt=1/252)
        print("\n--- Fitting Structural Model (Kalman Filter) ---")
        spread = governor.fit_kalman_cointegration(target_prices, synthetic_peer_index)
        governor.fit_ou_process(spread)
        
        final_beta = governor.beta_t[-1]
        print(f"Final Cointegration Vector (Beta_T): {final_beta:.4f}")
        print(f"Expected Trade Half-Life: {governor.half_life*252:.2f} Days")
        
        # 6. Extract Log Returns (Strictly I(0) transformations for the Intraday Copula)
        # np.diff(np.log()) automatically handles the T to T-1 shift.
        target_log_returns = np.diff(np.log(target_prices))
        peer_index_log_returns = np.diff(np.log(synthetic_peer_index))
        
        # 7. Intraday Microstructure Analysis
        from src.models_intraday import IntradayMicrostructure
        
        # We assume 390 minutes in a trading day.
        intraday_engine = IntradayMicrostructure(dt=1.0/(252*390))
        
        tau_lower = intraday_engine.estimate_empirical_tail_dependence(
            target_log_returns, 
            peer_index_log_returns, 
            q=0.05
        )
        print(f"\n--- Intraday Microstructure Analysis ---")
        print(f"Empirical Lower Tail Dependence (\tau^L at 5%): {tau_lower:.4f}")
        
        # Define simulation horizon: 1 trading day (390 minutes)
        horizon = 390 
        
        print(f"Simulating {100000} Jump-Diffusion paths across CPU cores...")
        survival_prob = intraday_engine.simulate_jump_diffusion_survival(
            z_0=spread[-1], 
            mu=governor.mu, 
            theta=governor.theta_params, 
            sigma=governor.sigma, 
            lambda_j=5.0,     # Assume 5 liquidity jumps expected per year
            mu_j=0.0,         # Jumps are directionally symmetric on average
            sigma_j=0.02,     # Jump volatility size
            horizon_steps=horizon,
            num_paths=100000
        )
        
        print(f"Probability of Mean-Reversion Survival before Jump: {survival_prob*100:.2f}%")
        
        # Final Goal Tie-in: The bot only trades if BOTH systems agree.
        # If survival probability is < 50%, we veto the trade regardless of Z-score.
        """ This section is temporarily removed 
        print("\n--- Final Governor Output ---")
        if signal == 1:
            print("STATE: ALLOW LONG (+1). Target is structurally underpriced relative to supply-chain peers.")
        elif signal == -1:
            print("STATE: ALLOW SHORT (-1). Target is structurally overpriced relative to supply-chain peers.")
        else:
            print("STATE: CASH ONLY (0). Spread is within normal noise variance. Do not trade.")
        """
    except Exception as e:
        print(f"Pipeline Error: {repr(e)}")
    finally:
        rd.close_session()

if __name__ == "__main__":
    run_midterm_alpha()