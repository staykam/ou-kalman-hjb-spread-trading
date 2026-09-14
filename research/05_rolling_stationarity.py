import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import refinitiv.data as rd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings

# Suppress statsmodels warnings for clean terminal output
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tnic_universe import TNICUniverse
from src.models_midterm import MidTermGovernor

def analyze_rolling_stationarity():
    target_ric = "XOM.N" 
    peer_mapping = {"5903": "CVX.N", "61971": "COP.N", "25964": "OXY.N", "294524": "MPC.N"}
    similarity_scores = np.array([0.0629, 0.0417, 0.0171, 0.0108])
    peer_rics = list(peer_mapping.values())
    
    print("Opening LSEG Session...")
    rd.open_session()
    
    try:
        # 1. Fetch Data
        all_rics = [target_ric] + peer_rics
        df_raw = rd.get_history(universe=all_rics, fields=["TRDPRC_1"], 
                                start="2020-01-01", end="2024-01-01", interval="1D")
        
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)

        with pd.option_context('future.no_silent_downcasting', True):
            df_merged = df_raw.ffill().bfill().infer_objects(copy=False)
            
        df_merged = df_merged.astype(float).dropna()
        target_prices = df_merged[target_ric].values
        peer_prices_matrix = df_merged[peer_rics].values
        dates = df_merged.index
        
        # 2. Extract the Global Kalman Spread
        universe = TNICUniverse(tnic_data_path="data/raw/tnic2_data.txt", target_year=2021)
        synthetic_peer_index = universe.construct_peer_index(similarity_scores, peer_prices_matrix)
        governor = MidTermGovernor(dt=1/252)
        spread = governor.fit_kalman_cointegration(target_prices, synthetic_peer_index)
        
        # 3. Define the Local Topologies (Windows in Trading Days)
        # 63 days ~ 3 Months, 126 days ~ 6 Months, 210 days ~ 10 Months
        windows = {'3 Months': 63, '6 Months': 126, '10 Months': 210}
        
        results_pvalue = {k: np.full(len(spread), np.nan) for k in windows.keys()}
        results_halflife = {k: np.full(len(spread), np.nan) for k in windows.keys()}
        
        print("\nRunning Rolling Econometric Tests...")
        for name, w in windows.items():
            print(f"Processing {name} Window (T={w})...")
            for t in range(w, len(spread)):
                local_spread = spread[t-w:t]
                
                # A. Rolling ADF Test
                try:
                    adf_stat = adfuller(local_spread, autolag='AIC')
                    results_pvalue[name][t] = adf_stat[1] # Store p-value
                except:
                    pass
                
                # B. Rolling Ornstein-Uhlenbeck Half-Life
                Z_t = local_spread[1:]
                Z_t_minus_1 = local_spread[:-1]
                X = sm.add_constant(Z_t_minus_1)
                
                try:
                    model = sm.OLS(Z_t, X).fit()
                    b = model.params[1]
                    if b < 1 and b > 0:
                        theta = -np.log(b) / (1/252)
                        results_halflife[name][t] = (np.log(2) / theta) * 252 # In Days
                except:
                    pass

        # 4. Visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Plot A: ADF P-Values (Is it stationary?)
        for name in windows.keys():
            ax1.plot(dates, results_pvalue[name], label=f'{name} Window', alpha=0.8)
        
        ax1.axhline(0.05, color='red', linestyle='--', linewidth=2, label='0.05 Significance (Reject Unit Root)')
        ax1.set_title('Rolling ADF Stationarity P-Value', fontsize=12, fontweight='bold')
        ax1.set_ylabel('P-Value')
        ax1.set_ylim(-0.05, 1.05)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot B: OU Half-Life (How fast is the rubber band?)
        for name in windows.keys():
            ax2.plot(dates, results_halflife[name], label=f'{name} Window', alpha=0.8)
            
        ax2.set_title('Rolling Expected Trade Half-Life (Days)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Half-Life (Trading Days)')
        ax2.set_ylim(0, 100) # Cap y-axis to see the detail
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Rolling Test Error: {e}")
    finally:
        rd.close_session()

if __name__ == "__main__":
    analyze_rolling_stationarity()