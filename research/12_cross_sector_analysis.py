import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import refinitiv.data as rd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.tnic_universe import TNICUniverse
from src.models_midterm import MidTermGovernor

# --- THE DATA ENGINEERING BRIDGE ---
# Mapping Refinitiv RICs to Compustat GVKEYs (Strings without leading zeros to match your prior file)
RIC_TO_GVKEY = {
    # Semis
    "NVDA.O": "104524", "AMD.O": "1161", "INTC.O": "6008", "TXN.O": "10493", "MU.O": "7309", "QCOM.O": "28308", "AVGO.O": "177265", "AMAT.O": "1436",
    # Banks
    "JPM.N": "2983", "BAC.N": "2086", "WFC.N": "11308", "C.N": "3058", "MS.N": "7435", "GS.N": "64011", "USB.N": "10973", "PNC.N": "8549",
    # Oil
    "XOM.N": "4503", "CVX.N": "5903", "COP.N": "61971", "OXY.N": "25964", "MPC.N": "294524", "VLO.N": "13721", "PSX.N": "146869", "HES.N": "5471",
    # Defense
    "LMT.N": "6840", "RTX.N": "11048", "GD.N": "5046", "NOC.N": "7954", "BA.N": "2209"
}
# Inverse mapping for the return trip
GVKEY_TO_RIC = {v: k for k, v in RIC_TO_GVKEY.items()}

def run_cross_sector_analysis():
    # Test 4 major sectors with vastly different supply chain physics
    sectors = {
        "Semiconductors": "NVDA.O",
        "Banking": "JPM.N",
        "Oil": "XOM.N",
        "Defense": "LMT.N"
    }
    
    top_n = 5 # Reduced to 5 to ensure we only hit highly liquid, mapped peers
    window = 126
    slippage = 0.0005 
    borrow_fee = 0.02 / 252.0 
    
    universe = TNICUniverse(tnic_data_path="data/raw/tnic2_data.txt", target_year=2021)
    
    results = []
    equity_curves = {}
    valid_dates = None
    
    rd.open_session()
    try:
        for sector_name, target_ric in sectors.items():
            print(f"\n[{sector_name}] Initializing dynamic topology for {target_ric}...")
            
            try:
                # Pass the bridging dictionaries into the graph search
                peer_rics, similarity_weights = universe.get_top_n_peers(target_ric, RIC_TO_GVKEY, GVKEY_TO_RIC, n_peers=top_n)
                
                df_merged = rd.get_history(universe=[target_ric] + peer_rics, fields=["TRDPRC_1"], 
                                           start="2021-01-01", end="2024-01-01", interval="1D")
                if isinstance(df_merged.columns, pd.MultiIndex):
                    df_merged.columns = df_merged.columns.get_level_values(0)
                    
                df_merged = df_merged.ffill().bfill().dropna().astype(float)
                target_prices = df_merged[target_ric].values
                peer_prices = df_merged[peer_rics].values
                dates = df_merged.index
                T = len(target_prices)
                
                if valid_dates is None:
                    valid_dates = dates[window:]
                
                synthetic_peer_index = universe.construct_peer_index(similarity_weights, peer_prices)
                
                governor = MidTermGovernor(dt=1/252)
                spread = governor.fit_kalman_cointegration(target_prices, synthetic_peer_index)
                
                pnl = np.zeros(T)
                daily_returns = np.zeros(T)
                current_pos = 0 
                
                for t in range(window, T):
                    local_spread = spread[t-window:t]
                    mu, sigma, half_life = governor.fit_rolling_ou_process(local_spread)
                    
                    is_stationary = (0 < half_life < 252)
                    if is_stationary:
                        theta = np.log(2) / half_life
                        robust_std = np.sqrt((sigma**2) / (2 * theta)) * 0.75 
                        z_score = (spread[t] - mu) / (robust_std + 1e-8)
                    else:
                        z_score = 0.0
                        
                    is_confident = (np.abs(governor.e_t[t]) / (np.sqrt(governor.S_t[t]) + 1e-8)) < 3.0
                    
                    if current_pos != 0:
                        d_target = target_prices[t] - target_prices[t-1]
                        d_peer = synthetic_peer_index[t] - synthetic_peer_index[t-1]
                        profit = current_pos * (d_target - governor.beta_t[t-1] * d_peer)
                        
                        exposure = (governor.beta_t[t-1]*synthetic_peer_index[t-1] if current_pos==1 else target_prices[t-1])
                        profit -= exposure * borrow_fee
                        pnl[t] = pnl[t-1] + profit
                        denominator = target_prices[t-1] + abs(governor.beta_t[t-1]*synthetic_peer_index[t-1])
                        daily_returns[t] = profit / denominator if denominator > 0 else 0
                    else:
                        pnl[t] = pnl[t-1]

                    prev_pos = current_pos
                    if not is_stationary:
                        current_pos = 0
                    else:
                        if current_pos == 0 and is_confident:
                            if z_score < -1.5: current_pos = 1
                            elif z_score > 1.5: current_pos = -1
                        elif (current_pos == 1 and z_score >= 0) or (current_pos == -1 and z_score <= 0):
                            current_pos = 0
                    
                    if current_pos != prev_pos:
                        cost = abs(current_pos - prev_pos) * (target_prices[t] + abs(governor.beta_t[t]*synthetic_peer_index[t])) * slippage
                        pnl[t] -= cost

                active_returns = daily_returns[daily_returns != 0]
                sharpe = (np.mean(active_returns) / np.std(active_returns)) * np.sqrt(252) if len(active_returns) > 0 else 0
                
                results.append({
                    "Sector": sector_name,
                    "Anchor RIC": target_ric,
                    "Opt Q (State)": f"{governor.opt_q:.2e}",
                    "Opt R (Obs)": f"{governor.opt_r:.2e}",
                    "Final P&L": round(pnl[-1], 2),
                    "Sharpe Ratio": round(sharpe, 2)
                })
                
                equity_curves[sector_name] = pnl[window:]
                
            except Exception as e:
                print(f"Failed to process {sector_name}: {str(e)}")

        df_results = pd.DataFrame(results)
        print("\n=== Macroscopic Cross-Sector Performance Matrix ===")
        print(df_results.to_string(index=False))

        fig = go.Figure()
        for sector_name, curve in equity_curves.items():
            fig.add_trace(go.Scatter(x=valid_dates, y=curve, mode='lines', name=sector_name, line=dict(width=2)))
            
        fig.update_layout(title="Cross-Sector Structural Alpha (Net of 5bps Slippage & 2% Borrow)",
                          xaxis_title="Date", yaxis_title="Cumulative Dollar P&L",
                          hovermode="x unified", template="plotly_white", height=800)
        fig.show()

    finally:
        rd.close_session()

if __name__ == "__main__":
    run_cross_sector_analysis()