import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.tnic_universe import TNICUniverse

# Make sure this matches your file name
file_path = "data/raw/tnic2_data.txt" 

try:
    universe = TNICUniverse(tnic_data_path=file_path, target_year=2021)
    
    # --- GLASS BOX DEBUGGING ---
    # Let's peek at the first 5 unique GVKEYs to see their format
    sample_gvkeys = universe.network['gvkey1'].unique()[:5]
    print(f"\n[DEBUG] Sample GVKEYs in the dataset: {sample_gvkeys}")
    
    # Test both formats for Exxon Mobil
    padded_gvkey = "004503"
    stripped_gvkey = "4503"
    
    target_gvkey = None
    if padded_gvkey in universe.network['gvkey1'].values:
        target_gvkey = padded_gvkey
    elif stripped_gvkey in universe.network['gvkey1'].values:
        target_gvkey = stripped_gvkey
    else:
        print(f"\n[WARNING] Exxon not found. Defaulting to the first available firm: {sample_gvkeys[0]}")
        target_gvkey = sample_gvkeys[0]
        
    # --- THE MATH TEST ---
    print(f"\nExtracting similarity vector for GVKEY: {target_gvkey}")
    peers = universe.get_peer_weights(target_gvkey)
    
    print(f"Found {len(peers)} textual peers.")
    print("Top 5 closest competitors based on 10-K text:")
    # Sort descending to see the highest cosine similarity scores
    print(peers.sort_values(ascending=False).head())

except Exception as e:
    print(f"Error: {e}")