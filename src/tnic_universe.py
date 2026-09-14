"""
TNIC pairwise similarity universe. To get a better structural correlation between firms rather than typical sector codes based on the Hoberg-Phillips data paper

TNIC Topological Graph Subspace & Synthetic Peer Construction

MECHANISTIC DRIVER (INCENTIVES):
Standard industry classifications (GICS/SIC) are static, linear, and fail to capture the evolving 
nature of supply chains. Firms do not operate in categorical silos; they exist on a continuous 
product-market manifold. We reject standard labels and utilize textual analysis of 10-K filings 
to dynamically map the true competitive topology.

MATHEMATICAL REPRESENTATION (DISTRIBUTION):
We map the asset universe as a weighted bipartite graph. Let X be the text-based feature space 
of 10-K filings. The Hoberg-Phillips TNIC database provides the cosine similarity:
    S_{i,j} = (X_i \cdot X_j) / (||X_i|| ||X_j||)
For a target asset y, we extract the top N peers to form a localized subspace Z. 
We normalize the similarity scores via the L1 norm to construct a projection vector W.
The synthetic peer index is generated via the einsum operation: I_t = \sum_{k=1}^N W_k P_{k,t}

IMPLICATION (RISK / ALPHA):
By anchoring the target asset to a structurally bound synthetic peer index, we ensure that the 
resulting spread is driven by temporary liquidity imbalances rather than permanent fundamental 
divergences. 

REQUIRED TESTS / ASSUMPTIONS:
- Assumption: 10-K product descriptions accurately reflect physical supply-chain exposure.
- Test: Rank condition of the subspace. We assume the top N peers provide sufficient linear 
  independence to span the target's fundamental exposure.
"""

import numpy as np
import pandas as pd
from typing import Dict, List

class TNICUniverse:
    def __init__(self, tnic_data_path: str, target_year: int):
        """
        Initializes the TNIC network for a specific year to conserve RAM.
        """
        self.tnic_data_path = tnic_data_path
        self.target_year = target_year
        self.network = self._load_network_memory_safe()
        
        # We will populate this mapping dictionary later (RIC -> GVKEY)
        self.ric_to_gvkey: Dict[str, str] = {}

    def get_top_n_peers(self, target_ric: str, ric_to_gvkey: dict, gvkey_to_ric: dict, n_peers: int = 10):
        """
        Dynamically searches the TNIC textual graph using GVKEYs, and translates 
        the structurally closest peers back into executable RICs.
        """
        if self.network is None:
            self.load_data()
            
        target_gvkey = str(ric_to_gvkey.get(target_ric))
        if not target_gvkey or target_gvkey == 'None':
            raise ValueError(f"Target RIC {target_ric} not found in crosswalk dictionary.")
            
        # Search the graph using the correct GVKEY
        peers_1 = self.network[self.network['gvkey1'] == target_gvkey][['gvkey2', 'score']].rename(columns={'gvkey2': 'peer'})
        peers_2 = self.network[self.network['gvkey2'] == target_gvkey][['gvkey1', 'score']].rename(columns={'gvkey1': 'peer'})
        
        all_peers = pd.concat([peers_1, peers_2]).sort_values(by='score', ascending=False)
        
        # Filter the graph to ONLY keep peers that we can translate back into RICs for execution
        all_peers['peer_ric'] = all_peers['peer'].astype(str).map(gvkey_to_ric)
        mapped_peers = all_peers.dropna(subset=['peer_ric'])
        
        # Keep the Top N highest similarity scores
        top_n = mapped_peers.head(n_peers)
        peer_rics = top_n['peer_ric'].tolist()
        raw_scores = top_n['score'].values
        
        if len(peer_rics) == 0:
            raise ValueError(f"No mapped peers found for {target_ric}.")
        
        # Normalize the scores so they sum to 1.0 (L1 Norm) to create an index weight vector
        normalized_weights = raw_scores / np.sum(raw_scores)
        
        print(f"Dynamically mapped {len(peer_rics)} executable TNIC peers for {target_ric}.")
        return peer_rics, normalized_weights
        
    def _load_network_memory_safe(self) -> pd.DataFrame:
        """
        Streams the massive text file and only keeps rows for the target year.
        Uses optimized dtypes to prevent memory overflow.
        """
        print(f"Streaming TNIC database for year {self.target_year}...")
        
        # Define strict types to save RAM (strings for IDs, float32 for weights)
        dtypes = {
            'year': np.int16,
            'gvkey1': str,
            'gvkey2': str,
            'score': np.float32
        }
        
        # Read in chunks (since it's too big for Excel, it's big for memory)
        # Assuming the file is comma or tab separated. If tab, use sep='\t'
        chunk_iter = pd.read_csv(
            self.tnic_data_path, 
            sep=None,          # Auto-detect separator (tab or comma)
            engine='python',   
            dtype=dtypes,
            chunksize=100_000
        )
        
        filtered_chunks = []
        for chunk in chunk_iter:
            # Filter the chunk down to just our year before saving it
            target_chunk = chunk[chunk['year'] == self.target_year]
            filtered_chunks.append(target_chunk)
            
        network_df = pd.concat(filtered_chunks, ignore_index=True)
        print(f"Loaded {len(network_df)} peer connections for {self.target_year}.")
        return network_df

    def get_peer_weights(self, target_gvkey: str) -> pd.Series:
        """
        Extracts the similarity vector (w_i) for a specific firm.
        Returns a Pandas Series mapping gvkey2 -> score.
        """
        peers = self.network[self.network['gvkey1'] == target_gvkey]
        if peers.empty:
            raise ValueError(f"Firm {target_gvkey} not found in TNIC network.")
            
        # Return a series where the index is the peer GVKEY and the value is the score
        return peers.set_index('gvkey2')['score']

    def construct_peer_index(self, similarity_scores: np.ndarray, peer_prices: np.ndarray) -> np.ndarray:
        """
        Projects the peer prices onto the normalized weight vector.
        similarity_scores: (N,)
        peer_prices: (T, N)
        """
        assert similarity_scores.shape[0] == peer_prices.shape[1], "Dimension mismatch"
        assert np.all(similarity_scores >= 0), "Scores must be non-negative"
        
        total_weight = np.sum(similarity_scores)
        if total_weight == 0:
            raise ValueError("All similarity scores are zero.")
            
        normalized_weights = similarity_scores / total_weight
        
        # Matrix multiplication: (T, N) @ (N,) -> (T,)
        synthetic_index = np.einsum('ti,i->t', peer_prices, normalized_weights)
        return synthetic_index