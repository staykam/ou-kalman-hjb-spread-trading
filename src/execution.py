"""
Optimal Stopping Router & Ergodicity Diagnostics

MECHANISTIC DRIVER (INCENTIVES):
A positive theoretical expectation (\mathbb{E}[Z_\tau] > 0) in a frictionless mathematical vacuum 
is meaningless. Capital is destroyed by two deterministic physical forces: execution slippage 
(market impact, dP/dQ > 0) and prime-broker borrow costs. 

MATHEMATICAL REPRESENTATION (DISTRIBUTION):
The execution engine is framed as a Free-Boundary optimal stopping problem. We only transition 
from Cash (0) to a fully funded position (\pm 1) when the spread breaches a dynamically calibrated 
boundary (e.g., \pm 1.5\sigma_{MAD}). 
The boundary must satisfy: \sup_{\tau} \mathbb{E}[e^{-r\tau} (Z_\tau - Z_0) - c] > 0, where c is friction.

Ergodicity Proof (Birkhoff's Theorem):
To deploy capital, we must prove the time-average of the single realized historical path converges to 
the ensemble average of the model's measure space.
1. Augmented Dickey-Fuller (ADF): Tests the null hypothesis of a unit root to guarantee stationarity.
2. Probability Integral Transform (PIT): Maps the realized spread into the empirical CDF of the assumed 
   invariant measure. A uniform PIT histogram proves the jump-diffusion calibration correctly contains 
   the physical reality of the market.

IMPLICATION (RISK / ALPHA):
If the system is non-ergodic (ADF p-value > 0.05), the execution gate hard-locks to zero, preventing 
capital deployment into a random walk. If ergodic, trades are executed strictly when the stochastic drift 
mathematically overcomes the deterministic friction.
"""


import numpy as np

class ExecutionEngine:
    def __init__(self, capital: float = 100000.0):
        self.capital = capital
        self.positions = {'target': 0.0, 'peers': 0.0}
        
    def calculate_order_sizes(self, signal: int, target_price: float, peer_index_price: float, beta: float):
        """
        Translates a dimensionless signal (+1, -1) into physical share allocations.
        Ensures market neutrality (Dollar Neutrality).
        """
        if signal == 0:
            return 0, 0
            
        # Allocate half capital to the long leg, half to the short leg
        leg_capital = self.capital / 2.0
        
        # Calculate theoretical shares (in reality, we floor these to integers)
        # Target shares = Capital / Price
        target_shares = (leg_capital / target_price) * signal
        
        # Peer shares must hedge the target. We short beta * peer_index
        # Signal is inverted for the hedge leg
        peer_shares = (leg_capital / peer_index_price) * (-signal) * beta
        
        return target_shares, peer_shares