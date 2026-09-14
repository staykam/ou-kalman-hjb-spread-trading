import numpy as np
from joblib import Parallel, delayed

"""
MODULE: Intraday Microstructure & Liquidity Dynamics
----------------------------------------------------
RATIONALE:
The Mid-Term Governor identifies structural alpha. However, at the intraday level, 
finite liquidity causes discrete jumps (dP/dQ > 0). This module evaluates the 
intraday safety of the structural signal by modelling asymmetric tail risks and 
simulating jump-diffusion survival probabilities.

MATHEMATICAL ASSUMPTIONS:
1. Log Returns: Data inputs must be continuous logarithmic returns \Delta \ln(P_t).
2. Non-Gaussian Dependence: The joint distribution is modeled via an empirical copula 
   to capture asymmetric lower tail dependence \tau^L.
3. Jump-Diffusion: The spread Z_t follows an SDE disrupted by a Poisson jump process N_t.

MECHANISM:
This code utilizes CPU-bound multiprocessing (joblib) to simulate M independent paths 
of the stochastic differential equation, returning the probability measure of survival.
"""

"""
-------------------------------------------------------------------------------------------------
Jump-Diffusion Microstructure Engine & Ergodic Variance Calibration

MECHANISTIC DRIVER (INCENTIVES):
Smaller market-cap constituents within a large index suffer from finite liquidity depth. When 
institutional flow enters the market, the price does not move continuously; it gaps. Standard 
Ornstein-Uhlenbeck (OU) models assume continuous Gaussian diffusion, which drastically underestimates 
the probability of massive, sudden drawdowns (heavy tails). 

MATHEMATICAL REPRESENTATION (DISTRIBUTION):
We model the short-term (intraday to days) liquidity voids as a Jump-Diffusion SDE:
    dZ_t = \theta(\mu - Z_t)dt + \sigma dW_t + \xi dN_t
Where:
- \theta(\mu - Z_t)dt is the deterministic structural pull (mean reversion).
- \sigma dW_t is the continuous Brownian noise.
- \xi dN_t is a Poisson jump measure representing discrete liquidity shocks (e.g., earnings, macro data).

Variance Calibration (L1 vs L2 Space):
Because standard deviation (L2 norm) squares errors, a single Poisson jump will contaminate the rolling 
variance, rendering the optimal stopping bounds permanently too wide. We map the errors to an L1 space 
using Median Absolute Deviation (MAD), surgically extracting the true structural \sigma while isolating 
the jumps.

IMPLICATION (RISK / ALPHA):
By acknowledging the discontinuous nature of the tape, we size the expected mean reversion against the 
true structural diffusion, rather than the jump-contaminated noise. This allows the model to buy precisely 
when the market has overreacted to a liquidity vacuum.
"""

class IntradayMicrostructure:
    def __init__(self, dt: float = 1.0 / (252 * 390)):
        self.dt = dt

    def estimate_empirical_tail_dependence(self, target_returns: np.ndarray, peer_returns: np.ndarray, q: float = 0.05) -> float:
        """
        Calculates the non-parametric lower tail dependence \tau^L.
        """
        assert target_returns.shape == peer_returns.shape, "Return vectors must be perfectly aligned."
        T = target_returns.shape[0]
        
        # Map to uniform marginals [0,1]
        u_target = target_returns.argsort().argsort() / T
        v_peer = peer_returns.argsort().argsort() / T
        
        # Indicator for the lower tail subspace [0, q] x [0, q]
        lower_tail_indicator = (u_target <= q) & (v_peer <= q)
        
        joint_prob_mass = np.sum(lower_tail_indicator) / T
        
        # Heuristic check to prevent division by zero in zero-tail edge cases
        if q == 0: return 0.0
        
        tau_lower = joint_prob_mass / q
        return float(tau_lower)

    def _simulate_single_batch(self, batch_size: int, z_0: float, mu: float, theta: float, 
                               sigma: float, lambda_j: float, mu_j: float, sigma_j: float, 
                               horizon_steps: int) -> int:
        """
        Internal worker function for joblib. Simulates a batch of paths and returns 
        the count of successful mean-reverting paths.
        """
        Z = np.zeros((batch_size, horizon_steps))
        Z[:, 0] = z_0
        
        # Pre-compute stochastic increments
        dW = np.random.normal(0, np.sqrt(self.dt), size=(batch_size, horizon_steps - 1))
        jump_prob = lambda_j * self.dt
        dN = np.random.binomial(1, jump_prob, size=(batch_size, horizon_steps - 1))
        xi = np.random.normal(mu_j, sigma_j, size=(batch_size, horizon_steps - 1))
        
        # Euler-Maruyama integration
        for t in range(1, horizon_steps):
            drift = theta * (mu - Z[:, t-1]) * self.dt
            diffusion = sigma * dW[:, t-1]
            jump = xi[:, t-1] * dN[:, t-1]
            Z[:, t] = Z[:, t-1] + drift + diffusion + jump
            
        if z_0 > mu:
            successes = np.sum(np.any(Z <= mu, axis=1))
        else:
            successes = np.sum(np.any(Z >= mu, axis=1))
            
        return int(successes)

    def simulate_jump_diffusion_survival(self, z_0: float, mu: float, theta: float, 
                                         sigma: float, lambda_j: float, mu_j: float, 
                                         sigma_j: float, horizon_steps: int, 
                                         num_paths: int = 100000, n_jobs: int = -1) -> float:
        """
        Executes parallel Monte Carlo simulation of the Jump-Diffusion PIDE.
        n_jobs=-1 utilizes all available CPU cores automatically.
        """
        # Distribute the paths evenly across 16 logical batches (for 16 threads)
        n_batches = 16
        paths_per_batch = num_paths // n_batches
        
        # Fire parallel workers
        results = Parallel(n_jobs=n_jobs,verbose=1)(
            delayed(self._simulate_single_batch)(
                paths_per_batch, z_0, mu, theta, sigma, lambda_j, mu_j, sigma_j, horizon_steps
            ) for _ in range(n_batches)
        )
        
        total_successes = sum(results)
        survival_probability = total_successes / (n_batches * paths_per_batch)
        
        return float(survival_probability)