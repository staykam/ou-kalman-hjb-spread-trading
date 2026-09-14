
"""
Adaptive State-Space Cointegration & Microstructure Noise Filtration

MECHANISTIC DRIVER (INCENTIVES):
Standard OLS regression (Engle-Granger) assumes infinite liquidity, static parameters, and normally 
distributed errors. Market structures are dynamical systems. The fundamental tether between companies 
drifts over time, and the daily price tape is heavily contaminated by market-maker bid-ask bounce.

MATHEMATICAL REPRESENTATION (DISTRIBUTION):
We model the cointegrating relationship as a Hidden Markov Model governed by an Adaptive Kalman Filter.
- State Equation (Structural Drift): \theta_t = \theta_{t-1} + w_t,  w_t \sim \mathcal{N}(0, Q)
- Observation Equation: y_t = x_t^T \theta_t + v_t,  v_t \sim \mathcal{N}(0, R_t)

Calibration via Maximum Likelihood Estimation (MLE):
We derive the baseline Noise-to-Signal ratio (Q/R) by minimizing the negative log-likelihood of the 
prediction error decomposition: L(\theta) = 0.5 \sum (\ln|S_t| + e_t^2 / S_t). 
If Q is optimized to a small scalar, the filter heavily penalizes sudden shifts, anchoring to the 
macro supply-chain mean.

Adaptive Mechanics:
1. Roll's Model: R_t is dynamically inflated using auto-covariance of returns to filter out bid-ask noise.
2. Mahalanobis Jump Gate: If the normalized innovation (e_t / \sqrt{S_t}) > 3.0, the structural topology 
   has broken. We instantly spike Q_t to force the filter to adapt, abandoning historical memory.

IMPLICATION (RISK / ALPHA):
The output spread Z_t represents the true geometric null space of the asset pair. By decoupling the 
slow-moving macroeconomic tether (Midterm Horizon: Weeks/Months) from the high-frequency market noise, 
we extract a clean, stationary I(0) signal.
"""

import numpy as np
from scipy.optimize import minimize

class MidTermGovernor:
    def __init__(self, dt: float = 1/252):
        self.dt = dt
        self.alpha_t = None
        self.beta_t = None
        self.e_t = None       
        self.S_t = None       
        
        # We will dynamically overwrite these via MLE
        self.opt_q = 1e-4 
        self.opt_r = 1e-3 

    def _kalman_negative_log_likelihood(self, params, target_prices, peer_index):
        """
        The Objective Function.
        We optimize the natural log of the parameters to strictly enforce positive variances.
        """
        q = np.exp(params[0])
        r = np.exp(params[1])
        
        T = len(target_prices)
        Q = np.eye(2) * q
        P = np.zeros((2, 2))
        theta = np.zeros(2) 
        
        nll = 0.0
        
        for t in range(T):
            x_t = np.array([1.0, peer_index[t]])
            y_t = target_prices[t]
            
            # A priori prediction
            P = P + Q
            y_hat = np.dot(x_t, theta)
            e_t = y_t - y_hat
            
            # System Variance
            S_t = np.dot(x_t, np.dot(P, x_t)) + r
            
            # Accumulate the log-likelihood (skip first 10 days for numerical stabilization)
            if t > 10:
                nll += 0.5 * (np.log(S_t) + (e_t**2) / S_t)
                
            # A posteriori update
            K_t = np.dot(P, x_t) / S_t
            theta = theta + K_t * e_t
            
            # Joseph stabilized covariance update
            I_minus_Kx = np.eye(2) - np.outer(K_t, x_t)
            P = I_minus_Kx @ P @ I_minus_Kx.T + np.outer(K_t, K_t) * r
            
        return nll

    def calibrate_mle(self, target_prices: np.ndarray, peer_index: np.ndarray):
        """
        Finds the physical parameters of the structural relationship using 
        Maximum Likelihood on the burn-in sample (Year 1).
        """
        # Restrict calibration to the first 252 days to prevent look-ahead bias
        calib_len = min(252, len(target_prices))
        t_calib = target_prices[:calib_len]
        p_calib = peer_index[:calib_len]
        
        # Initial guess in log-space
        init_guess = [np.log(1e-4), np.log(1e-3)]
        
        # L-BFGS-B minimizes the negative log-likelihood
        res = minimize(self._kalman_negative_log_likelihood, init_guess, 
                       args=(t_calib, p_calib), method='L-BFGS-B')
        
        self.opt_q = np.exp(res.x[0])
        self.opt_r = np.exp(res.x[1])
        print(f"[MLE Calibration] Structural State Noise (Q): {self.opt_q:.2e} | Observation Noise (R): {self.opt_r:.2e}")

    def fit_kalman_cointegration(self, target_prices: np.ndarray, peer_index: np.ndarray) -> np.ndarray:
        # 1. Calibrate the physics engine automatically
        self.calibrate_mle(target_prices, peer_index)
        
        T = len(target_prices)
        Q_base = np.eye(2) * self.opt_q
        R_base = self.opt_r
        
        P = np.zeros((2, 2))
        theta = np.zeros(2) 
        
        self.alpha_t = np.zeros(T)
        self.beta_t = np.zeros(T)
        self.e_t = np.zeros(T)
        self.S_t = np.zeros(T)
        spread = np.zeros(T)
        
        target_returns = np.diff(target_prices, prepend=target_prices[0])
        
        for t in range(T):
            x_t = np.array([1.0, peer_index[t]])
            y_t = target_prices[t]
            
            # --- Roll's Model: Dynamic Observation Noise (R_t) ---
            if t > 5:
                cov_matrix = np.cov(target_returns[t-5:t], target_returns[t-6:t-1])
                roll_variance = max(0, -cov_matrix[0, 1])
            else:
                roll_variance = 0.0
            V_v = R_base + roll_variance
            
            # --- ADAPTIVE REGIME SHIFT LOGIC ---
            Q_t = Q_base.copy()
            if t > 20: 
                normalized_sq_error = (self.e_t[t-1]**2) / self.S_t[t-1]
                if normalized_sq_error > 9.0: 
                    # If structurally broken, inject a massive multiple of the baseline variance
                    Q_t = Q_t + np.eye(2) * (self.opt_q * 100)
            
            # Standard Prediction & Update
            P = P + Q_t
            y_hat = np.dot(x_t, theta)
            e_t = y_t - y_hat
            
            S_t = np.dot(x_t, np.dot(P, x_t)) + V_v
            K_t = np.dot(P, x_t) / S_t
            
            theta = theta + K_t * e_t
            I_minus_Kx = np.eye(2) - np.outer(K_t, x_t)
            P = I_minus_Kx @ P @ I_minus_Kx.T + np.outer(K_t, K_t) * V_v
            
            self.alpha_t[t] = theta[0]
            self.beta_t[t] = theta[1]
            self.e_t[t] = e_t
            self.S_t[t] = S_t
            spread[t] = y_t - (theta[0] + theta[1] * peer_index[t])
            
        return spread

    def fit_rolling_ou_process(self, local_spread: np.ndarray):
        Z_t = local_spread[1:]
        Z_t_minus_1 = local_spread[:-1]
        
        X = np.vstack([np.ones(len(Z_t_minus_1)), Z_t_minus_1]).T
        
        try:
            params = np.linalg.inv(X.T @ X) @ X.T @ Z_t
            a, b = params[0], params[1]
        except np.linalg.LinAlgError:
            return 0.0, 0.0, np.inf 
        
        if b >= 1 or b <= 0:
            return 0.0, 0.0, np.inf 
            
        theta_param = -np.log(b) / self.dt
        mu = a / (1 - b)
        
        # Revert to unscaled Robust MAD calibration because the Kalman output is now highly accurate
        epsilon = Z_t - (a + b * Z_t_minus_1)
        mad = np.median(np.abs(epsilon - np.median(epsilon)))
        robust_std = 1.4826 * mad + 1e-8 
        sigma = robust_std / np.sqrt(self.dt)
        
        half_life = np.log(2) / theta_param
        
        return mu, sigma, half_life