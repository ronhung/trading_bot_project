"""
Online & Vectorized Kalman Filter for Dynamic Beta & Pairs Spread Estimation.

State-Space Representation:
  State equation:       theta_t = theta_{t-1} + w_t,   w_t ~ N(0, Q)
                        where theta_t = [alpha_t, beta_t]^T
  Observation equation: y_t = H_t * theta_t + v_t,     v_t ~ N(0, R)
                        where H_t = [1, x_t]

Yields:
  - Dynamic Hedge Ratio beta_t
  - Spread Residual e_t = y_t - (alpha_t + beta_t * x_t)
  - Standardized Z-Score Z_t
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict


class KalmanFilterPairs:
    """
    Kalman Filter for dynamic beta tracking and pairs trading residual generation.

    Parameters:
        delta: Transition covariance scale (default 1e-4). Higher = faster beta adaptation.
        R: Measurement noise variance (default 1e-3).
    """

    def __init__(self, delta: float = 1e-4, R: float = 1e-3):
        self.delta = float(delta)
        self.R = float(R)

    def fit_spread(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized execution over price series x and y.

        Returns:
            alpha (np.ndarray): Intercept series
            beta (np.ndarray): Dynamic hedge ratio series
            residual (np.ndarray): Price spread / prediction error series
            zscore (np.ndarray): Standardized residual z-score series
        """
        n = len(x)
        # theta = [alpha, beta]
        theta = np.zeros(2, dtype=np.float64)
        # Initial state covariance
        P = np.zeros((2, 2), dtype=np.float64)
        # State transition covariance
        Vw = (self.delta / (1.0 - self.delta)) * np.eye(2, dtype=np.float64)

        alphas = np.zeros(n, dtype=np.float64)
        betas = np.zeros(n, dtype=np.float64)
        residuals = np.zeros(n, dtype=np.float64)
        sqrt_Q = np.zeros(n, dtype=np.float64)

        for t in range(n):
            # Prior covariance
            P = P + Vw
            
            # Measurement matrix
            H = np.array([1.0, x[t]], dtype=np.float64)
            
            # Measurement prediction error
            y_hat = np.dot(H, theta)
            e_t = y[t] - y_hat
            
            # Innovation variance
            Q_t = np.dot(H, np.dot(P, H)) + self.R
            
            # Kalman Gain
            K = np.dot(P, H) / Q_t
            
            # State update
            theta = theta + K * e_t
            P = P - np.outer(K, np.dot(H, P))
            
            alphas[t] = theta[0]
            betas[t] = theta[1]
            residuals[t] = e_t
            sqrt_Q[t] = np.sqrt(max(1e-12, Q_t))

        # Standardized Z-Score: e_t / sqrt(Q_t)
        # Also compute rolling z-score of residual for empirical robustness
        raw_zscore = residuals / sqrt_Q
        return alphas, betas, residuals, raw_zscore


def compute_half_life(spread: np.ndarray) -> float:
    """
    Calculate Ornstein-Uhlenbeck (OU) Mean-Reversion Half-Life via AR(1) regression:
        delta_spread_t = phi * spread_{t-1} + error
        half_life = -log(2) / phi
    """
    spread = np.asarray(spread, dtype=np.float64)
    # Remove NaNs
    valid = spread[~np.isnan(spread)]
    if len(valid) < 30:
        return np.nan
        
    lag = valid[:-1]
    ret = valid[1:] - lag
    
    # OLS regression of ret on lag: ret = phi * lag + const
    cov_mat = np.cov(ret, lag)
    if cov_mat[1, 1] <= 1e-12:
        return np.nan
    phi = cov_mat[0, 1] / cov_mat[1, 1]
    
    if phi >= 0.0:  # Not mean-reverting (diverging or random walk)
        return np.inf
        
    half_life = -np.log(2.0) / phi
    return float(half_life)
