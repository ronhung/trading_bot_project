"""
Kalman Filter Pairs Trading Arbitrage Trigger.

Quant Mechanism:
  1. Computes dynamic hedge ratio beta_t between Asset Y and Asset X via Online Kalman Filter.
  2. Generates the cointegrated spread residual: e_t = Y_t - (alpha_t + beta_t * X_t).
  3. Standardizes the residual into rolling Z-Score Z_t.
  4. Generates market-neutral pair signals:
     - Short Y / Long beta*X when Z_t >= +entry_z (Spread overpriced)
     - Long Y / Short beta*X when Z_t <= -entry_z (Spread underpriced)
     - Exit when |Z_t| <= exit_z (Mean-reversion complete)
     - Stop loss when |Z_t| >= stop_z (Structural cointegration breakdown)
"""

import numpy as np
import pandas as pd
from typing import Optional

from core.trigger import BaseEventTrigger
from core.strategy_factory import register_trigger
from research.models.kalman_filter import KalmanFilterPairs, compute_half_life


@register_trigger("kalman_pairs")
class KalmanPairsTrigger(BaseEventTrigger):
    """
    Kalman Filter Statistical Arbitrage Pairs Trigger.

    Parameters:
        delta: Kalman transition covariance scale (default 1e-4).
        R: Kalman measurement variance (default 1e-3).
        entry_z: Z-score threshold for entering pair arbitrage (default 2.0).
        exit_z: Z-score threshold for closing / profit-taking (default 0.2).
        stop_z: Z-score threshold for structural stop-loss (default 4.0).
        zscore_window: Rolling window for residual standardization (default 120 bars = 2h).
        cooldown: Minimum bars between entries (default 60).
    """

    def __init__(
        self,
        delta: float = 1e-4,
        R: float = 1e-3,
        entry_z: float = 2.0,
        exit_z: float = 0.2,
        stop_z: float = 4.0,
        zscore_window: int = 120,
        cooldown: int = 60,
        **kwargs,
    ):
        self.delta = float(delta)
        self.R = float(R)
        self.entry_z = float(entry_z)
        self.exit_z = float(exit_z)
        self.stop_z = float(stop_z)
        self.zscore_window = int(zscore_window)
        self.cooldown = int(cooldown)

    def generate_pair_signals(
        self,
        price_y: pd.Series,
        price_x: pd.Series,
    ) -> pd.DataFrame:
        """
        Generate pair trading signals and state diagnostics from price series Y and X.

        Returns DataFrame containing:
          - signal_y (+1, -1, 0)
          - signal_x (-1, +1, 0)
          - beta (dynamic hedge ratio)
          - spread (residual)
          - zscore
        """
        py_arr = price_y.values.astype(np.float64)
        px_arr = price_x.values.astype(np.float64)

        kf = KalmanFilterPairs(delta=self.delta, R=self.R)
        alphas, betas, residuals, raw_z = kf.fit_spread(px_arr, py_arr)

        res_series = pd.Series(residuals, index=price_y.index)
        roll_mean = res_series.rolling(self.zscore_window).mean()
        roll_std = res_series.rolling(self.zscore_window).std().replace(0, 1.0)
        zscores = ((res_series - roll_mean) / roll_std).bfill().values

        n = len(zscores)
        sig_y = np.zeros(n, dtype=int)
        sig_x = np.zeros(n, dtype=int)
        
        pos = 0  # +1 = long Y / short X; -1 = short Y / long X; 0 = flat
        last_sig_idx = -self.cooldown - 1

        for t in range(n):
            z = zscores[t]
            if np.isnan(z):
                continue

            if pos == 0:
                if t - last_sig_idx >= self.cooldown:
                    if z >= self.entry_z:
                        pos = -1  # Short Y, Long X
                        sig_y[t] = -1
                        sig_x[t] = 1
                        last_sig_idx = t
                    elif z <= -self.entry_z:
                        pos = 1   # Long Y, Short X
                        sig_y[t] = 1
                        sig_x[t] = -1
                        last_sig_idx = t
            elif pos == 1:
                # Long Y / Short X: exit when z reverts to exit_z or stop loss at -stop_z / +stop_z
                if z >= -self.exit_z or z <= -self.stop_z:
                    pos = 0
            elif pos == -1:
                # Short Y / Long X: exit when z reverts to exit_z or stop loss at stop_z
                if z <= self.exit_z or z >= self.stop_z:
                    pos = 0

        out_df = pd.DataFrame({
            "signal_y": sig_y,
            "signal_x": sig_x,
            "alpha": alphas,
            "beta": betas,
            "residual": residuals,
            "zscore": zscores,
        }, index=price_y.index)
        return out_df

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Single-asset interface compliance (defaults to signal_y if price_x present)."""
        if "close_x" in data.columns and "close_y" in data.columns:
            res = self.generate_pair_signals(data["close_y"], data["close_x"])
            return res["signal_y"]
        return pd.Series(0, index=data.index)
