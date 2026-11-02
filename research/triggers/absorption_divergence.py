"""
Liquidity Absorption & Divergence Trigger (Iceberg Order Detection).

Microstructure Alpha targeting tops and bottoms:
1. Detects abnormal volume spikes: Volume_t > vol_mult * MA(Volume, vol_window).
2. Detects extreme aggressive taker bias:
   - Top: TakerRatio > taker_high (heavy market buying)
   - Bottom: TakerRatio < taker_low (heavy market selling)
3. Detects price stagnation (Iceberg absorption):
   - Top: Price fails to advance: (Close - Open) <= absorption_ratio * ATR, or long upper shadow.
   - Bottom: Price fails to drop: (Open - Close) <= absorption_ratio * ATR, or long lower shadow.
4. Generates Short (-1) at absorbed tops and Long (+1) at absorbed bottoms.
"""

import numpy as np
import pandas as pd

from core.trigger import BaseEventTrigger
from core.strategy_factory import register_trigger


@register_trigger("absorption_divergence")
class AbsorptionDivergenceTrigger(BaseEventTrigger):
    """
    Liquidity Absorption & Divergence Trigger.

    Parameters:
        vol_window: Lookback for volume moving average (default 60 = 1 hour).
        vol_mult: Multiplier for volume surge (default 2.5).
        taker_high: Threshold for extreme taker buying (default 0.70).
        taker_low: Threshold for extreme taker selling (default 0.30).
        absorption_ratio: Maximum real body as a fraction of ATR (default 0.20).
        atr_period: Lookback period for ATR (default 60).
        cooldown: Minimum bars between consecutive signals (default 60).
        allow_long: Allow bottom buying signals (default True).
        allow_short: Allow top selling signals (default True).
    """

    def __init__(
        self,
        vol_window: int = 60,
        vol_mult: float = 2.5,
        taker_high: float = 0.70,
        taker_low: float = 0.30,
        absorption_ratio: float = 0.20,
        atr_period: int = 60,
        cooldown: int = 60,
        allow_long: bool = True,
        allow_short: bool = True,
        regime_filter: bool = True,
        regime_window: int = 1440,
        regime_quantile: float = 0.35,
        **kwargs,
    ):
        self.vol_window = int(vol_window)
        self.vol_mult = float(vol_mult)
        self.taker_high = float(taker_high)
        self.taker_low = float(taker_low)
        self.absorption_ratio = float(absorption_ratio)
        self.atr_period = int(atr_period)
        self.cooldown = int(cooldown)
        self.allow_long = bool(allow_long)
        self.allow_short = bool(allow_short)
        self.regime_filter = bool(regime_filter)
        self.regime_window = int(regime_window)
        self.regime_quantile = float(regime_quantile)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"]
        open_p = data["open"]
        high = data["high"]
        low = data["low"]
        vol = data["volume"]

        # Calculate True Range & ATR
        tr = np.maximum(high - low, np.maximum((high - close.shift(1)).abs(), (low - close.shift(1)).abs()))
        atr = tr.rolling(self.atr_period).mean().bfill()

        # Volume Surge Condition
        vol_ma = vol.rolling(self.vol_window).mean()
        vol_surge = vol > (self.vol_mult * vol_ma)

        # Taker Ratio
        if "taker_ratio" in data.columns:
            taker_ratio = data["taker_ratio"]
        elif "taker_buy_base" in data.columns:
            safe_vol = vol.replace(0, np.nan)
            taker_ratio = (data["taker_buy_base"] / safe_vol).fillna(0.5)
        else:
            taker_ratio = pd.Series(0.5, index=data.index)

        # Absorption metrics
        candle_range = (high - low).replace(0, np.nan)
        upper_shadow = (high - np.maximum(open_p, close)) / candle_range
        lower_shadow = (np.minimum(open_p, close) - low) / candle_range
        body_abs = (close - open_p).abs()

        is_absorbed = (body_abs <= self.absorption_ratio * atr)

        # Long Signal: Extreme selling + price didn't fall / long lower shadow
        long_cond = (
            self.allow_long
            & vol_surge
            & (taker_ratio < self.taker_low)
            & (is_absorbed | (lower_shadow >= 0.40))
        )

        # Short Signal: Extreme buying + price didn't rise / long upper shadow
        short_cond = (
            self.allow_short
            & vol_surge
            & (taker_ratio > self.taker_high)
            & (is_absorbed | (upper_shadow >= 0.40))
        )

        # Regime filter: only active in compressed ranging conditions (ATR <= quantile)
        if self.regime_filter:
            atr_q = atr.rolling(self.regime_window).quantile(self.regime_quantile)
            regime_ok = (atr <= atr_q)
            long_cond = long_cond & regime_ok
            short_cond = short_cond & regime_ok

        signals = pd.Series(0, index=data.index, dtype=int)
        signals[long_cond] = 1
        signals[short_cond] = -1

        # Apply cooldown to avoid consecutive clustered signals
        if self.cooldown > 0:
            sig_arr = signals.values.copy()
            last_sig_idx = -self.cooldown - 1
            for i in range(len(sig_arr)):
                if sig_arr[i] != 0:
                    if i - last_sig_idx < self.cooldown:
                        sig_arr[i] = 0
                    else:
                        last_sig_idx = i
            signals = pd.Series(sig_arr, index=data.index)

        return signals
