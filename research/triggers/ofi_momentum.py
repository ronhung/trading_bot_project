"""
Order Flow Imbalance (OFI) Momentum Trigger.

Microstructure Alpha utilizing aggregated trade flow:
  1. Computes rolling Order Flow Imbalance (OFI) = (taker_buy - taker_sell) / total_volume.
  2. Confirms aggressive institutional flow: rolling OFI > ofi_threshold.
  3. Gated by short-term price breakout (Donchian high) and macro trend alignment (trend MA).
  4. Returns +1 (long), -1 (short), or 0 (no event).
"""

import numpy as np
import pandas as pd

from core.trigger import BaseEventTrigger
from core.strategy_factory import register_trigger


@register_trigger("ofi_momentum")
class OFIMomentumTrigger(BaseEventTrigger):
    """
    Microstructure Order Flow Imbalance Momentum Trigger.

    Parameters:
        entry_period: Lookback for price breakout (default 180 = 3h at 1m).
        ofi_window: Lookback for smoothing OFI (default 60 = 1h at 1m).
        ofi_threshold: Minimum OFI intensity threshold (default 0.05).
        trend_period: Macro trend moving average period (default 1440 = 24h at 1m).
        long_only: Whether to restrict to long entries (default True).
    """

    def __init__(
        self,
        entry_period: int = 180,
        ofi_window: int = 60,
        ofi_threshold: float = 0.05,
        trend_period: int = 1440,
        long_only: bool = True,
    ):
        self.entry_period = int(entry_period)
        self.ofi_window = int(ofi_window)
        self.ofi_threshold = float(ofi_threshold)
        self.trend_period = int(trend_period)
        self.long_only = bool(long_only)

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """Scan data and return event signals (+1, -1, 0)."""
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Shifted rolling breakout bounds
        entry_high = high.rolling(self.entry_period).max().shift(1)
        entry_low = low.rolling(self.entry_period).min().shift(1)
        trend_ma = close.rolling(self.trend_period).mean()

        # Microstructure OFI
        if "ofi" in data.columns:
            ofi_series = data["ofi"]
        elif "taker_buy_base" in data.columns and "volume" in data.columns:
            vol = data["volume"].replace(0, np.nan)
            taker_sell = vol - data["taker_buy_base"]
            ofi_series = (data["taker_buy_base"] - taker_sell) / vol
        else:
            # Fallback if no taker data: use close-to-close return direction
            ofi_series = np.sign(close.pct_change(fill_method=None).fillna(0))

        roll_ofi = ofi_series.rolling(self.ofi_window, min_periods=self.ofi_window // 2).mean()

        long_mask = (close > entry_high) & (roll_ofi > self.ofi_threshold) & (close > trend_ma)

        signals = pd.Series(0, index=data.index, dtype=np.int8)
        signals[long_mask] = 1

        if not self.long_only:
            short_mask = (close < entry_low) & (roll_ofi < -self.ofi_threshold) & (close < trend_ma)
            signals[short_mask] = -1

        return signals
