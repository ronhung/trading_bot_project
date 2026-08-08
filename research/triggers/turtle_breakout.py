"""
TurtleBreakoutTrigger — Donchian channel breakout event detection.

Implements BaseEventTrigger. Wraps the existing vectorized trigger logic
from research.dataset_builder.make_turtle_breakout_trigger().
"""

import numpy as np
import pandas as pd

from core.trigger import BaseEventTrigger
from research.features import add_indicators


class TurtleBreakoutTrigger(BaseEventTrigger):
    """
    20-day (or custom) Donchian channel breakout detection.

    A breakout event occurs when close > entry_high (long) or
    close < entry_low (short), with an optional intensity filter
    measured in ATR units.

    This is the Phase 1 entry point for the Turtle strategy.
    """

    def __init__(
        self,
        entry_period: int = 20,
        atr_period: int = 20,
        atr_mult: float = 2.0,
        intensity_threshold: float = 0.0,
        signed: bool = True,
    ):
        """
        Args:
            entry_period: Donchian channel lookback (bars).
            atr_period: ATR smoothing period.
            atr_mult: ATR multiplier (used for stop computation downstream).
            intensity_threshold: Minimum breakout intensity in ATR units.
                0.0 = accept any breakout. 0.5 = require at least 0.5 ATR.
            signed: If True, returns {-1, 0, 1}. If False, returns boolean.
        """
        self.entry_period = entry_period
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.intensity_threshold = intensity_threshold
        self.signed = signed

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Scan for Donchian channel breakouts.

        Args:
            data: OHLCV DataFrame sorted chronologically.

        Returns:
            pd.Series with same index. 1=long, -1=short, 0=no event
            (if signed=True), else boolean.
        """
        # Ensure indicators are computed (uses add_indicators from features.py)
        ind = add_indicators(
            data,
            entry_period=self.entry_period,
            exit_period=max(self.entry_period // 2, 1),
            atr_period=self.atr_period,
        )

        close = ind["close"].values
        entry_high = ind["entry_high"].values
        entry_low = ind["entry_low"].values
        atr = ind["atr"].values

        # Guard: zero/NaN ATR → use 1.0 (matches turtle_math.py fallback)
        valid_atr = np.where((~np.isnan(atr)) & (atr > 0), atr, 1.0)

        # Breakout detection (vectorized)
        long_breakout = (close > entry_high) & (~np.isnan(entry_high))
        short_breakout = (close < entry_low) & (~np.isnan(entry_low))

        # Intensity filter
        long_intensity = (close - entry_high) / valid_atr
        short_intensity = (entry_low - close) / valid_atr

        long_event = long_breakout & (long_intensity > self.intensity_threshold)
        short_event = short_breakout & (short_intensity > self.intensity_threshold)

        if self.signed:
            result = pd.Series(0, index=ind.index, dtype=int)
            result.loc[long_event] = 1
            result.loc[short_event] = -1
        else:
            result = pd.Series(False, index=ind.index)
            result.loc[long_event | short_event] = True

        return result
