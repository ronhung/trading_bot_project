"""
AdamBreakoutTrigger — recent high/low breakout (Phase 1 event trigger).

A long event fires when `close` breaks above the highest high of the previous
`period` bars; a short event fires when `close` breaks below the lowest low of
the previous `period` bars. This is the classic Donchian breakout, without the
ATR intensity filter used by the Turtle trigger.

Example: `period=43200` at 1m bars = breakout of the 30-day high/low.
"""

import numpy as np
import pandas as pd

from core.trigger import BaseEventTrigger
from research.features import add_indicators


class AdamBreakoutTrigger(BaseEventTrigger):
    """
    Breakout of the recent high/low.

    Long:  close > highest high of the previous `period` bars.
    Short: close < lowest  low  of the previous `period` bars.

    Uses `add_indicators` (the single source of truth for indicator periods),
    which shifts the rolling extrema by one bar so row `i` only sees bars < `i`.
    """

    def __init__(self, period: int = 43200, signed: bool = True):
        """
        Args:
            period: lookback window in bars. 43200 = 30 days at 1m.
            signed: If True, returns {-1, 0, 1}. If False, returns boolean.
        """
        self.period = period
        self.signed = signed

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Scan for breakouts of the prior `period`-bar high/low.

        Args:
            data: OHLCV DataFrame sorted chronologically (oldest first).
                  Must contain at minimum: high, low, close.

        Returns:
            pd.Series with the same index as `data`.
            Values: 1 = long entry, -1 = short entry, 0 = no event.
        """
        # Reuse pre-computed entry_high/entry_low when the caller already ran
        # add_indicators (the streaming StrategyWrapper and lightweight_backtest
        # both do). Only recompute for raw OHLCV input.
        if "entry_high" not in data.columns or "entry_low" not in data.columns:
            data = add_indicators(data, entry_period=self.period)
        ind = data

        close = ind["close"].values
        entry_high = ind["entry_high"].values
        entry_low = ind["entry_low"].values

        long_event = (close > entry_high) & ~np.isnan(entry_high)
        short_event = (close < entry_low) & ~np.isnan(entry_low)

        if self.signed:
            result = pd.Series(0, index=ind.index, dtype=int)
            result.loc[long_event] = 1
            result.loc[short_event] = -1
        else:
            result = pd.Series(False, index=ind.index)
            result.loc[long_event | short_event] = True

        return result


if __name__ == "__main__":
    # Quick self-test with a deterministic synthetic series.
    n = 120
    close = np.concatenate([
        np.full(40, 100.0),
        np.linspace(100.0, 130.0, 20),   # strong up-move -> long breakout
        np.full(20, 130.0),
        np.linspace(130.0, 90.0, 20),    # strong down-move -> short breakout
        np.full(20, 90.0),
    ])
    high = close + 0.5
    low = close - 0.5
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    df = pd.DataFrame({
        "open": close, "high": high, "low": low, "close": close,
        "volume": np.full(n, 100.0),
    }, index=idx)

    trigger = AdamBreakoutTrigger(period=20)
    sig = trigger.generate_signals(df)

    print(f"long signals : {int((sig == 1).sum())}")
    print(f"short signals: {int((sig == -1).sum())}")
    assert (sig.iloc[:20] == 0).all(), "lookahead leak: signal fired during warmup"
    assert (sig == 1).sum() > 0, "expected at least one long breakout"
    assert (sig == -1).sum() > 0, "expected at least one short breakout"
    assert isinstance(trigger, BaseEventTrigger)
    print("AdamBreakoutTrigger self-test PASSED")
