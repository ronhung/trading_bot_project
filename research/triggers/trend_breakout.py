"""
TrendFilteredBreakoutTrigger — low-frequency trend-following entry.

Thesis: BTC trends at low frequency. A Donchian breakout is only taken in the
direction of the longer-term trend (close above/below a trend MA), which filters
out counter-trend whipsaws.

Long:  close > entry_period high  AND  close > trend MA
Short: close < entry_period low   AND  close < trend MA

All rolling windows use .shift(1) so bar i only sees bars < i (zero lookahead).
"""

import numpy as np
import pandas as pd

from core.trigger import BaseEventTrigger


class TrendFilteredBreakoutTrigger(BaseEventTrigger):
    """
    Donchian breakout gated by a trend MA filter.
    """

    def __init__(self, entry_period: int = 7200, trend_period: int = 28800, long_only: bool = True):
        self.entry_period = entry_period
        self.trend_period = trend_period
        self.long_only = long_only

    def generate_signals(self, data):
        # Streaming path: a single-bar dict of precomputed indicators -> scalar.
        if isinstance(data, dict):
            close = data["close"]
            entry_high = data.get("entry_high", np.nan)
            entry_low = data.get("entry_low", np.nan)
            trend_ma = data.get("ma", np.nan)
            long_e = (close > entry_high) and (close > trend_ma) \
                and not np.isnan(entry_high) and not np.isnan(trend_ma)
            if self.long_only:
                return 1 if long_e else 0
            short_e = (close < entry_low) and (close < trend_ma) \
                and not np.isnan(entry_low) and not np.isnan(trend_ma)
            return 1 if long_e else (-1 if short_e else 0)

        close = data["close"].values
        # Reuse pre-computed columns when present (streaming + lightweight_backtest
        # both pass an indicator frame); only compute for raw OHLCV input.
        if "entry_high" in data.columns and "entry_low" in data.columns and "ma" in data.columns:
            entry_high = data["entry_high"].values
            entry_low = data["entry_low"].values
            trend_ma = data["ma"].values
        else:
            entry_high = data["high"].rolling(self.entry_period).max().shift(1).values
            entry_low = data["low"].rolling(self.entry_period).min().shift(1).values
            trend_ma = data["close"].rolling(self.trend_period).mean().shift(1).values

        long_event = (
            (close > entry_high) & (close > trend_ma)
            & ~np.isnan(entry_high) & ~np.isnan(trend_ma)
        )
        short_event = (
            (close < entry_low) & (close < trend_ma)
            & ~np.isnan(entry_low) & ~np.isnan(trend_ma)
        )

        result = pd.Series(0, index=data.index, dtype=int)
        result.loc[long_event] = 1
        if not self.long_only:
            result.loc[short_event] = -1
        return result


if __name__ == "__main__":
    import numpy as np
    n = 40000
    rng = np.random.default_rng(2)
    close = 40000.0 + np.cumsum(rng.normal(0, 100, n))  # trending path
    close = np.maximum(close, 100.0)
    high = close + 200
    low = close - 200
    idx = pd.date_range("2023-01-01", periods=n, freq="1min")
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close,
                       "volume": np.full(n, 100.0)}, index=idx)
    trig = TrendFilteredBreakoutTrigger(entry_period=720, trend_period=2880)
    sig = trig.generate_signals(df)
    print(f"long={int((sig == 1).sum())} short={int((sig == -1).sum())}")
    assert (sig.iloc[:2880] == 0).all(), "lookahead leak"
    print("TrendFilteredBreakoutTrigger self-test PASSED")
