"""
BollingerMeanReversionTrigger — intraday mean-reversion entry.

Thesis: at high frequency (hourly), BTC over-extends and rapidly reverts to its
short-term mean. Enter long when close dips below the lower Bollinger band
(oversold) and short when close pops above the upper band (overbought).

All bands use .shift(1) so bar i only sees bars < i (zero lookahead).
"""

import numpy as np
import pandas as pd

from core.trigger import BaseEventTrigger


class BollingerMeanReversionTrigger(BaseEventTrigger):
    """
    Mean-reversion trigger on Bollinger bands.

    Long:  close <  (ma - num_std * std)
    Short: close >  (ma + num_std * std)
    """

    def __init__(self, period: int = 120, num_std: float = 2.0, long_only: bool = True):
        self.period = period
        self.num_std = num_std
        self.long_only = long_only

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        close = data["close"].values
        ma = data["close"].rolling(self.period).mean().shift(1).values
        std = data["close"].rolling(self.period).std(ddof=0).shift(1).values
        lower = ma - self.num_std * std
        upper = ma + self.num_std * std

        long_event = (close < lower) & ~np.isnan(lower)
        short_event = (close > upper) & ~np.isnan(upper)

        result = pd.Series(0, index=data.index, dtype=int)
        result.loc[long_event] = 1
        if not self.long_only:
            result.loc[short_event] = -1
        return result


if __name__ == "__main__":
    import numpy as np
    n = 600
    rng = np.random.default_rng(1)
    # mean-reverting synthetic path (OU-like) — should fire both directions
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    close = 100.0 + 0.9 * (close - close.mean())  # pull back toward mean
    high = close + 0.5
    low = close - 0.5
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close,
                       "volume": np.full(n, 100.0)}, index=idx)
    trig = BollingerMeanReversionTrigger(period=60, num_std=2.0)
    sig = trig.generate_signals(df)
    print(f"long={int((sig == 1).sum())} short={int((sig == -1).sum())}")
    assert (sig.iloc[:60] == 0).all(), "lookahead leak"
    assert (sig == 1).sum() > 0 and (sig == -1).sum() > 0
    print("BollingerMeanReversionTrigger self-test PASSED")
