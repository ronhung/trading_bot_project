"""
Parity test: IncrementalIndicators (O(1) streaming) == add_indicators (vectorized).

Feeds the same bars through both and asserts every produced column matches at
every bar (NaN == NaN), so the streaming brain's indicators are exactly the
vectorized reference.
"""

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from research.features import add_indicators
from research.features_incremental import IncrementalIndicators

COLUMNS = [
    "close", "entry_high", "entry_low", "exit_high", "exit_low", "atr",
    "vol_ratio", "vol_zscore", "channel_pos", "ret_5", "ret_10", "ret_30",
    "taker_buy_ratio", "ma",
]


def _make_data(n=2000, seed=7):
    rng = np.random.default_rng(seed)
    close = 40000.0 + np.cumsum(rng.normal(0, 120, n))
    close = np.maximum(close, 5000.0)
    high = close * (1.0 + rng.uniform(0.0005, 0.004, n))
    low = close * (1.0 - rng.uniform(0.0005, 0.004, n))
    volume = rng.lognormal(8, 0.5, n)
    taker_buy = volume * rng.uniform(0.3, 0.7, n)
    return pd.DataFrame({
        "open": close, "high": high, "low": low, "close": close,
        "volume": volume, "taker_buy_base": taker_buy,
    })


def _assert_close(name, ref, got, rtol=1e-9, atol=1e-9):
    ref = np.asarray(ref, dtype=float)
    got = np.asarray(got, dtype=float)
    ref_nan = np.isnan(ref)
    got_nan = np.isnan(got)
    if not np.array_equal(ref_nan, got_nan):
        diff = np.where(ref_nan != got_nan)[0]
        raise AssertionError(f"{name}: NaN mask mismatch at bars {diff[:5]}...")
    mask = ~ref_nan
    if mask.any() and not np.allclose(ref[mask], got[mask], rtol=rtol, atol=atol):
        bad = np.where(mask & ~np.isclose(ref, got, rtol=rtol, atol=atol))[0]
        raise AssertionError(
            f"{name}: value mismatch at {len(bad)} bars (first {bad[:5]}): "
            f"ref={ref[bad[:1]]} got={got[bad[:1]]}"
        )


def _run(entry, exit_, atr, vol, ma):
    data = _make_data()
    ref = add_indicators(data, entry_period=entry, exit_period=exit_,
                         atr_period=atr, vol_period=vol, ma_period=ma)

    inc = IncrementalIndicators(entry_period=entry, exit_period=exit_,
                                atr_period=atr, vol_period=vol, ma_period=ma)
    rows = []
    for _, bar in data.iterrows():
        rows.append(inc.update(bar.to_dict()))
    got = pd.DataFrame(rows)

    for col in COLUMNS:
        if col in ref.columns and col in got.columns:
            _assert_close(col, ref[col].values, got[col].values)


def test_incremental_matches_add_indicators():
    for entry, exit_, atr, vol, ma in [(20, 10, 20, 20, 200), (60, 30, 60, 30, 100)]:
        _run(entry, exit_, atr, vol, ma)
    print("IncrementalIndicators matches add_indicators across 2 param sets.")


if __name__ == "__main__":
    test_incremental_matches_add_indicators()
    print("All incremental-indicator parity checks passed.")
