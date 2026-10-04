"""
Parity test: the C++ TrailingStop exit (faithful port) == Python TrailingExitLabeler.

The C++ TrailingStop lives in live_engine/src/core/trailing_stop.{h,cpp}. This
test ports its streaming algorithm line-for-line and checks it against the
vectorized reference (research.labeling.TrailingExitLabeler) on the same data
and entries. A mismatch here means the C++ exit would diverge from Python.
"""

import os
import sys
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from research.features import add_indicators
from research.labeling import TrailingExitLabeler


def _simulate_single_entry(lows, highs, closes, ev, side, indicator, hard_stop, period):
    """
    Faithful port of TrailingStop::on_bar()/observe()/on_entry().

    Per bar t > ev:
      trailing = min(low[t-period..t-1]) for donchian_low
               = max(high[t-period..t-1]) for donchian_high
      stop = max(hard_stop, extreme) for long / min(...) for short
      exit if low[t] <= stop (long) / high[t] >= stop (short)
      extreme = running max/min of trailing (never reverses)
    Returns the exit bar index, or None if still open at the last bar.
    """
    n = len(lows)
    extreme = hard_stop
    for t in range(ev + 1, n):
        lo = max(0, t - period)
        if indicator == "donchian_low":
            trailing = lows[lo:t].min()
        elif indicator == "donchian_high":
            trailing = highs[lo:t].max()
        else:  # moving_average
            trailing = closes[lo:t].mean() if t > lo else closes[t - 1]

        if side > 0:
            stop = max(hard_stop, extreme)
            if lows[t] <= stop:
                return t
            extreme = max(extreme, trailing)
        else:
            stop = min(hard_stop, extreme)
            if highs[t] >= stop:
                return t
            extreme = min(extreme, trailing)
    return None


def _make_data(n=2000, seed=7):
    rng = np.random.default_rng(seed)
    close = 40000.0 + np.cumsum(rng.normal(0, 120, n))
    close = np.maximum(close, 5000.0)
    high = close * (1.0 + rng.uniform(0.0005, 0.004, n))
    low = close * (1.0 - rng.uniform(0.0005, 0.004, n))
    volume = rng.lognormal(8, 0.5, n)
    return pd.DataFrame({
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def _compare(exit_period, seed=7, n=2000):
    data = _make_data(n=n, seed=seed)
    ind = add_indicators(data, exit_period=exit_period)

    lows = data["low"].values
    highs = data["high"].values
    closes = data["close"].values
    exit_low = ind["exit_low"].values
    exit_high = ind["exit_high"].values

    # Entries: spread long/short, all past the exit_period warmup.
    rng = np.random.default_rng(seed + 1)
    entries = sorted(set(int(x) for x in rng.integers(exit_period + 5, n - 50, 8)))
    sides = [1 if i % 2 == 0 else -1 for i in range(len(entries))]

    events = pd.Series(0, index=data.index)
    for ev, side in zip(entries, sides):
        events.iloc[ev] = side

    ref = TrailingExitLabeler(trail_period=exit_period).compute_labels(ind, events)
    mismatches = []
    for i, (ev, side) in enumerate(zip(entries, sides)):
        indicator = "donchian_low" if side > 0 else "donchian_high"
        hard_stop = (exit_low[ev] if side > 0 else exit_high[ev])

        # Reference exit (match TrailingExitLabeler's "still_open" convention).
        ref_row = ref.loc[ev]
        if ref_row["barrier_hit"] == "still_open":
            ref_exit = None
        else:
            ref_exit = int(ref_row["exit_idx"])

        got = _simulate_single_entry(lows, highs, closes, ev, side, indicator, hard_stop, exit_period)
        if got != ref_exit:
            mismatches.append((ev, side, ref_exit, got))

    return mismatches, entries, sides


def test_trailing_stop_parity():
    """C++ TrailingStop port must match TrailingExitLabeler exit bars."""
    for exit_period in (10, 30):
        mismatches, entries, sides = _compare(exit_period)
        assert not mismatches, (
            f"exit_period={exit_period}: {len(mismatches)} mismatches "
            f"(entry, side, ref_exit, got) = {mismatches}"
        )
    print(f"Parity OK: {len(entries)} entries x 2 exit_periods, all exit bars match.")


if __name__ == "__main__":
    for ep in (10, 30):
        mm, entries, sides = _compare(ep)
        print(f"exit_period={ep}: {len(entries)} entries, mismatches={len(mm)}")
        for ev, side, ref, got in mm:
            print(f"  MISMATCH ev={ev} side={side} ref={ref} got={got}")
    test_trailing_stop_parity()
    print("All trailing-stop parity checks passed.")
