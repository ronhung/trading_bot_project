"""
Trigger effectiveness analysis — forward-outcome event study (Phase 1 → 2 bridge).

Before Phase 2 labeling, check whether a trigger's events carry directional
information about the future. Two layers:

  evaluate_trigger(trigger, df, labeler)  — ONE labeler ("barrier") at a time.
  analyze_trigger(trigger, df, mode, ...) — parameterized sweep over a chosen
      barrier family, so the grid is easy to edit in one place.

The barrier SHOULD mirror the strategy's actual exit mechanism:
  FixedHorizonLabeler   → "exit after N bars"
  TripleBarrierLabeler  → "take-profit / stop-loss / timeout"

Works with ANY ``BaseEventTrigger`` + ANY ``BaseLabeler``; nothing is hardcoded
to a specific strategy or barrier.
"""

from typing import Optional

import numpy as np
import pandas as pd

from core.labeler import BaseLabeler
from core.trigger import BaseEventTrigger
from research.features import add_indicators
from research.labeling import FixedHorizonLabeler, TripleBarrierLabeler


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

# Preferred return column, in order. fixed_horizon -> "raw_return"/"y_norm";
# triple_barrier -> "actual_return". All are trade-side aware (positive = win).
_RETURN_COLS = ("actual_return", "raw_return", "y_norm")


def _extract_return_outcome(labels: pd.DataFrame):
    """Return (return_values, outcome_values_or_None) from a labeler output."""
    ret = None
    for col in _RETURN_COLS:
        if col in labels.columns:
            ret = labels[col].values
            break
    if ret is None:
        ret = np.full(len(labels), np.nan)
    outcome = labels["barrier_hit"].values if "barrier_hit" in labels.columns else None
    return ret, outcome


def _summarize(ret: np.ndarray, outcome: Optional[np.ndarray] = None) -> dict:
    """Aggregate trade-side returns (+ optional barrier outcome) into stats."""
    n = int(len(ret))
    s = {
        "n": n,
        "mean_return": float(np.nanmean(ret)) if n else np.nan,
        "hit_rate": float(np.nanmean(ret > 0)) if n else np.nan,
    }
    if outcome is not None and n:
        up = int((outcome == "upper").sum())
        lo = int((outcome == "lower").sum())
        to = int((outcome == "timeout").sum())
        s["upper"] = up
        s["lower"] = lo
        s["timeout"] = to
        s["win_rate"] = float(up / (up + lo)) if (up + lo) > 0 else np.nan
    return s


def _evaluate(
    sig: pd.Series,
    ind: pd.DataFrame,
    labeler: BaseLabeler,
    n_baseline: Optional[int],
    seed: int,
) -> dict:
    """Core: evaluate ONE labeler given precomputed signals + indicators."""
    event_idx = np.flatnonzero(np.asarray(sig.values) != 0)
    sides = np.asarray(sig.values)[event_idx].astype(int)

    labels = labeler.compute_labels(ind, sig)
    ret, outcome = _extract_return_outcome(labels)
    long_mask = sides == 1
    short_mask = sides == -1

    # Baseline: random non-event bars, evaluated BOTH as long and as short
    # (side-aware, so each side compares against its own mirror).
    n = len(ind)
    rng = np.random.default_rng(seed)
    n_bl = max(2000, len(event_idx)) if n_baseline is None else n_baseline
    candidates = np.arange(n)
    candidates = candidates[~np.isin(candidates, event_idx)]
    n_sample = min(n_bl, len(candidates)) if len(candidates) else 0
    sample = (
        rng.choice(candidates, size=n_sample, replace=False)
        if n_sample
        else np.array([], dtype=int)
    )

    def _baseline(side: int) -> dict:
        s = pd.Series(0, index=ind.index, dtype=int)
        if len(sample):
            s.iloc[sample] = side
        lbl = labeler.compute_labels(ind, s)
        r, o = _extract_return_outcome(lbl)
        return _summarize(r, o)

    return {
        "long": _summarize(ret[long_mask], outcome[long_mask] if outcome is not None else None),
        "short": _summarize(ret[short_mask], outcome[short_mask] if outcome is not None else None),
        "baseline_long": _baseline(1),
        "baseline_short": _baseline(-1),
    }


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def evaluate_trigger(
    trigger: BaseEventTrigger,
    df: pd.DataFrame,
    labeler: BaseLabeler,
    atr_period: int = 20,
    n_baseline: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """
    Evaluate ONE barrier/labeler for a trigger.

    Args:
        trigger: a ``BaseEventTrigger`` instance (any strategy).
        df: OHLCV DataFrame sorted chronologically.
        labeler: a ``BaseLabeler`` instance (``FixedHorizonLabeler`` or
            ``TripleBarrierLabeler``) — this is the "barrier" being tested.
        atr_period: ATR smoothing period for volatility normalization.
        n_baseline: random baseline sample size (default ``max(2000, n_events)``).
        seed: RNG seed (deterministic baseline).

    Returns:
        dict with keys ``trigger``, ``labeler``, and per-side results
        ``long`` / ``short`` / ``baseline_long`` / ``baseline_short``.
    """
    sig = trigger.generate_signals(df)
    ind = add_indicators(df, atr_period=atr_period)
    result = _evaluate(sig, ind, labeler, n_baseline, seed)
    result["trigger"] = type(trigger).__name__
    result["labeler"] = type(labeler).__name__
    return result


def analyze_trigger(
    trigger: BaseEventTrigger,
    df: pd.DataFrame,
    mode: str = "fixed_horizon",
    horizons=(1440, 4320, 7200, 14400, 43200),
    triple_grid=((0.02, -0.01, 1440), (0.02, -0.01, 4320), (0.05, -0.02, 1440)),
    tb_mode: str = "pct",
    atr_period: int = 20,
    n_baseline: Optional[int] = None,
    seed: int = 0,
) -> dict:
    """
    Parameterized barrier sweep over a chosen mode.

    Args:
        trigger: any ``BaseEventTrigger``.
        df: OHLCV DataFrame sorted chronologically.
        mode: ``"fixed_horizon"`` or ``"triple_barrier"``.
        horizons: grid for fixed-horizon mode (forward bars).
        triple_grid: grid for triple-barrier mode, as ``(upper, lower, horizon)``
            tuples (explicit list, not a Cartesian product).
        tb_mode: ``"pct"`` (percentage) or ``"atr"`` (ATR-multiplier barriers).
        atr_period: ATR smoothing period for normalization.
        n_baseline, seed: baseline sampling.

    Returns:
        dict::

            {
              "trigger", "mode", "n_long", "n_short",
              "results": { <grid label>: {long/short/baseline_long/baseline_short}, ... }
            }

    Signals and indicators are computed ONCE and reused across the whole grid.
    """
    sig = trigger.generate_signals(df)
    ind = add_indicators(df, atr_period=atr_period)

    event_idx = np.flatnonzero(np.asarray(sig.values) != 0)
    sides = np.asarray(sig.values)[event_idx].astype(int)

    if mode == "fixed_horizon":
        grid = [(f"h={h}", FixedHorizonLabeler(horizon=h)) for h in horizons]
    elif mode == "triple_barrier":
        grid = [
            (
                f"tp={u}/sl={l}/h={h}",
                TripleBarrierLabeler(
                    upper_barrier=u, lower_barrier=l, horizon=h, barrier_mode=tb_mode
                ),
            )
            for (u, l, h) in triple_grid
        ]
    else:
        raise ValueError(f"mode must be 'fixed_horizon' or 'triple_barrier', got {mode!r}")

    results = {
        label: _evaluate(sig, ind, labeler, n_baseline, seed)
        for label, labeler in grid
    }

    return {
        "trigger": type(trigger).__name__,
        "mode": mode,
        "n_long": int((sides == 1).sum()),
        "n_short": int((sides == -1).sum()),
        "results": results,
    }


if __name__ == "__main__":
    from research.dataset_builder import make_synthetic_ohlcv
    from research.triggers.adam_breakout import AdamBreakoutTrigger

    df = make_synthetic_ohlcv(n_bars=20000, seed=42)
    trigger = AdamBreakoutTrigger(period=1440)  # 1-day breakout -> plenty of events

    print("=== evaluate_trigger (single labeler) ===")
    for lbl in (FixedHorizonLabeler(horizon=1440),
                TripleBarrierLabeler(upper_barrier=0.02, lower_barrier=-0.01, horizon=1440)):
        r = evaluate_trigger(trigger, df, lbl, seed=0)
        print(f"  {r['labeler']:24s} long n={r['long']['n']} mean={r['long']['mean_return']:+.5f} "
              f"| baseline_long mean={r['baseline_long']['mean_return']:+.5f}")

    print("=== analyze_trigger (fixed_horizon) ===")
    r = analyze_trigger(trigger, df, mode="fixed_horizon", horizons=(720, 1440), seed=0)
    print(f"  trigger={r['trigger']} n_long={r['n_long']} n_short={r['n_short']}")
    for k, cell in r["results"].items():
        print(f"  {k:>6}: long mean={cell['long']['mean_return']:+.5f} "
              f"| base_long={cell['baseline_long']['mean_return']:+.5f} "
              f"| short mean={cell['short']['mean_return']:+.5f} "
              f"| base_short={cell['baseline_short']['mean_return']:+.5f}")

    print("=== analyze_trigger (triple_barrier) ===")
    r = analyze_trigger(trigger, df, mode="triple_barrier",
                        triple_grid=((0.02, -0.01, 720), (0.02, -0.01, 1440)), seed=0)
    for k, cell in r["results"].items():
        lo = cell["long"]
        print(f"  {k:>18}: long win_rate={lo['win_rate']:.3f} (upper/lower/timeout="
              f"{lo['upper']}/{lo['lower']}/{lo['timeout']}) "
              f"| base_long win_rate={cell['baseline_long']['win_rate']:.3f}")

    print("trigger_analysis self-test PASSED")
