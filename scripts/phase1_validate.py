"""Phase 1 validation: do a trigger's events beat a random-entry baseline?

Usage:
  python scripts/phase1_validate.py [--all | --bollinger | --trend]
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from research.trigger_analysis import analyze_trigger
from research.triggers.bollinger_mean_reversion import BollingerMeanReversionTrigger
from research.triggers.trend_breakout import TrendFilteredBreakoutTrigger


def load_train():
    path = os.path.join(_PROJECT_ROOT, "data", "historical_data", "BTCUSDT_1m_full.parquet")
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    train = df[df["datetime"] < "2024-01-01"].reset_index(drop=True)
    print(f"train bars: {len(train):,} (2020-2023)")
    return train


def _report(name, r):
    print(f"\n=== {name} (n_long={r['n_long']}, n_short={r['n_short']}) ===")
    print(f"{'horizon':>10} | {'long':>10} {'base_long':>10} | {'short':>10} {'base_short':>10}")
    for k, cell in r["results"].items():
        lo = cell["long"]; sh = cell["short"]
        bl = cell["baseline_long"]; bs = cell["baseline_short"]
        print(f"{k:>10} | {lo['mean_return']:>+10.5f} {bl['mean_return']:>+10.5f} "
              f"| {sh['mean_return']:>+10.5f} {bs['mean_return']:>+10.5f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--bollinger", action="store_true")
    ap.add_argument("--trend", action="store_true")
    args = ap.parse_args()
    if not (args.all or args.bollinger or args.trend):
        args.all = True

    df = load_train()

    if args.all or args.bollinger:
        trig = BollingerMeanReversionTrigger(period=120, num_std=2.0)
        r = analyze_trigger(trig, df, mode="fixed_horizon",
                            horizons=(30, 60, 120, 240, 720), seed=0)
        _report("BollingerMeanReversion (period=120, k=2.0)", r)

    if args.all or args.trend:
        trig = TrendFilteredBreakoutTrigger(entry_period=7200, trend_period=28800)
        r = analyze_trigger(trig, df, mode="fixed_horizon",
                            horizons=(1440, 4320, 7200, 14400), seed=0)
        _report("TrendFilteredBreakout (entry=7200, trend=28800)", r)


if __name__ == "__main__":
    main()
