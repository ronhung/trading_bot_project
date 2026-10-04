"""
Phase 4 reference backtest for the Adam trailing strategy.

Runs the vectorized lightweight_backtest with the SAME components the Phase 5
C++ engine drives (AdamBreakoutTrigger + trailing Donchian exit + FixedRiskSizer),
on the same CSV the C++ DataReplayer reads. This is the baseline that the C++
backtest report should match "same magnitude, same direction".

Usage:
  python scripts/adam_reference_backtest.py [--period N] [--trail N]
"""

import argparse
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd

from research.backtest import lightweight_backtest
from research.labeling import TrailingExitLabeler
from research.triggers.adam_breakout import AdamBreakoutTrigger
from execution.sizers import FixedRiskSizer


def load_csv(csv_path):
    df = pd.read_csv(csv_path)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", type=int, default=120, help="entry breakout lookback (bars)")
    parser.add_argument("--trail", type=int, default=60, help="trailing exit lookback (bars)")
    parser.add_argument("--risk-pct", type=float, default=0.10)
    parser.add_argument("--max-leverage", type=float, default=100.0)
    parser.add_argument("--capital", type=float, default=100000.0)
    parser.add_argument("--csv", default=os.path.join(_PROJECT_ROOT, "data", "historical_data", "BTCUSDT_1m_full.csv"))
    args = parser.parse_args()

    df = load_csv(args.csv)
    print(f"Data: {len(df):,} bars ({df['datetime'].iloc[0]} -> {df['datetime'].iloc[-1]})")

    trigger = AdamBreakoutTrigger(period=args.period)
    exit_labeler = TrailingExitLabeler(trail_period=args.trail)
    sizer = FixedRiskSizer(risk_pct=args.risk_pct, max_leverage=args.max_leverage)

    result = lightweight_backtest(
        df,
        trigger=trigger,
        exit_labeler=exit_labeler,
        position_sizer=sizer,
        indicator_params={"entry_period": args.period, "exit_period": args.trail,
                          "atr_period": args.period},
        initial_capital=args.capital,
        commission=0.0005,
    )

    print("\n=== Phase 4 reference (Adam trailing) ===")
    print(f"  Sharpe:        {result['sharpe']}")
    print(f"  Total Return:  {result['total_return_pct']:.2f}%")
    print(f"  Max Drawdown:  {result['max_dd_pct']:.2f}%")
    print(f"  Trades:        {result['n_trades']} (W:{result['n_wins']} L:{result['n_losses']})")
    print(f"  Win Rate:      {result['win_rate']*100:.1f}%")
    print(f"  Avg Bars Held: {result['avg_bars_held']}")
    print(f"  Final Capital: ${result['final_capital']:.2f}")

    if len(result["trades_df"]):
        t = result["trades_df"]
        print(f"\n  First 8 trades:")
        print(t[["side", "entry_idx", "exit_idx", "entry_price", "exit_price", "size", "pnl"]]
              .head(8).to_string())


if __name__ == "__main__":
    main()
