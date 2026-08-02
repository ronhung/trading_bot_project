"""
Visualize backtest_trades.csv produced by backtest_engine.exe
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TRADES = os.path.join(PROJECT_ROOT, "data", "historical_data", "backtest_trades.csv")


def plot_results(trades_path: str, out_png: str | None = None):
    if not os.path.exists(trades_path):
        raise FileNotFoundError(f"Trades CSV not found: {trades_path}")

    df = pd.read_csv(trades_path)
    if df.empty:
        print("No trades to plot.")
        return

    df["datetime"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
    df = df.sort_values("datetime")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(df["datetime"], df["balance_after"], color="#1f6f4a", linewidth=1.4)
    axes[0].set_ylabel("Equity (USDT)")
    axes[0].set_title("Backtest Equity Curve")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["datetime"], df["position_after"], color="#0b3d5c", linewidth=1.2)
    axes[1].set_ylabel("Position (BTC)")
    axes[1].grid(True, alpha=0.3)

    closes = df[df["reason"] == "close"]
    if not closes.empty:
        colors = ["#1f6f4a" if p >= 0 else "#a33b2b" for p in closes["pnl"]]
        axes[2].bar(closes["datetime"], closes["pnl"], color=colors, width=0.02)
    axes[2].axhline(0, color="#444", linewidth=0.8)
    axes[2].set_ylabel("Trade PnL")
    axes[2].set_xlabel("Time")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()

    if out_png is None:
        out_png = os.path.join(os.path.dirname(trades_path), "backtest_equity.png")
    fig.savefig(out_png, dpi=140)
    print(f"✅ Chart saved: {out_png}")
    print(f"   Trades: {len(df)} | Final balance: {df['balance_after'].iloc[-1]:.2f}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot C++ backtest trade report")
    parser.add_argument("--trades", default=DEFAULT_TRADES)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    plot_results(args.trades, args.out)


if __name__ == "__main__":
    main()
