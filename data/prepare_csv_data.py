"""
Convert Parquet historical klines into a C++-friendly CSV for backtest_engine.

Output columns (no datetime — C++ parses timestamps directly):
  open_time,open,high,low,close,volume,close_time,quote_volume,
  trades_count,taker_buy_base,taker_buy_quote
"""
import os
import argparse
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PARQUET = os.path.join(PROJECT_ROOT, "data", "historical_data", "BTCUSDT_1m_full.parquet")
DEFAULT_CSV = os.path.join(PROJECT_ROOT, "data", "historical_data", "BTCUSDT_1m_full.csv")

CSV_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades_count",
    "taker_buy_base", "taker_buy_quote",
]


def prepare_csv(parquet_path: str, csv_path: str, start: str | None = None, end: str | None = None):
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    print(f"[prepare] Reading {parquet_path} ...")
    df = pd.read_parquet(parquet_path)

    if "open_time" not in df.columns:
        raise ValueError("Parquet missing open_time column")

    if start:
        start_ms = int(pd.Timestamp(start).timestamp() * 1000)
        df = df[df["open_time"] >= start_ms]
    if end:
        end_ms = int(pd.Timestamp(end).timestamp() * 1000)
        df = df[df["open_time"] <= end_ms]

    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Parquet missing columns: {missing}")

    out = df[CSV_COLUMNS].sort_values("open_time").drop_duplicates(subset=["open_time"])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    print(f"[prepare] Writing {len(out):,} rows -> {csv_path}")
    out.to_csv(csv_path, index=False)
    size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"[prepare] Done ({size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Prepare CSV for C++ backtest_engine")
    parser.add_argument("--parquet", default=DEFAULT_PARQUET)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--start", default=None, help="e.g. 2024-01-01")
    parser.add_argument("--end", default=None, help="e.g. 2024-06-01")
    args = parser.parse_args()
    prepare_csv(args.parquet, args.csv, args.start, args.end)


if __name__ == "__main__":
    main()
