"""
Incremental Binance USDT-M futures historical data updater.

Efficiency rules:
  1. If parquet already exists, resume from the last open_time (skip downloaded months).
  2. Monthly archives for completed months; daily archives for the current month.
  3. Merge, dedupe, sort, rewrite parquet in place.
  4. Optionally refresh the C++ CSV via --prepare-csv.
"""
import os
import io
import zipfile
import argparse
import requests
import pandas as pd
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

SYMBOL = "BTCUSDT"
INTERVAL = "1m"
START_DATE = datetime(2020, 1, 1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "historical_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades_count",
    "taker_buy_base", "taker_buy_quote", "ignore",
]


def download_binance_zip(url: str) -> pd.DataFrame | None:
    print(f"  ↓ {url}")
    response = requests.get(url, timeout=120)
    if response.status_code == 404:
        print("    (404 skip)")
        return None
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            return pd.read_csv(f, header=0, low_memory=False)


def normalize_klines(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if len(df.columns) >= 12:
        df.columns = KLINE_COLS[: len(df.columns)]
    if "ignore" in df.columns:
        df = df.drop(columns=["ignore"])

    float_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    for col in ["open_time", "close_time", "trades_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df = df.dropna(subset=["open_time"])
    return df


def month_iter(start: datetime, end: datetime):
    cur = datetime(start.year, start.month, 1)
    last = datetime(end.year, end.month, 1)
    while cur <= last:
        yield cur.year, cur.month
        cur += relativedelta(months=1)


def day_iter(start: datetime, end: datetime):
    cur = start.date()
    last = end.date()
    while cur <= last:
        yield cur
        cur += relativedelta(days=1)


def load_existing(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path):
        return None
    print(f"📦 Existing parquet: {path}")
    df = pd.read_parquet(path)
    print(f"   rows={len(df):,} last_open_time={df['open_time'].max()}")
    return df


def update_klines(full_refresh: bool = False) -> str:
    kline_output = os.path.join(OUTPUT_DIR, f"{SYMBOL}_{INTERVAL}_full.parquet")
    existing = None if full_refresh else load_existing(kline_output)

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    resume_from = START_DATE
    chunks: list[pd.DataFrame] = []

    if existing is not None and len(existing) > 0:
        last_ms = int(existing["open_time"].max())
        resume_from = datetime.utcfromtimestamp(last_ms / 1000.0) + relativedelta(minutes=1)
        chunks.append(existing)
        print(f"⏩ Incremental resume from {resume_from.isoformat()} UTC")
    else:
        print("🆕 Full download from 2020-01-01")

    # Completed months use monthly archives; current month uses daily.
    current_month_start = datetime(now.year, now.month, 1)

    for year, month in month_iter(resume_from, now):
        month_dt = datetime(year, month, 1)
        if month_dt >= current_month_start:
            break
        # Skip months fully before resume_from month if we already have them
        if month_dt + relativedelta(months=1) <= resume_from:
            continue
        url = (
            f"https://data.binance.vision/data/futures/um/monthly/klines/"
            f"{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{year}-{month:02d}.zip"
        )
        df = download_binance_zip(url)
        if df is not None:
            chunks.append(normalize_klines(df))

    # Daily files for current month (and any gap after last monthly)
    daily_start = max(resume_from, current_month_start)
    for d in day_iter(daily_start, now):
        url = (
            f"https://data.binance.vision/data/futures/um/daily/klines/"
            f"{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{d.strftime('%Y-%m-%d')}.zip"
        )
        df = download_binance_zip(url)
        if df is not None:
            chunks.append(normalize_klines(df))

    if not chunks:
        print("No kline data downloaded.")
        return kline_output

    total = pd.concat(chunks, ignore_index=True)
    if "datetime" in total.columns:
        total = total.drop(columns=["datetime"])

    total = total.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")
    total.insert(0, "datetime", pd.to_datetime(total["open_time"].astype("int64"), unit="ms"))

    # Drop rows before resume only matters for cleanliness — keep full history.
    total.to_parquet(kline_output, index=False)
    print(f"✅ klines saved: {kline_output} ({len(total):,} rows)")
    return kline_output


def update_funding(full_refresh: bool = False) -> str:
    funding_output = os.path.join(OUTPUT_DIR, f"{SYMBOL}_funding_rate_full.parquet")
    existing = None if full_refresh else (pd.read_parquet(funding_output) if os.path.exists(funding_output) else None)

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    resume_from = START_DATE
    chunks: list[pd.DataFrame] = []

    if existing is not None and len(existing) > 0:
        time_col = "calc_time" if "calc_time" in existing.columns else existing.columns[1]
        last_ms = int(pd.to_numeric(existing[time_col], errors="coerce").max())
        resume_from = datetime.utcfromtimestamp(last_ms / 1000.0)
        chunks.append(existing)
        print(f"⏩ Funding resume from {resume_from.isoformat()} UTC")

    for year, month in month_iter(resume_from, now):
        url = (
            f"https://data.binance.vision/data/futures/um/monthly/fundingRate/"
            f"{SYMBOL}/{SYMBOL}-fundingRate-{year}-{month:02d}.zip"
        )
        df = download_binance_zip(url)
        if df is None:
            continue
        df = df.copy()
        time_col = "calc_time"
        rate_col = "last_funding_rate"
        if time_col not in df.columns:
            # older schema fallback
            time_col = df.columns[0]
            rate_col = df.columns[1]
        df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
        df["funding_rate"] = pd.to_numeric(df[rate_col], errors="coerce")
        df = df.dropna(subset=[time_col, "funding_rate"])
        df["datetime"] = pd.to_datetime(df[time_col], unit="ms")
        chunks.append(df[["datetime", time_col, "funding_rate"]].rename(columns={time_col: "calc_time"}))

    if not chunks:
        print("No funding data downloaded.")
        return funding_output

    total = pd.concat(chunks, ignore_index=True)
    total = total.sort_values("calc_time").drop_duplicates(subset=["calc_time"], keep="last")
    total.to_parquet(funding_output, index=False)
    print(f"✅ funding saved: {funding_output} ({len(total):,} rows)")
    return funding_output


def main():
    parser = argparse.ArgumentParser(description="Incrementally update Binance historical data")
    parser.add_argument("--full-refresh", action="store_true", help="Ignore existing parquet and redownload all")
    parser.add_argument("--klines-only", action="store_true")
    parser.add_argument("--funding-only", action="store_true")
    parser.add_argument("--prepare-csv", action="store_true", help="Also emit BTCUSDT_1m_full.csv for C++")
    parser.add_argument("--csv-start", default=None)
    parser.add_argument("--csv-end", default=None)
    args = parser.parse_args()

    if not args.funding_only:
        update_klines(full_refresh=args.full_refresh)
    if not args.klines_only:
        update_funding(full_refresh=args.full_refresh)

    if args.prepare_csv:
        from prepare_csv_data import prepare_csv
        parquet_path = os.path.join(OUTPUT_DIR, f"{SYMBOL}_{INTERVAL}_full.parquet")
        csv_path = os.path.join(OUTPUT_DIR, f"{SYMBOL}_{INTERVAL}_full.csv")
        prepare_csv(parquet_path, csv_path, args.csv_start, args.csv_end)


if __name__ == "__main__":
    main()
