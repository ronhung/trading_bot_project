"""
High-efficiency streaming downloader & Parquet converter for Binance aggTrades.

Design:
  1. Downloads monthly zip archives from Binance Vision directly into memory.
  2. Converts raw CSV streams into compact ZSTD-compressed Parquet files in chunks.
  3. Never writes massive raw CSVs to disk, saving 90%+ disk space.
  4. Resumes automatically by skipping already-converted Parquet files.
"""

import os
import sys
import io
import gc
import zipfile
import argparse
import urllib.request
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from dateutil.relativedelta import relativedelta

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SYMBOL = "BTCUSDT"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "historical_data", "aggtrades")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARQUET_SCHEMA = pa.schema([
    ("transact_time", pa.uint64()),
    ("price", pa.float32()),
    ("quantity", pa.float32()),
    ("is_buyer_maker", pa.bool_()),
])


def month_range(start_year: int, start_month: int, end_year: int, end_month: int):
    cur = datetime(start_year, start_month, 1)
    end = datetime(end_year, end_month, 1)
    while cur <= end:
        yield cur.year, cur.month
        cur += relativedelta(months=1)


def process_month(year: int, month: int, symbol: str = SYMBOL) -> str | None:
    out_file = os.path.join(OUTPUT_DIR, f"{symbol}-aggTrades-{year}-{month:02d}.parquet")
    if os.path.exists(out_file) and os.path.getsize(out_file) > 1024 * 1024:
        print(f"[Skip] {os.path.basename(out_file)} already exists ({os.path.getsize(out_file)/1024/1024:.1f} MB)")
        return out_file

    url = (
        f"https://data.binance.vision/data/futures/um/monthly/aggTrades/"
        f"{symbol}/{symbol}-aggTrades-{year}-{month:02d}.zip"
    )
    print(f"[Download] {year}-{month:02d} from {url} ...")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            content = resp.read()
    except Exception as e:
        print(f"   [Error] Failed to download {url}: {e}")
        return None

    print(f"   Converting {len(content)/1024/1024:.1f} MB zip to Parquet (streaming)...")
    temp_out = out_file + ".tmp"
    writer = None

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            csv_name = [name for name in z.namelist() if name.endswith(".csv")][0]
            with z.open(csv_name) as f:
                chunk_iter = pd.read_csv(
                    f,
                    header=0,
                    usecols=["price", "quantity", "transact_time", "is_buyer_maker"],
                    dtype={
                        "price": "float32",
                        "quantity": "float32",
                        "transact_time": "uint64",
                        "is_buyer_maker": "bool",
                    },
                    chunksize=1_000_000,
                )

                for chunk in chunk_iter:
                    table = pa.Table.from_pandas(chunk, schema=PARQUET_SCHEMA, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(temp_out, PARQUET_SCHEMA, compression="zstd")
                    writer.write_table(table)

        if writer is not None:
            writer.close()
            if os.path.exists(out_file):
                os.remove(out_file)
            os.rename(temp_out, out_file)
            size_mb = os.path.getsize(out_file) / 1024 / 1024
            print(f"   [OK] Saved {os.path.basename(out_file)} ({size_mb:.1f} MB)")
            return out_file
    except Exception as e:
        print(f"   [Error] Error processing {year}-{month:02d}: {e}")
        if os.path.exists(temp_out):
            try:
                os.remove(temp_out)
            except Exception:
                pass
        return None
    finally:
        del content
        gc.collect()


def download_range(start_year: int, start_month: int, end_year: int, end_month: int, symbol: str = SYMBOL):
    print(f"[Start] aggTrades download: {start_year}-{start_month:02d} to {end_year}-{end_month:02d} for {symbol}")
    success_count = 0
    total_count = 0
    for y, m in month_range(start_year, start_month, end_year, end_month):
        total_count += 1
        res = process_month(y, m, symbol)
        if res:
            success_count += 1
    print(f"[Done] Finished: {success_count}/{total_count} months ready in {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Binance aggTrades streaming downloader")
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--end-month", type=int, default=12)
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    args = parser.parse_args()

    download_range(args.start_year, args.start_month, args.end_year, args.end_month, args.symbol)
