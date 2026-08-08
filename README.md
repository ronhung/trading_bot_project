# Trading Bot Project — BTCUSDT Turtle Trend-Following System

A hybrid **C++ / Python** automated trading system running a **Donchian Channel (Turtle Trading)** strategy on Binance USDT-M futures.

Three execution modes, all sharing one strategy brain (`shared/core_logic/turtle_math.py`):

| Mode | Engine | When to use |
|------|--------|-------------|
| **Path A — Python Backtest** | Backtrader (`backtesting/run_backtest.py`) | Fast strategy prototyping & parameter sweeps |
| **Path B — C++ Backtest** | `backtest_engine.exe` + the same live Python bot | Rehearse the exact live code path |
| **Live Trading** | `live_engine.exe` + the same live Python bot | Paper/live on Binance Testnet |

### Strategy signals

`calculate_turtle_signals(df, entry_period, exit_period, atr_period, atr_mult)` returns a signal and a stop price:

| Signal | Meaning |
|--------|---------|
| `1` | Open Long — price broke above the `entry_period`-bar high |
| `-1` | Open Short — price broke below the `entry_period`-bar low |
| `2` | Close Long — price broke below the `exit_period`-bar low |
| `-2` | Close Short — price broke above the `exit_period`-bar high |
| `0` | No action |

---

## 1. Quick Reference

```bash
# 1) Install Python deps + build C++
pip install backtrader pandas numpy pyzmq requests matplotlib pyarrow
cd live_engine && mkdir build && cd build && cmake .. && cmake --build . --config Debug

# 2) Download data (incremental; add --full-refresh for a full redownload)
cd ../../data && python download_binance_data.py --prepare-csv

# 3) Path A backtest
cd ../backtesting && python run_backtest.py

# 4) Path B backtest — two terminals
#    T1:
cd ../live_strategy && python live_trend_bot.py --no-warmup
#    T2:
cd ../live_engine/build/Debug && backtest_engine.exe

# 5) Live (Testnet) — two terminals
#    T1:
cd live_engine/build/Debug && live_engine.exe
#    T2:
cd live_strategy && python live_trend_bot.py
```

> After any C++ change, rebuild with `cmake --build . --config Debug`. Stick to one config (Debug) to avoid running stale binaries.

---

## 2. Setup

### 2.1 Python dependencies

```bash
pip install backtrader pandas numpy pyzmq requests matplotlib pyarrow
```

### 2.2 Build the C++ engine

```bash
cd live_engine
mkdir build && cd build
cmake ..
cmake --build . --config Debug
```

Produces `backtest_engine.exe` and `live_engine.exe` in `live_engine/build/Debug`.
All C++ dependencies (nlohmann/json, IXWebSocket, libzmq + cppzmq, cpp-httplib, OpenSSL) are fetched automatically by CMake `FetchContent` — you only need CMake ≥ 3.14, a C++17 compiler (MSVC 2019+), and OpenSSL development headers.

### 2.3 Configuration — `shared/config.json`

```json
{
  "api_key": "YOUR_BINANCE_TESTNET_API_KEY",
  "secret_key": "YOUR_BINANCE_TESTNET_SECRET_KEY",
  "zmq": { "market_feed_port": 5555, "signal_port": 5556 },
  "database": { "path": "data/trading.db" },
  "backtest": { "initial_balance": 100000.0, "fee_rate": 0.0005, "slippage_bps": 1.0 }
}
```

Both executables and the Python bot read this file. The C++ engines auto-discover the project root by walking up from the executable path until they find `shared/config.json`.

- Live trading uses **Binance Futures Testnet** (`https://testnet.binancefuture.com`), hardcoded for orders. Get keys from [Binance Futures Testnet](https://testnet.binancefuture.com/).
- To switch to **Mainnet**, change the base URL to `https://fapi.binance.com` in every HTTP call site in `live_engine/src/live/binance_live_executor.cpp`.

### 2.4 Historical data

```bash
cd data
python download_binance_data.py --prepare-csv
```

| Flag | Meaning |
|------|---------|
| *(none)* | Incremental — resume from the last `open_time` in the existing parquet |
| `--full-refresh` | Ignore existing parquet and redownload everything from 2020-01-01 |
| `--prepare-csv` | After updating the parquet, also emit `BTCUSDT_1m_full.csv` for the C++ engine |
| `--klines-only` / `--funding-only` | Skip the other download |
| `--csv-start YYYY-MM-DD` / `--csv-end YYYY-MM-DD` | Filter the CSV output (with `--prepare-csv`) |

Outputs: `data/historical_data/BTCUSDT_1m_full.parquet`, plus `.csv` with `--prepare-csv`.

---

## 3. Backtesting

### 3.1 Path A — Python-only (Backtrader)

```bash
cd backtesting
python run_backtest.py
```

Tune in `run_backtest.py`:

```python
start_date = datetime.datetime(2023, 1, 1)
end_date   = datetime.datetime(2025, 1, 1)

cerebro.addstrategy(TurtleDonchianStrategy,
    entry_period=20,    # breakout lookback
    exit_period=10,     # exit lookback
    atr_period=20,
    atr_multiplier=2.0)

cerebro.broker.setcash(10000.0)
cerebro.broker.setcommission(commission=0.0005)
```

> The 1m dataset is large — start with a 3-month window, and comment out `cerebro.plot()` for full-dataset runs.

Reports Sharpe ratio, max drawdown, win rate, and return vs buy-and-hold, and simulates 8-hourly funding payments.

### 3.2 Path B — C++ engine + live Python brain

Path B uses the **same IPC protocol, the same Python strategy process, and the same order-handling code as live trading**. Only the executor is swapped (`MockExecutor` for simulated fills vs `BinanceLiveExecutor` for real orders). A backtest that passes here is a true rehearsal of the live system.

```bash
# Terminal 1 — Python brain
# NOTE: --no-warmup is required so the strategy buffer is filled only by the
# historical CSV replay (mixing live REST bars into a replay would corrupt the
# Donchian channels at the start of the backtest).
cd live_strategy
python live_trend_bot.py --no-warmup

# Terminal 2 — C++ backtest
cd live_engine/build/Debug
backtest_engine.exe

# Terminal 3 — visualize (optional)
cd backtesting
python plot_results.py
```

What the engine does:
1. Waits for the Python brain to connect (ZMQ handshake).
2. Replays every CSV bar in sequence: publish → wait for ACK → check stop-loss → process any order.
3. `MockExecutor` simulates fills with configurable slippage (`slippage_bps`) and fees (`fee_rate`).
4. Writes `data/historical_data/backtest_trades.csv` (also used by `plot_results.py`).

Custom run:

```bash
backtest_engine.exe <csv_path> <trades_output_path> <initial_balance>
# e.g. backtest_engine.exe "C:\data\BTCUSDT_1m_full.csv" "C:\data\my_trades.csv" 50000.0
```

**Verify in the Python terminal:**
- `C++ state: pos=0.0000 | flat=True | signal=1` on the first breakout bar → BUY sent
- `C++ state: pos=X.XXXX | flat=False | signal=1` on later bars → `Long breakout but pos=X.XXXX (not flat); skip BUY`
- C++ should no longer print `[IPC] Already in position` for gated signals

**Path A vs Path B:**

| | Path A (Backtrader) | Path B (C++) |
|---|---|---|
| Engine | Python | C++ |
| Strategy | same `turtle_math.py` | same `turtle_math.py` |
| Execution | Backtrader broker | MockExecutor (slippage + fees) |
| Live-code parity | No | **Yes** |
| Funding sim | Yes (8h) | No |

---

## 4. Research Toolkit — ML Dataset Builder & Parameter Sweeps

Pure pandas/numpy research modules for systematic strategy development. No Backtrader dependency — signals are vectorized for speed.

```bash
cd research

# Build ML pretrain dataset (X, y) from historical data
python build_2024_dataset.py

# Run all self-tests
python -m research.labeling        # triple-barrier labeler
python -m research.features        # indicator precomputation + feature pipeline
python -m research.dataset_builder # build_ml_dataset() demo
python -m research.backtest        # vectorized backtest demo (~50-200x faster than Backtrader)
python -m research.param_sweep     # multiprocessing grid search demo
```

### 4.1 `build_ml_dataset()` — raw klines → (X, y)

```python
from research.dataset_builder import build_ml_dataset, make_turtle_breakout_trigger
from research.features import add_indicators, default_feature_pipeline

df = add_indicators(raw_data)
trigger = make_turtle_breakout_trigger(entry_period=20, atr_period=20, signed=True)
X, meta = build_ml_dataset(df, trigger, default_feature_pipeline(),
                            {"upper_barrier": 0.02, "lower_barrier": -0.01, "horizon": 288})
# X: DataFrame with 12 features + triple-barrier labels (1=TP, -1=SL, 0=timeout)
```

### 4.2 `lightweight_backtest()` — vectorized backtester

```python
from research.backtest import lightweight_backtest

result = lightweight_backtest(df, entry_period=20, exit_period=10,
                               atr_period=20, atr_mult=2.0)
# ~0.02s per run on synthetic data — suitable for large parameter sweeps
```

### 4.3 `run_parameter_sweep()` — multiprocessing grid search

```python
from research.param_sweep import run_parameter_sweep
from research.param_sweep import _backtest_target, _dataset_target

results = run_parameter_sweep(
    target_func=_backtest_target,
    param_grid={"entry_period": [10, 20, 40], "atr_mult": [1.5, 2.0, 3.0]},
    raw_data=raw_data,
    n_jobs=-1,
    rank_by="sharpe",
)
```

### 4.4 Phase 3 — ML Signal Analysis (XGBoost + IC + ML-filtered Backtest)

Train/test pipeline with fixed-horizon labeling and XGBoost ranking:

```bash
# 1. Build train (2020-2023) + test (2024) datasets
python research/build_train_test_dataset.py

# 2. Train XGBoost, evaluate Spearman IC + Decile staircase
python research/ml_analysis.py
# Output: xgb_model.json (saved model), Spearman IC, decile table

# 3. Backtest with ML filter — only enter breakouts the model approves
python research/backtest.py --real --year 2024 \
    --entry 28800 --exit 14400 --atr-period 28800 --atr-mult 4.0 \
    --ml-filter --ml-threshold 50
```

Key results (20-day Turtle, TP=4×ATR SL, 2024 out-of-sample):

| ML Threshold | Trades | Return | Sharpe |
|-------------|--------|--------|--------|
| None | 83 | -5% | 0.42 |
| pred > 50 | 55 | +102% | 0.92 |
| pred > 100 | 10 | +266% | 1.58 |

### 4.5 Module overview

| Module | Purpose |
|--------|---------|
| `research/labeling.py` | Triple-barrier + fixed-horizon labelers |
| `research/features.py` | `add_indicators()` + 14 feature callables |
| `research/dataset_builder.py` | `build_ml_dataset()` orchestrator + synthetic data |
| `research/backtest.py` | `lightweight_backtest()` + `--ml-filter` XGBoost integration |
| `research/param_sweep.py` | `run_parameter_sweep()` — grid expansion + `ProcessPoolExecutor` |
| `research/build_train_test_dataset.py` | One-shot 2020-2023/2024 dataset builder |
| `research/ml_analysis.py` | XGBoost regressor + Spearman IC + decile analysis |

---

## 5. Live Trading (Binance Testnet)

```bash
# Terminal 1 — C++ live engine
cd live_engine/build/Debug
live_engine.exe

# Terminal 2 — Python strategy
cd live_strategy
python live_trend_bot.py
```

What the engine does at startup:
1. Reads API keys from `shared/config.json`.
2. Queries initial USDT balance and BTCUSDT position from Testnet.
3. Requests a user-data `listenKey` and opens the private WebSocket (order/account updates).
4. Binds ZMQ PUB 5555 (market data) and PULL 5556 (commands).
5. Connects to the Binance public 1m kline WebSocket.
6. Starts the order monitor/watchdog thread.

The Python bot:
1. **Warmup:** fetches the last ~8 closed 1m klines from Binance REST (`GET /fapi/v1/klines`) to pre-fill the Donchian buffer, so the first real-time bar is evaluated immediately. The still-forming (in-progress) candle is dropped so no partial bar enters the buffer. Falls back to cold start if the REST call fails.
2. Connects ZMQ SUB 5555 / PUSH 5556.
3. On each closed bar, calls `calculate_turtle_signals()` and, if the state gate passes, sends an order signal to C++.

### Signal flow (one bar)

1. **Python gates on the injected C++ state** — every kline carries `current_position`, `available_balance`, `stop_price` from C++'s RiskManager (the authoritative source):
   - `BUY` / `SELL` only if `current_position ≈ 0` (flat)
   - `CLOSE_LONG` only if `current_position > 0`
   - `CLOSE_SHORT` only if `current_position < 0`
   - Invalid signals are logged and dropped — they never reach C++.
2. Python sends e.g. `{"action": "BUY", "symbol": "BTCUSDT", "price": 42100.0, "stop_price": 41800.0}`.
3. C++ `IpcServer::handle_message()` validates:
   - **Position guard:** refuses if already in a position (no pyramiding).
   - **Open-order guard:** refuses if `has_open_order()` (a NEW / PARTIALLY_FILLED order is pending).
   - **Sizing** (`RiskManager::calculate_target_size()`): stop-direction sanity, size = `(Balance × 2%) / |entry − stop|`, capped at 20× leverage, floored to 0.001 BTC.
4. `BinanceLiveExecutor` POSTs a LIMIT order with a unique `clientOrderId` (e.g. `BOT1700000060123_1`), parses the response, and registers it in `OrderTracker`.
5. **Order lifecycle:** the private WS `ORDER_TRADE_UPDATE` events update status (`NEW → PARTIALLY_FILLED → FILLED`, or `CANCELED` / `EXPIRED` / `REJECTED`). On terminal states an `order_update` message is published to Python. `ACCOUNT_UPDATE` events are the **single** source of truth for RiskManager balance/position.
6. **Watchdog** (every 10 s): cancels orders stuck in NEW / PARTIALLY_FILLED for > 3 min, then re-prices the remainder at the current market price (capped at 2 reprice attempts, then reports `timeout_exhausted`).

### Risk parameters

- `live_engine/src/core/risk_manager.cpp` (constructor): `risk_pct` (default 2%), `max_leverage` (default 20×).
- `live_strategy/live_trend_bot.py` (`__init__`): `entry_period`, `exit_period`, `atr_period`, `atr_mult`.
- `live_engine/src/live/binance_live_executor.h`: `kOrderTimeout` (3 min), `kMaxRepriceAttempts` (2).

**Verify in the bot terminal:**
- `C++ state: pos=0.0000 | bal=... | stop=0.0000 | flat=True` on a breakout bar → BUY/SELL sent
- `C++ state: pos=X.XXXX | flat=False` on a breakout bar → `skip BUY` / `skip SELL` (gating works)
- On a fill: an `order_update` line `[Order] <cid>: FILLED ...` and a `[Radar] ... position updated` line
- On a 3-min timeout: `[Order] <cid>: CANCELED ... reason=timeout` (exactly once per order)

---

## 6. IPC Protocol (C++ ↔ Python over ZMQ)

| Direction | Pattern | Port | Content |
|-----------|---------|------|---------|
| C++ → Python | PUB | 5555 | `kline` and `order_update` JSON |
| Python → C++ | PULL | 5556 | order signals and `ack` |

Message shapes:

- **kline** (C++ → Python):
  ```json
  {"type":"kline", "symbol":"BTCUSDT", "open_time":1700000000000, "close_time":1700000059999,
   "open":42000.0, "high":42100.0, "low":41950.0, "close":42080.0, "volume":12.5,
   "quote_volume":525000.0, "taker_buy_base":6.2, "taker_buy_quote":260000.0,
   "trades_count":340, "is_closed":true,
   "current_position":0.123, "available_balance":9850.5, "stop_price":41800.0}
  ```
  `current_position`, `available_balance`, `stop_price` are read from RiskManager (mutex-protected) at publish time.
- **order_update** (C++ → Python, on terminal states):
  ```json
  {"type":"order_update", "symbol":"BTCUSDT", "client_order_id":"BOT1700000060123_1",
   "order_id":836214097, "side":"BUY", "order_type":"LIMIT", "quantity":0.123,
   "price":42100.0, "status":"FILLED", "reduce_only":false, "reason":""}
  ```
  `reason` is `"timeout"` or `"timeout_exhausted"` for watchdog cancellations.
- **signal** (Python → C++): `{"action":"BUY|SELL|CLOSE_LONG|CLOSE_SHORT", "symbol":"BTCUSDT", "price":42100.0, "stop_price":41800.0, "timestamp":1700000060.123}`
- **ack** (Python → C++, backtest sync only): `{"type":"ack"}`

Behaviour:
- **Backtest (sync):** publish one bar, block until ACK — every bar is processed in order; orders are applied before the next bar, so the injected state reflects the post-trade position.
- **Live (async):** a background PULL thread reads signals; order-update JSON is queued thread-safely from any thread and drained on the main loop via `pump_order_updates()` (~250 ms) because the PUB socket is not thread-safe.
- **Handshake (backtest only):** the engine pings until the brain ACKs; warmup pings carry `is_closed=false` (Python skips them) but still include the current RiskManager state.

---

## 7. Directory Structure (key files)

```
shared/config.json                     API keys, ZMQ ports, backtest params
shared/core_logic/turtle_math.py       THE strategy brain (single source of truth)
data/download_binance_data.py          kline/funding downloader + CSV export
backtesting/run_backtest.py            Path A Backtrader backtest
backtesting/plot_results.py            Visualize Path B trade CSV
research/labeling.py                  triple-barrier labeler (vectorized + reference)
research/features.py                  add_indicators() + feature pipeline
research/dataset_builder.py           build_ml_dataset() + synthetic data
research/backtest.py                  lightweight_backtest() — ~50-200x faster than Backtrader
research/param_sweep.py               run_parameter_sweep() — multiprocessing grid search
research/build_2024_dataset.py        one-shot 2024 ML pretrain dataset
research/build_train_test_dataset.py  train (2020-2023) / test (2024) dataset builder
research/ml_analysis.py               XGBoost regressor + Spearman IC + decile analysis
live_engine/src/core/                  ipc_server, risk_manager, i_order_executor, thread_safe_queue
live_engine/src/backtest/              main_backtest, mock_executor, data_replayer
live_engine/src/live/                  main_live, binance_ws, binance_live_executor, order_tracker
live_strategy/live_trend_bot.py        Python brain (live + C++ backtest)
live_strategy/zmq_client.py            ZMQ SUB/PUSH client
```

---

## 8. Dependencies

**Python:** `backtrader`, `pandas`, `numpy`, `pyzmq`, `requests`, `matplotlib`, `pyarrow` / `fastparquet`.

**C++:** nlohmann/json, IXWebSocket, libzmq + cppzmq, cpp-httplib, OpenSSL — fetched automatically by CMake.

**System:** CMake ≥ 3.14, C++17 compiler, OpenSSL dev headers. Windows is the primary target; Linux/macOS work with minor path adjustments.
