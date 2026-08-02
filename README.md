# Trading Bot Project — BTCUSDT Turtle Trend-Following System

A hybrid **C++ / Python** automated trading system that runs a **Donchian Channel (Turtle Trading)** strategy on Binance USDT-M futures. It supports three execution modes: Python-only backtesting (Backtrader), C++-driven backtesting (with a live Python strategy brain), and live paper/live trading on Binance Testnet.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Directory Structure](#directory-structure)
- [System Interactions](#system-interactions)
  - [The Shared Brain](#the-shared-brain)
  - [IPC Protocol (C++ ↔ Python via ZMQ)](#ipc-protocol-c--python-via-zmq)
  - [Data Flow Diagrams](#data-flow-diagrams)
- [Backtesting](#backtesting)
  - [Path A — Python-only Backtrader Backtest](#path-a--python-only-backtrader-backtest)
  - [Path B — C++ Backtest Engine (with Live Brain)](#path-b--c-backtest-engine-with-live-brain)
- [Testnet / Live Trading](#testnet--live-trading)
- [State Sync & Order Tracking](#state-sync--order-tracking)
- [Configuration](#configuration)
- [How to Improve Each Component](#how-to-improve-each-component)
- [Build & Run Cheat Sheet](#build--run-cheat-sheet)
- [Dependencies](#dependencies)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        SHARED BRAIN                                  │
│           shared/core_logic/turtle_math.py                           │
│    calculate_turtle_signals(df, entry, exit, atr, mult)              │
│    → signal (1/-1/2/-2/0) + stop_price                              │
└──────┬───────────────────────┬───────────────────────────────────────┘
       │                       │
       ▼                       ▼
┌──────────────┐   ┌────────────────────────────┐
│  BACKTESTING │   │     LIVE TRADING PATH       │
│  (Python)    │   │                              │
│              │   │  ┌──────────────────────┐   │
│ Backtrader   │   │  │ C++ live_engine.exe  │   │
│ Cerebro      │   │  │                      │   │
│ engine       │   │  │ BinanceWS ──► ZMQ ──►│   │
│              │   │  │ (market data)   PUB   │   │
│              │   │  │                      │   │
│              │   │  │ ZMQ ◄── Python       │   │
│              │   │  │ PULL   live_trend_   │   │
│              │   │  │        bot.py        │   │
│              │   │  │                      │   │
│              │   │  │ BinanceLiveExecutor  │   │
│              │   │  │ → Binance Testnet    │   │
│              │   │  └──────────────────────┘   │
│              │   │                              │
│              │   │  ┌──────────────────────┐   │
│              │   │  │ C++ backtest_engine  │   │
│              │   │  │                      │   │
│              │   │  │ CSV → ZMQ → Python   │   │
│              │   │  │        brain         │   │
│              │   │  │                      │   │
│              │   │  │ MockExecutor         │   │
│              │   │  │ (simulated fills)    │   │
│              │   │  └──────────────────────┘   │
│              │   │    C++ BACKTEST (Path B)     │
└──────────────┘   └────────────────────────────┘
```

The system is split into **four layers**:

| Layer | Language | Role |
|-------|----------|------|
| **Strategy Logic** | Python | Pure math — signal generation from OHLCV data, gated on C++ state |
| **Execution Engine** | C++ | Market data ingestion, order routing, order lifecycle tracking, risk management |
| **IPC Bridge** | ZMQ | PUB/PULL sockets linking C++ and Python (klines, order updates, ACKs) |
| **State Sync** | C++ → Python | Every kline carries live `current_position` + `available_balance` from RiskManager |

---

## Directory Structure

```
trading_bot_project/
│
├── shared/                          # 🔗 Shared across ALL components
│   ├── config.json                  # API keys, ZMQ ports, backtest parameters
│   └── core_logic/
│       └── turtle_math.py           # 🧠 THE strategy brain (single source of truth)
│
├── data/                            # 📊 Historical data pipeline
│   ├── download_binance_data.py     # Incremental Binance kline + funding downloader
│   ├── prepare_csv_data.py          # Parquet → CSV converter (for C++ backtest)
│   └── historical_data/             # Stored data & backtest outputs (gitignored)
│
├── backtesting/                     # 🧪 Python Backtrader backtest
│   ├── run_backtest.py              # Entry point — Cerebro engine runner
│   ├── data_feeds.py                # Custom Backtrader PandasData classes
│   ├── indicators.py                # Custom indicators (stub — extend here)
│   ├── plot_results.py              # Visualize C++ backtest trade CSV output
│   └── strategies/
│       ├── base_strategy.py         # Base class: funding, risk sizing, order logging
│       └── trend_following.py       # Turtle Donchian strategy implementation
│
├── live_engine/                     # ⚡ C++ high-performance engine
│   ├── CMakeLists.txt               # Build config (ZMQ, IXWebSocket, OpenSSL, httplib)
│   └── src/
│       ├── core/                    # Shared core (used by both live & backtest)
│       │   ├── i_order_executor.h   # Abstract executor interface + OrderStatusUpdate struct
│       │   ├── kline_data.h         # K-line data structure
│       │   ├── ipc_server.h/.cpp    # ZMQ IPC server (PUB klines + order updates, PULL commands)
│       │   ├── risk_manager.h/.cpp  # Position sizing & risk management (thread-safe)
│       │   └── thread_safe_queue.h  # Generic thread-safe queue with timed wait
│       │
│       ├── backtest/                # C++ backtest subsystem
│       │   ├── main_backtest.cpp    # Entry point → backtest_engine.exe
│       │   ├── mock_executor.h/.cpp # Simulated order execution (slippage + fees)
│       │   └── data_replayer.h/.cpp # CSV bar-by-bar replay engine
│       │
│       └── live/                    # Live trading subsystem
│           ├── main_live.cpp        # Entry point → live_engine.exe (WS handlers, watchdog, main loop)
│           ├── binance_ws.h/.cpp    # Binance public WebSocket (1m klines)
│           ├── binance_live_executor.h/.cpp  # REST orders + order monitor + cancel/reprice
│           └── order_tracker.h/.cpp # Order lifecycle tracker (NEW→FILLED→CANCELED, timeout detection)
│
└── live_strategy/                   # 🐢 Python live trading process
    ├── live_trend_bot.py            # Live Turtle bot (connects to C++ via ZMQ)
    └── zmq_client.py                # ZMQ client (SUB market data, PUSH order signals)
```

---

## System Interactions

### The Shared Brain

The file `shared/core_logic/turtle_math.py` contains a single function:

```python
calculate_turtle_signals(df, entry_period, exit_period, atr_period, atr_mult)
    → (signal, stop_price)
```

**Signal values:**
| Signal | Meaning |
|--------|---------|
| `1` | Open Long (price broke above `entry_period`-bar high) |
| `-1` | Open Short (price broke below `entry_period`-bar low) |
| `2` | Close Long (price broke below `exit_period`-bar low) |
| `-2` | Close Short (price broke above `exit_period`-bar high) |
| `0` | No action |

This function is **imported identically** by:
- `backtesting/strategies/trend_following.py` (Python Backtrader backtest)
- `live_strategy/live_trend_bot.py` (live trading + C++ backtest via ZMQ)

Any change to the strategy logic is automatically reflected everywhere.

### IPC Protocol (C++ ↔ Python via ZMQ)

The C++ engine and Python strategy process communicate via two **ZeroMQ** sockets:

| Direction | ZMQ Pattern | Default Port | Purpose |
|-----------|-------------|--------------|---------|
| C++ → Python | **PUB** | `5555` | Market data (kline JSON objects) |
| Python → C++ | **PULL** | `5556` | Order signals + ACK messages |

**Market data message (C++ → Python):**
```json
{
  "type": "kline",
  "symbol": "BTCUSDT",
  "open_time": 1700000000000,
  "close_time": 1700000059999,
  "open": 42000.0, "high": 42100.0, "low": 41950.0, "close": 42080.0,
  "volume": 12.5, "quote_volume": 525000.0,
  "taker_buy_base": 6.2, "taker_buy_quote": 260000.0,
  "trades_count": 340,
  "is_closed": true,
  "current_position": 0.123,
  "available_balance": 9850.50,
  "stop_price": 41800.0
}
```

> **State injection:** `current_position`, `available_balance`, and `stop_price` are read from C++'s `RiskManager` (mutex-protected) at publish time. Python uses these as the **absolute source of truth** for gating signals — it will not emit `BUY`/`SELL` unless `current_position ≈ 0`, and will not emit `CLOSE_LONG`/`CLOSE_SHORT` unless holding the matching direction. This eliminates the previous split-brain problem where Python blindly emitted signals and relied on C++ to silently drop invalid ones.

**Order update message (C++ → Python):**
```json
{
  "type": "order_update",
  "symbol": "BTCUSDT",
  "client_order_id": "BOT1700000060123_1",
  "order_id": 836214097,
  "side": "BUY",
  "order_type": "LIMIT",
  "quantity": 0.123,
  "price": 42100.0,
  "status": "FILLED",
  "reduce_only": false,
  "reason": ""
}
```

This message is published whenever an order reaches a terminal state (`FILLED`, `CANCELED`, `EXPIRED`, `REJECTED`), or when the watchdog cancels a timed-out order. `reason` is `"timeout"` for watchdog cancellations or `"timeout_exhausted"` when all reprice attempts are exhausted.

**Order signal message (Python → C++):**
```json
{
  "action": "BUY",
  "symbol": "BTCUSDT",
  "price": 42100.0,
  "stop_price": 41800.0,
  "timestamp": 1700000060.123
}
```

Valid actions: `BUY`, `SELL`, `CLOSE_LONG`, `CLOSE_SHORT`.

**ACK message (Python → C++):**
```json
{"type": "ack"}
```

- In **backtest mode** (sync): The C++ engine publishes one bar, then blocks until it receives an ACK from Python. This ensures the strategy processes every bar in order. Orders sent by Python are processed synchronously inside `recv_until_ack()` before the next bar is published, so the injected `current_position`/`available_balance` reflect the post-trade state.
- In **live mode** (async): A background thread continuously reads order signals from the PULL socket without blocking the market data publisher. Order status updates are queued thread-safely and drained on the main-loop thread (ZMQ sockets are not thread-safe) via `pump_order_updates()` every ~250 ms. The main loop also uses a timed `wait_and_pop(250ms)` so it can flush updates even when no new kline has arrived.

**Handshake (backtest only):** The C++ backtest engine pings the Python process repeatedly until an ACK arrives, ensuring the brain is connected before replay starts. Warmup pings carry `is_closed=false` (so Python does not evaluate them) but still include the current RiskManager state.

### Data Flow Diagrams

#### Live Trading Flow
```
Binance WS ──► C++ main_live ──► ZMQ PUB ──► Python live_trend_bot.py
  (klines)       thread           :5555         (SUB socket)
                    │                │
                    │       kline + current_position
                    │       + available_balance
                    │       + stop_price
                    │                │
                    │       ┌────────┴────────────────────┐
                    │       │ Python gates signal on      │
                    │       │ real C++ state:             │
                    │       │  - BUY/SELL only if flat    │
                    │       │  - CLOSE only if matching   │
                    │       └────────┬────────────────────┘
                    │                │
               ZMQ PULL ◄────────────┘
               :5556
                  │
          ┌───────┴──────────┐
          │ C++ handles       │
          │ message:          │
          │ RiskManager       │
          │ validates size    │
          │ has_open_order()? │
          │      │            │
          │ BinanceLive       │
          │ Executor sends    │
          │ LIMIT REST order  │
          │ (with clientOID)  │
          │      │            │
          │ OrderTracker      │
          │ registers order   │
          └──────┬────────────┘
                 │
    ┌────────────┴────────────┐
    │ Private WS (user data)  │
    │ ORDER_TRADE_UPDATE ──►  │
    │   FILLED/CANCELED/...   │
    │ ACCOUNT_UPDATE ──►      │
    │   balance/position      │
    │            │            │
    │ OrderTracker.on_update()│
    │ RiskManager updated     │
    │            │            │
    │ ZMQ PUB ◄──┘            │
    │ order_update message    │
    │ → Python logs status    │
    └─────────────────────────┘

    Watchdog Thread (every 10s):
      OrderTracker.get_timed_out_orders(3min)
        → cancel_order(clientOID)
        → reprice_order() at current market
        (capped at 2 reprice attempts)
```

#### C++ Backtest Flow
```
CSV file ──► DataReplayer ──► ZMQ PUB ──► Python live_trend_bot.py
               (bar by bar)     :5555         (same code as live!)
                                     ◄── ACK + order via ZMQ PULL :5556
                │
                ▼
          MockExecutor
          (simulated fill
           with slippage
           & fees)
                │
                ▼
          backtest_trades.csv
                │
                ▼
          plot_results.py
          (equity curve chart)
```

---

## Backtesting

The project has **two independent backtesting paths**. Both use the same strategy logic but differ in engine and purpose.

### Path A — Python-only Backtrader Backtest

**Entry point:** `backtesting/run_backtest.py`

**How it works:**
1. Loads Parquet kline data (1m) and funding rate data into Backtrader `PandasData` feeds.
2. Resamples 1m → 1d for daily Donchian channel computation.
3. Runs the `TurtleDonchianStrategy` inside Backtrader's `Cerebro` engine.
4. Backtrader's built-in broker simulates fills, commissions (0.05%), and cash management.
5. `BaseStrategy._check_funding_rate()` simulates 8-hourly funding payments.
6. Outputs: Sharpe ratio, max drawdown, win rate, total return vs buy-and-hold.

**Running:**
```bash
cd backtesting
python run_backtest.py
```

**Key parameters** (edit in `run_backtest.py`):
```python
start_date = datetime.datetime(2023, 1, 1)
end_date = datetime.datetime(2025, 1, 1)

cerebro.addstrategy(TurtleDonchianStrategy,
    entry_period=20,    # breakout lookback
    exit_period=10,     # exit lookback
    atr_period=20,
    atr_multiplier=2.0
)

cerebro.broker.setcash(10000.0)
cerebro.broker.setcommission(commission=0.0005)
```

⚠️ **Warning:** The 1m dataset is large. For initial testing, use a 3-month window. Comment out `cerebro.plot()` when running the full dataset to avoid memory issues.

**When to use Path A:**
- Quick strategy prototyping and parameter sweeps
- You need Backtrader's built-in analyzers (Sharpe, Drawdown, TradeAnalyzer)
- You're iterating on strategy logic frequently
- You don't need to test the exact live-execution code path

### Path B — C++ Backtest Engine (with Live Brain)

**Entry points:**
- C++ side: `live_engine/build/backtest_engine.exe`
- Python side: `live_strategy/live_trend_bot.py`

**Why this path exists:** The C++ backtest engine uses the **exact same IPC protocol, the exact same Python strategy process, and the exact same order-handling code** as live trading. Only the executor is swapped (`MockExecutor` vs `BinanceLiveExecutor`). This means a backtest that passes here is a true rehearsal of the live system.

**How it works:**
1. **Prepare CSV data:**
   ```bash
   cd data

   # Default: incremental update — resumes from the last open_time in the existing parquet
   python download_binance_data.py

   # Full redownload from 2020-01-01 (ignore existing parquet)
   python download_binance_data.py --full-refresh

   # Incremental update + also produce the C++ CSV (BTCUSDT_1m_full.csv)
   python download_binance_data.py --prepare-csv
   ```

   | Flag | Description |
   |------|-------------|
   | *(none)* | Incremental update — resumes from the last `open_time` in the existing parquet file. Only downloads new months/days. |
   | `--full-refresh` | Ignores existing parquet, redownloads all monthly + daily archives from 2020-01-01. |
   | `--prepare-csv` | After updating the parquet, also emits `data/historical_data/BTCUSDT_1m_full.csv` for the C++ backtest engine. |
   | `--klines-only` | Skip funding rate download. |
   | `--funding-only` | Skip kline download. |
   | `--csv-start YYYY-MM-DD` | Filter CSV output to a start date (only with `--prepare-csv`). |
   | `--csv-end YYYY-MM-DD` | Filter CSV output to an end date (only with `--prepare-csv`). |

   The output parquet is `data/historical_data/BTCUSDT_1m_full.parquet`; the C++ engine consumes the CSV produced by `--prepare-csv`.

2. **Build the C++ engine:**
   ```bash
   cd live_engine
   mkdir build && cd build
   cmake ..
   cmake --build . --config Debug
   ```
   This produces two executables: `backtest_engine.exe` and `live_engine.exe`.

3. **Start the Python strategy brain** (in one terminal):
   ```bash
   cd live_strategy
   python live_trend_bot.py
   ```
   The bot will wait for market data from C++.

4. **Start the C++ backtest** (in another terminal):
   ```bash
   cd live_engine/build/Debug
   backtest_engine.exe
   ```
   The engine will:
   - Wait for the Python brain to connect (handshake)
   - Replay every CSV bar in sequence
   - For each bar: publish → wait for ACK → check stop-loss → process any order
   - `MockExecutor` simulates fills with configurable slippage and fees
   - Output `backtest_trades.csv` with every trade record

5. **Visualize results:**
   ```bash
   cd backtesting
   python plot_results.py
   ```

**Custom parameters:**
```bash
backtest_engine.exe <csv_path> <trades_output_path> <initial_balance>
# Example:
backtest_engine.exe "C:\data\BTCUSDT_1m_full.csv" "C:\data\my_trades.csv" 50000.0
```

**Comparison of Path A vs Path B:**

| Aspect | Path A (Backtrader) | Path B (C++) |
|--------|---------------------|--------------|
| Engine | Python (Backtrader) | C++ (custom) |
| Strategy code | Same `turtle_math.py` | Same `turtle_math.py` |
| Execution model | Backtrader broker | MockExecutor (slippage + fees) |
| IPC | None (in-process) | ZMQ PUB/PULL |
| Live code parity | No | **Yes — exact same code path** |
| Speed | Slower (Python loop) | Faster (C++ loop) |
| Output | Console report | CSV + PNG chart |
| Funding simulation | Yes (8h settlement) | No (not yet) |

---

## Testnet / Live Trading

The system is designed to trade on **Binance Futures Testnet** by default. The endpoint is hardcoded to `https://testnet.binancefuture.com`.

### Step-by-Step

1. **Get API keys** from [Binance Futures Testnet](https://testnet.binancefuture.com/) and put them in `shared/config.json`:
   ```json
   {
     "api_key": "YOUR_TESTNET_API_KEY",
     "secret_key": "YOUR_TESTNET_SECRET_KEY",
     ...
   }
   ```

2. **Build the C++ live engine** (same build as backtest — both executables are produced):
   ```bash
   cd live_engine/build
   cmake ..
   cmake --build . --config Debug
   ```

3. **Start the C++ live engine** (terminal 1):
   ```bash
   live_engine.exe
   ```
   The engine will:
   - Read API keys from `shared/config.json`
   - Query initial account state (USDT balance, BTCUSDT position) from Binance Testnet
   - Request a user data listenKey for private WebSocket updates
   - Open ZMQ PUB on port 5555 and ZMQ PULL on port 5556
   - Connect to Binance public WebSocket for 1m kline stream
   - Wait for the Python strategy to connect

4. **Start the Python strategy** (terminal 2):
   ```bash
   cd live_strategy
   python live_trend_bot.py
   ```
   The bot will:
   - Connect ZMQ SUB → port 5555 (market data)
   - Connect ZMQ PUSH → port 5556 (order signals)
   - Buffer klines as they arrive
   - On each **closed** 1m bar, call `calculate_turtle_signals()`
   - Send order signals back to C++

5. **What happens on a signal:**
   - Python gates the signal against C++'s real position state first:
     - `BUY`/`SELL`: only emitted if `current_position ≈ 0` (flat)
     - `CLOSE_LONG`: only emitted if `current_position > 0`
     - `CLOSE_SHORT`: only emitted if `current_position < 0`
     - Invalid signals are logged and dropped — they never reach C++
   - Python sends e.g. `{"action": "BUY", "symbol": "BTCUSDT", "price": 42100.0, "stop_price": 41800.0}`
   - C++ `IpcServer::handle_message()` receives it and validates:
     - Position guard: refuses if `abs(current_position) > 0` (already in position)
     - Open-order guard: refuses if `has_open_order()` (pending NEW/PARTIALLY_FILLED order exists)
     - `RiskManager::calculate_target_size()` validates:
       - Stop direction sanity (long stop must be below entry, short stop above)
       - Position size = `(Balance × 2%) / |Entry − Stop|`
       - Caps notional at 20× leverage
       - Rounds to 0.001 BTC (Binance min qty step)
   - `BinanceLiveExecutor::send_order()` generates a unique `clientOrderId` (e.g., `BOT1700000060123_1`), includes it in the REST POST, parses the response, and registers the order with `OrderTracker`
   - **Order lifecycle (fully tracked):**
     1. Order sent → `OrderTracker` records it as `NEW` with timestamp
     2. Private WebSocket receives `ORDER_TRADE_UPDATE` events → `OrderTracker.on_order_update()` updates status + filled quantity
     3. On `FILLED`/`CANCELED`/`EXPIRED`/`REJECTED`: an `order_update` JSON message is queued and published to Python via ZMQ PUB
     4. **Watchdog thread** (every 10s): checks for orders stuck in `NEW` or `PARTIALLY_FILLED` for > 3 minutes → cancels the original order → re-prices at the current market price (capped at 2 reprice attempts, then reports `timeout_exhausted`)
     5. `ACCOUNT_UPDATE` events keep `RiskManager` balance/position in sync with the exchange

### Risk Parameters (editable in code)

In `live_engine/src/core/risk_manager.cpp` (constructor):
- `risk_pct`: 2% of balance per trade (default)
- `max_leverage`: 20× (capping mechanism)

In `live_strategy/live_trend_bot.py` (`__init__`):
- `entry_period`, `exit_period`, `atr_period`, `atr_mult` — strategy parameters

---

## State Sync & Order Tracking

### State Injection (C++ → Python)

Every kline JSON published by C++ includes three live fields from `RiskManager`:

| Field | Source | Description |
|-------|--------|-------------|
| `current_position` | `RiskManager::get_current_position()` | Signed BTC position (positive = long, negative = short, 0 = flat) |
| `available_balance` | `RiskManager::get_current_balance()` | USDT wallet balance (marked, including unrealized PnL in live mode; realized equity in backtest) |
| `stop_price` | `RiskManager::get_stop_price()` | Active stop-loss level (0 if no position) |

**Thread safety:** RiskManager getters are mutex-protected. In live mode, the private WebSocket thread updates balance/position via `ACCOUNT_UPDATE`, while the main-loop thread reads them in `publish_kline()`. In backtest mode, `MockExecutor` updates RiskManager synchronously inside `recv_until_ack()`, so the state in each kline reflects the post-trade state of the previous bar.

**Python consumption:** `live_trend_bot.on_handle_market_data()` reads these fields on every message and stores them as `self.current_position`, `self.available_balance`, and `self.current_stop`. `execute_strategy_logic()` uses them to gate signals before sending to C++.

### Order Lifecycle Tracking

```
send_order()
    │
    ├─► POST /fapi/v1/order (with clientOrderId)
    ├─► Parse response → OrderTracker.register_order()
    │       status: NEW
    │
    ▼
Private WS ORDER_TRADE_UPDATE
    │
    ├─► OrderTracker.on_order_update(cid, status, filled_qty)
    │       status: NEW → PARTIALLY_FILLED → FILLED
    │                    → CANCELED / EXPIRED / REJECTED
    │
    ├─► Terminal state? → fire callback → IpcServer.queue_order_update()
    │       → ZMQ PUB "order_update" message → Python logs
    │
    ▼
Watchdog Thread (every 10s)
    │
    ├─► OrderTracker.get_timed_out_orders(180000)  // 3 min
    │
    ├─► Timed out + reprice_attempts < 2?
    │       → cancel_order(clientOrderId)
    │       → publish CANCELED + reason="timeout"
    │       → reprice_order() → new clientOrderId, reprice_attempts++
    │
    └─► Timed out + reprice_attempts >= 2?
            → cancel_order(clientOrderId)
            → publish CANCELED + reason="timeout_exhausted"
            → no more repricing
```

**Key design decisions:**
- **clientOrderId** (not server `orderId`) is the primary correlation key between REST responses and WebSocket events. This avoids needing to parse the REST response body at all.
- **Watchdog uses a dedicated thread** with a 10-second condition-variable wait so it can be woken for clean shutdown.
- **ZMQ PUB is only written from the main-loop thread.** Order status updates are queued via `ThreadSafeQueue<std::string>` from any thread and drained by `pump_order_updates()` in the main loop.
- **Deduplication:** `reported_terminal` flag prevents duplicate `order_update` messages when both the watchdog cancel and the subsequent WebSocket event fire for the same order.

---

## Configuration

`shared/config.json`:
```json
{
  "api_key": "YOUR_BINANCE_API_KEY",
  "secret_key": "YOUR_BINANCE_SECRET_KEY",
  "zmq": {
    "market_feed_port": 5555,
    "signal_port": 5556
  },
  "database": {
    "path": "data/trading.db"
  },
  "backtest": {
    "initial_balance": 100000.0,
    "fee_rate": 0.0005,
    "slippage_bps": 1.0
  }
}
```

Both the C++ executables and the Python scripts read from this file. The C++ engine auto-discovers the project root by walking up from the executable path until it finds `shared/config.json`.

---

## How to Improve Each Component

### 1. Strategy Logic (`shared/core_logic/turtle_math.py`)

**What to change:** Entry/exit rules, indicator calculations, signal generation.

**Impact:** Changes here affect **all three execution paths** (Python backtest, C++ backtest, live trading) simultaneously.

**How to do it:**
1. Modify the function signature and body in `turtle_math.py`.
2. Update callers if the signature changes:
   - `backtesting/strategies/trend_following.py` line 40
   - `live_strategy/live_trend_bot.py` line 62
3. Run Path A backtest first for fast iteration.
4. Run Path B backtest to verify the live code path.
5. If adding new indicator columns, also update `KLineData` struct in `live_engine/src/core/kline_data.h` and the JSON serialization in `ipc_server.cpp::publish_kline()`.

**To add a completely new strategy:**
1. Create a new function in `shared/core_logic/` (e.g., `mean_reversion.py`).
2. Create a new Backtrader strategy in `backtesting/strategies/` that imports it.
3. Create a new live bot in `live_strategy/` that imports it.
4. The C++ side likely needs no changes — it just passes market data.

### 2. Python Backtrader Backtest (`backtesting/`)

**What to change:** Data loading, analyzers, parameter optimization.

**Files to modify:**
- `run_backtest.py`: Add new data feeds, analyzers, or parameter sweeps.
- `data_feeds.py`: Add custom Backtrader lines if your new strategy needs them.
- `indicators.py`: Implement custom Backtrader indicators here.
- `strategies/base_strategy.py`: Add new housekeeping (e.g., margin checks).

**How to test:** Just run `python run_backtest.py`. No C++ build needed.

### 3. C++ Execution Engine (`live_engine/src/core/`)

**What to change:** Risk management rules, IPC protocol, data structures.

- **Risk parameters** (`risk_manager.cpp`): Change `risk_pct_` and `max_leverage_` constructor defaults, or modify `calculate_target_size()` for a different sizing formula.
- **IPC protocol** (`ipc_server.cpp`): Modify `publish_kline()` to send additional state fields (they are automatically forwarded to Python). Add new message types (e.g., `"type": "alert"`) via `queue_order_update()` or a new queue. The `order_update_queue_` pattern (enqueue from any thread, drain on main-loop via `pump_order_updates()`) should be replicated for any new ZMQ message types since the PUB socket is not thread-safe.
- **IPC command handling** (`handle_message()`): Add new action types beyond BUY/SELL/CLOSE_LONG/CLOSE_SHORT. The existing guards (position check, open-order check) will apply automatically.
- **Data structures** (`kline_data.h`): Add fields needed by new strategies — then update CSV parsing in `data_replayer.cpp`, WebSocket parsing in `binance_ws.cpp`, and JSON serialization in `ipc_server.cpp`.
- **Thread safety** (`thread_safe_queue.h`): Already generic with timed `wait_and_pop` — no changes needed unless you need bounded queues.
- **State injection** (`ipc_server.cpp::publish_kline()`): To add more RiskManager fields to the kline JSON, just add `j["new_field"] = risk_manager_->get_xxx();` in `publish_kline()`, then read it in Python's `on_handle_market_data()`.

### 4. Mock Executor (`live_engine/src/backtest/mock_executor.cpp`)

**What to change:** Fill simulation, slippage model, fee calculation.

- Modify `apply_slippage()` for a different slippage model.
- Modify `send_order()` to simulate partial fills or latency.
- Modify the trade record format in `TradeRecord` struct (`mock_executor.h`).

### 5. Live Executor (`live_engine/src/live/binance_live_executor.cpp`)

**What to change:** Order type, exchange endpoint, error handling, order timeout behavior.

- Currently sends **LIMIT** orders. To use MARKET orders, change `"type=LIMIT"` → `"type=MARKET"` in `place_order_internal()` and remove `timeInForce`.
- To switch from **Testnet to Mainnet**, change the base URL from `https://testnet.binancefuture.com` to `https://fapi.binance.com` in all HTTP call sites (`place_order_internal`, `cancel_order`, `get_market_price`, `get_listen_key`, `get_initial_state`).
- **Order timeout:** Adjust `kOrderTimeout` (default 3 min) and `kMaxRepriceAttempts` (default 2) in `binance_live_executor.h`.
- **Reprice logic:** Modify `reprice_order()` to use a different pricing strategy (e.g., offset from market by N bps).
- **ClientOrderId format:** Change `next_client_order_id()` if you need a different ID scheme.

### 5b. Order Tracker (`live_engine/src/live/order_tracker.h/.cpp`)

**What to change:** Order lifecycle tracking, timeout detection, status publishing.

- `register_order()`: Called after a successful REST POST — stores order metadata + timestamp.
- `on_order_update()`: Called from the private WebSocket `ORDER_TRADE_UPDATE` handler — maps Binance status strings (`NEW`, `PARTIALLY_FILLED`, `FILLED`, `CANCELED`, `EXPIRED`, `REJECTED`) to the `TrackedOrderStatus` enum and fires the callback on terminal transitions.
- `get_timed_out_orders(ms)`: Scans for orders stuck in `NEW` or `PARTIALLY_FILLED` longer than `ms`. Used by the watchdog thread.
- `mark_cancel_requested(cid)`: Marks an order as cancelled after the watchdog sends a cancel request. Deduplicates against the subsequent WebSocket event.
- `has_open_order()`: Returns true if any order is still `NEW` or `PARTIALLY_FILLED`. Used by `IpcServer::handle_message()` to reject duplicate open signals.
- `prune(age_ms)`: Removes terminal orders older than `age_ms` (default 10 min) to prevent memory leaks.
- To track additional fields from Binance's `ORDER_TRADE_UPDATE` payload (e.g., `stopPrice`, `activationPrice`), add them to `TrackedOrder` and update `on_order_update()`.

### 6. Data Pipeline (`data/`)

**What to change:** Data sources, supported symbols, timeframes.

- `download_binance_data.py`: Change `SYMBOL` or `INTERVAL` constants. Add `--symbol` and `--interval` CLI flags for flexibility.
- `prepare_csv_data.py`: Adjust `CSV_COLUMNS` if your C++ engine expects different columns.

### 7. Live Strategy Process (`live_strategy/`)

**What to change:** Strategy parameters, signal gating, logging.

- `live_trend_bot.py`: Adjust `entry_period`, `exit_period`, `atr_period`, `atr_mult` in `__init__`.
- **Signal gating** (`execute_strategy_logic()`): The bot now reads `current_position`/`available_balance`/`stop_price` from every kline (synced from C++ RiskManager). Modify the gating conditions if you want different behavior (e.g., allow pyramiding, add balance-based size limits on the Python side).
- **State injection:** To consume additional RiskManager fields, add `self.new_field = data.get("new_field", default)` in `on_handle_market_data()`.
- **Order updates:** `on_order_update()` receives order status changes (FILLED, CANCELED, timeout, etc.). Extend this method to take action on fills (e.g., update internal P&L tracking, send alerts).
- `zmq_client.py`: The message dispatcher now routes `"order_update"` messages to a separate callback. To add new ZMQ message types, add an `elif mtype == "new_type"` branch in `start_listening()`. ACKs are only sent for kline messages (to keep the backtest sync path working).

---

## Build & Run Cheat Sheet

### One-Time Setup

```bash
# 1. Download historical data (incremental by default, add --full-refresh for full redownload)
cd data
python download_binance_data.py --prepare-csv

# 2. Build C++ engine
cd ../live_engine
mkdir build && cd build
cmake ..
cmake --build . --config Debug
```

> **Note:** After any C++ code changes, rebuild with `cmake --build . --config Debug`. If you previously built in Release, either rebuild both or stick to one config to avoid running stale binaries.

### Python Backtest (Path A)
```bash
cd backtesting
python run_backtest.py
```

### C++ Backtest (Path B)
```bash
# Terminal 1 — Python brain
cd live_strategy
python live_trend_bot.py

# Terminal 2 — C++ backtest
cd live_engine/build/Debug
backtest_engine.exe

# Visualize
cd backtesting
python plot_results.py
```

**What to verify in the Python terminal:**
- `📡 C++ state: pos=0.0000 | flat=True | signal=1` on the first breakout bar → BUY sent
- `📡 C++ state: pos=X.XXXX | flat=False | signal=1` on subsequent bars → `🔒 Long breakout but pos=X.XXXX (not flat); skip BUY`
- C++ should no longer print `🚫 [IPC] Already in position` for gated signals

### Testnet Live Trading
```bash
# Terminal 1 — C++ live engine
cd live_engine/build/Debug
live_engine.exe

# Terminal 2 — Python strategy
cd live_strategy
python live_trend_bot.py
```

---

## Dependencies

### Python
| Package | Used By |
|---------|---------|
| `backtrader` | Python backtesting |
| `pandas` | Data loading, strategy calculations |
| `numpy` | Numerical operations |
| `pyzmq` | IPC communication |
| `requests` | Historical data download |
| `matplotlib` | Backtest visualization |
| `pyarrow` / `fastparquet` | Parquet file support |

Install:
```bash
pip install backtrader pandas numpy pyzmq requests matplotlib pyarrow
```

### C++
| Library | Purpose |
|---------|---------|
| [nlohmann/json](https://github.com/nlohmann/json) | JSON parsing |
| [IXWebSocket](https://github.com/machinezone/IXWebSocket) | Binance WebSocket client |
| [libzmq + cppzmq](https://github.com/zeromq) | ZeroMQ IPC |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) | REST API client |
| OpenSSL | HMAC-SHA256 signature for Binance API |

All C++ dependencies are fetched automatically by CMake's `FetchContent` — no manual installation needed beyond having CMake, a C++17 compiler, and OpenSSL development headers on your system.

### System
- **CMake** ≥ 3.14
- **C++17** compiler (MSVC 2019+, GCC 8+, Clang 7+)
- **OpenSSL** development libraries
- **Windows** (primary target; Linux/macOS supported with minor path adjustments)
