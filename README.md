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

The system is split into **three layers**:

| Layer | Language | Role |
|-------|----------|------|
| **Strategy Logic** | Python | Pure math — signal generation from OHLCV data |
| **Execution Engine** | C++ | Market data ingestion, order routing, risk management |
| **IPC Bridge** | ZMQ | PUB/PULL sockets linking C++ and Python |

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
│       │   ├── i_order_executor.h   # Abstract order executor interface
│       │   ├── kline_data.h         # K-line data structure
│       │   ├── ipc_server.h/.cpp    # ZMQ IPC server (PUB + PULL)
│       │   ├── risk_manager.h/.cpp  # Position sizing & risk management
│       │   └── thread_safe_queue.h  # Generic thread-safe queue
│       │
│       ├── backtest/                # C++ backtest subsystem
│       │   ├── main_backtest.cpp    # Entry point → backtest_engine.exe
│       │   ├── mock_executor.h/.cpp # Simulated order execution (slippage + fees)
│       │   └── data_replayer.h/.cpp # CSV bar-by-bar replay engine
│       │
│       └── live/                    # Live trading subsystem
│           ├── main_live.cpp        # Entry point → live_engine.exe
│           ├── binance_ws.h/.cpp    # Binance public WebSocket (1m klines)
│           └── binance_live_executor.h/.cpp  # Real Binance Testnet REST orders
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
  "is_closed": true
}
```

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

- In **backtest mode** (sync): The C++ engine publishes one bar, then blocks until it receives an ACK from Python. This ensures the strategy processes every bar in order.
- In **live mode** (async): A background thread continuously reads order signals from the PULL socket without blocking the market data publisher.

**Handshake (backtest only):** The C++ backtest engine pings the Python process repeatedly until an ACK arrives, ensuring the brain is connected before replay starts.

### Data Flow Diagrams

#### Live Trading Flow
```
Binance WS ──► C++ main_live ──► ZMQ PUB ──► Python live_trend_bot.py
  (klines)       thread           :5555         (SUB socket)
                                       
               ┌─────────────────────────────┐
               │ Python calculates signal    │
               │ via turtle_math.py          │
               └──────────────┬──────────────┘
                              │
               ZMQ PULL ◄─────┘
               :5556
                  │
          ┌───────┴───────┐
          │ C++ handles    │
          │ message:       │
          │ RiskManager    │
          │ validates size │
          │      │         │
          │ BinanceLive    │
          │ Executor sends │
          │ REST order to  │
          │ Binance Testnet│
          └────────────────┘
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
   python download_binance_data.py --prepare-csv
   ```
   This downloads historical klines from Binance Vision as Parquet, then converts to a CSV file at `data/historical_data/BTCUSDT_1m_full.csv` that the C++ engine can parse.

2. **Build the C++ engine:**
   ```bash
   cd live_engine
   mkdir build && cd build
   cmake ..
   cmake --build . --config Release
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
   cd live_engine/build/Release   # or build/ backtest_engine.exe location
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
   cmake --build . --config Release
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
   - Python sends e.g. `{"action": "BUY", "symbol": "BTCUSDT", "price": 42100.0, "stop_price": 41800.0}`
   - C++ `IpcServer::handle_message()` receives it
   - `RiskManager::calculate_target_size()` validates:
     - Stop direction sanity (long stop must be below entry, short stop above)
     - Position size = `(Balance × 2%) / |Entry − Stop|`
     - Caps notional at 20× leverage
     - Rounds to 0.001 BTC (Binance min qty step)
   - Refuses to open if already in position (no pyramiding)
   - `BinanceLiveExecutor::send_order()` sends a LIMIT order via REST to Binance Testnet

### Risk Parameters (editable in code)

In `live_engine/src/core/risk_manager.cpp` (constructor):
- `risk_pct`: 2% of balance per trade (default)
- `max_leverage`: 20× (capping mechanism)

In `live_strategy/live_trend_bot.py` (`__init__`):
- `entry_period`, `exit_period`, `atr_period`, `atr_mult` — strategy parameters

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
- **IPC protocol** (`ipc_server.cpp`): Modify `publish_kline()` to send additional fields, or `handle_message()` to support new action types.
- **Data structures** (`kline_data.h`): Add fields needed by new strategies — then update CSV parsing in `data_replayer.cpp`, WebSocket parsing in `binance_ws.cpp`, and JSON serialization in `ipc_server.cpp`.
- **Thread safety** (`thread_safe_queue.h`): Already generic — no changes needed unless you need bounded queues.

### 4. Mock Executor (`live_engine/src/backtest/mock_executor.cpp`)

**What to change:** Fill simulation, slippage model, fee calculation.

- Modify `apply_slippage()` for a different slippage model.
- Modify `send_order()` to simulate partial fills or latency.
- Modify the trade record format in `TradeRecord` struct (`mock_executor.h`).

### 5. Live Executor (`live_engine/src/live/binance_live_executor.cpp`)

**What to change:** Order type, exchange endpoint, error handling.

- Currently sends **LIMIT** orders. To use MARKET orders, change `"type=LIMIT"` → `"type=MARKET"` in `send_order()` and remove `timeInForce`.
- To switch from **Testnet to Mainnet**, change the base URL from `https://testnet.binancefuture.com` to `https://fapi.binance.com` in all three HTTP call sites.
- To add **order status tracking**, store the order ID from the REST response and poll for fills.

### 6. Data Pipeline (`data/`)

**What to change:** Data sources, supported symbols, timeframes.

- `download_binance_data.py`: Change `SYMBOL` or `INTERVAL` constants. Add `--symbol` and `--interval` CLI flags for flexibility.
- `prepare_csv_data.py`: Adjust `CSV_COLUMNS` if your C++ engine expects different columns.

### 7. Live Strategy Process (`live_strategy/`)

**What to change:** Strategy parameters, position tracking, logging.

- `live_trend_bot.py`: Adjust `entry_period`, `exit_period`, `atr_period`, `atr_mult` in `__init__`.
- `zmq_client.py`: Add reconnection logic or heartbeat monitoring.

---

## Build & Run Cheat Sheet

### One-Time Setup

```bash
# 1. Download historical data
cd data
python download_binance_data.py --prepare-csv

# 2. Build C++ engine
cd ../live_engine
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

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
cd live_engine/build/Release    # or build/
backtest_engine.exe

# Visualize
cd backtesting
python plot_results.py
```

### Testnet Live Trading
```bash
# Terminal 1 — C++ live engine
cd live_engine/build/Release
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
