# Trading Bot Project — Hybrid Quantitative Trading & Microstructure System

A high-performance, event-driven **C++ / Python** quantitative trading system built on a **6-phase quant architecture** featuring **Binary MessagePack IPC**, **Queue-based Matching Simulation**, **Order Flow Imbalance (OFI) Microstructure Analytics**, and **Pure YAML-Driven Strategy Assembly**.

The split-brain design keeps a **Python "brain"** (research, feature engineering, ML signal filtering, and position sizing) and an ultra-low-latency **C++ engine** (data replay, bracket stop management, realistic order book queue simulation, and live Binance exchange connectivity).

---

## Table of Contents

1. [Architecture Overview & Flow](#architecture-overview--flow)
2. [Core Principles & Architectural Manifesto](#core-principles--architectural-manifesto)
3. [Quick Start & One-Command Runner](#quick-start--one-command-runner)
4. [3-Year Binance Microstructure Data Pipeline](#3-year-binance-microstructure-data-pipeline)
5. [The Quant Research & Execution Lifecycle (Phase 1–6)](#the-quant-research--execution-lifecycle-phase-16)
6. [Supported Strategies](#supported-strategies)
7. [Adding a New Strategy (Zero-Code / YAML-Only)](#adding-a-new-strategy-zero-code--yaml-only)
8. [Directory Structure](#directory-structure)
9. [Test Suite](#test-suite)

---

## Architecture Overview & Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│                        RESEARCH & SIGNAL BRAIN (Python)               │
│                                                                        │
│   [Event Trigger] ──> [Feature Set] ──> [Labeler / Barrier]            │
│         │                  │                      │                    │
│         └───────────┬──────┴──────────────────────┘                    │
│                     ▼                                                  │
│             [ModelEvaluator] ──> XGBoost (model.json + features.json)   │
│                     │                                                  │
│                     ▼                                                  │
│             [StrategyFactory] (Pure YAML-driven reflection)            │
│                     │                                                  │
│                     ▼                                                  │
│             [StrategyWrapper] (O(1) Streaming Incremental Indicators)  │
│                     │                                                  │
│                     ▼ (OrderPayload with ExecutionSpec)                │
└─────────────────────┬──────────────────────────────────────────────────┘
                      │  Binary MessagePack over ZMQ (TCP:5555 / 5556)
┌─────────────────────▼──────────────────────────────────────────────────┐
│                   EXECUTION & MATCHING ENGINE (C++)                    │
│                                                                        │
│   • IpcServer: High-performance binary MessagePack publisher/receiver │
│   • DataReplayer: High-throughput 1m microstructure streaming replay   │
│   • MockExecutor: Queue penetration simulation + non-linear slippage   │
│   • TrailingStopManager: Deterministic hard stop & trailing exit engine│
│   • RiskManager: Leverage limits & max drawdown circuit breaker        │
│   • Live Binance Gateway: WebSocket feed & REST order execution        │
└────────────────────────────────────────────────────────────────────────┘
```

| Component | Responsibility | Technology |
| :--- | :--- | :--- |
| **Research Pipeline** | Event detection, forward edge analysis, labeling, ML filtering | Python 3.12, Pandas, NumPy, Scikit-learn, XGBoost |
| **Strategy Assembly** | Dynamic component registration and configuration mapping | `core/strategy_factory.py`, PyYAML |
| **IPC Transport** | Zero-latency binary serialization between Python and C++ | ZeroMQ + MessagePack (`msgpack`) |
| **Execution Engine** | Order book queue tracking, bracket exit lifecycle, margin risk | C++20, CMake, MSYS2/MinGW, nlohmann-json (msgpack) |
| **Data Engine** | Multi-year trade tick ingestion, 1m OFI/CVD aggregation | Apache Arrow, Parquet (Zstandard) |

---

## Core Principles & Architectural Manifesto

1. **Config-Driven Philosophy**: Strategy development must minimize code modifications. Adding or modifying an Alpha is done entirely in YAML (`config/*.yaml`).
2. **Abstract Interface Pattern**: New triggers subclass `BaseEventTrigger` with a single method (`generate_signals`). Zero coupling to execution logic, network sockets, or C++ internals.
3. **Parity & Determinism**: All features, rolling indicators, and trailing stop exits guarantee 100% mathematical parity across vectorized research, streaming incremental Python, and C++.
4. **Realistic Market Microstructure**: No resting limit order fills immediately. Simulated fills require queue volume clearance or price penetration; market orders incur non-linear slippage scaling with trade notional.
5. **Codebase Hygiene**: Zero dead code. All legacy, unreferenced, or ad-hoc scripts are purged once replaced by standardized CLI pipelines.

---

## Quick Start & One-Command Runner

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

*(Key requirements: `pandas`, `numpy`, `xgboost`, `scipy`, `scikit-learn`, `pyyaml`, `pyzmq`, `pyarrow`, `msgpack`)*

### 2. Build C++ Engine

```powershell
.\build.ps1
```
*(Requires MSYS2 UCRT64. Compiles `live_engine/build_cmake/backtest_engine.exe` and `live_engine.exe`)*

### 3. One-Command System Runner (`scripts/run_system.py`)

No need to open two separate terminals. The orchestration CLI manages process lifetimes, ZMQ handshake, and clean shutdown automatically:

```bash
# Run C++ high-fidelity backtest with queue matching
python scripts/run_system.py --config config/ofi_momentum.yaml --mode backtest

# Run live / paper trading on Binance Testnet
python scripts/run_system.py --config config/ofi_momentum.yaml --mode live
```

---

## 3-Year Binance Microstructure Data Pipeline

The system includes an in-memory streaming ingestion pipeline capable of downloading and converting 36 full months (2021-01 to 2023-12) of Binance USD-M Futures `aggTrades` (over 250 million raw trades) without filling the disk with raw CSVs:

```bash
# 1. Download and convert monthly aggTrades zip directly into compressed Parquet
python data/download_binance_aggtrades.py --start-year 2021 --start-month 1 --end-year 2023 --end-month 12

# 2. Aggregate raw ticks into 1-minute microstructure bars
python data/build_microstructure_dataset.py
```

### Dataset Specifications (`BTCUSDT_1m_microstructure_3yr.parquet`)
- **Period**: 2021-01-01 00:00:00 to 2023-12-31 23:59:00 (3 full years)
- **Bar Count**: 1,565,083 1-minute bars
- **Storage**: ~85.2 MB (Zstandard compressed, loads in < 1 second)
- **Features Included**:
  - Standard OHLCV (`open`, `high`, `low`, `close`, `volume`)
  - Microstructure Flow: `taker_buy_base`, `taker_buy_quote`, `turnover`, `trades`
  - Signals: `ofi` (Order Flow Imbalance), `vwap` (Volume Weighted Average Price), `taker_ratio`

---

## The Quant Research & Execution Lifecycle (Phase 1–6)

All phases support `--config <path>`:

### Phase 1 — Event Trigger Validation
Tests whether the trigger's entries carry a statistical forward return edge over a random-entry baseline:
```bash
python scripts/phase1_validate.py --config config/ofi_momentum.yaml
```

### Phase 2 & 3 — Feature Engineering & ML Training
Extracts lookahead-free features, computes trailing exit labels, trains an XGBoost classifier/regressor, and checks AUC / Rank IC / Decile monotonicity:
```bash
python scripts/phase3_train.py --config config/ofi_momentum.yaml
# Saves: research/outputs/<prefix>_model.json and <prefix>_features.json
```

### Phase 3b — Vectorized Backtest
Fast vectorized profitability check across Train, Test, and Full 3-Year datasets:
```bash
python scripts/phase3b_backtest.py --config config/ofi_momentum.yaml
```

### Phase 4 — Position Sizing & Margin Risk Limits
Configured in YAML (`sizing:` and `risk:` blocks) and enforced natively by `execution/sizers.py` and C++ `RiskManager`.

### Phase 5 — High-Fidelity C++ Engine Backtest
Runs full historical replay with queue-based order book penetration, non-linear market impact, and taker fees:
```bash
python scripts/run_system.py --config config/ofi_momentum.yaml --mode backtest
```

### Phase 6 — Live Exchange Connectivity (Binance Testnet / Mainnet)
Connects to real-time Binance WebSocket feed and executes orders through C++ gateway:
```bash
python scripts/run_system.py --config config/ofi_momentum.yaml --mode live
```

---

## Supported Strategies

| Strategy | Trigger Class | Concept | 3-Year Performance (2021–2023) | Configuration File |
| :--- | :--- | :--- | :--- | :--- |
| **OFI Momentum** | `OFIMomentumTrigger` | Order Flow Imbalance surge + Donchian price breakout + macro trend | **OOS Sharpe +1.92**, Return +30.1%, Max DD 8.2% | `config/ofi_momentum.yaml` |
| **Trend Breakout (High-Freq)** | `TrendFilteredBreakoutTrigger` | 1.5-day breakout gated by 6-day trend MA + XGBoost win-probability gate | Sharpe +0.74, Return +81%, Max DD 22% | `config/trend_breakout_highfreq.yaml` |
| **Trend Breakout (Low-Freq)** | `TrendFilteredBreakoutTrigger` | 5-day breakout gated by 20-day trend MA (pure rule-based) | Low turnover (~1 trade/month), robust trend capture | `config/trend_breakout_lowfreq.yaml` |

---

## Adding a New Strategy (Zero-Code / YAML-Only)

Thanks to the dynamic `StrategyFactory`, creating a new strategy requires only subclassing `BaseEventTrigger`:

### 1. Write the Trigger Subclass
Create `research/triggers/my_alpha.py`:

```python
import pandas as pd
from core.trigger import BaseEventTrigger
from core.strategy_factory import register_trigger

@register_trigger("my_alpha")
class MyAlphaTrigger(BaseEventTrigger):
    def __init__(self, period: int = 120):
        self.period = period

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Return pd.Series of +1 (long), -1 (short), or 0 (idle)
        return ...
```

### 2. Configure Pure YAML
Create `config/my_alpha.yaml`:

```yaml
strategy:
  name: "my_alpha"
  symbol: "BTCUSDT"

data:
  source: "data/historical_data/BTCUSDT_1m_microstructure_3yr.parquet"
  train_range: ["2021-01-01", "2023-01-01"]
  test_range:  ["2023-01-01", "2023-12-31"]

trigger:
  type: "my_alpha"
  params:
    period: 120

indicators:
  entry_period: 120
  exit_period: 60

sizing:
  type: "fixed_risk"
  risk_pct: 0.02
  max_leverage: 10.0

execution:
  entry:
    order_type: market
  exit:
    order_type: market
    timeout_ms: 3000
    unfilled_policy: market
```

### 3. Run Validation Pipeline
You can immediately run all phases without touching any other files:
```bash
python scripts/phase1_validate.py --config config/my_alpha.yaml
python scripts/phase3_train.py --config config/my_alpha.yaml
python scripts/phase3b_backtest.py --config config/my_alpha.yaml
python scripts/run_system.py --config config/my_alpha.yaml
```

---

## Directory Structure

```
core/                          ABC contracts & StrategyFactory
  strategy_factory.py          Dynamic reflection engine & component registry
  trigger.py  feature.py  labeler.py       (Phase 1-3 ABCs)
  position_sizer.py  risk_manager.py       (Phase 4 ABCs)
  order_payload.py  execution_spec.py      (Python → C++ order protocol)
  strategy_wrapper.py                      (Execution bridge & state machine)

execution/                     Concrete position sizers & risk managers
  sizers.py  risk_managers.py

research/                      Research toolkit & Alpha triggers
  triggers/                    BaseEventTrigger implementations (OFI, Adam, Trend)
  features.py                  Vectorized zero-lookahead feature extraction
  features_incremental.py      O(1) streaming indicators for real-time replay
  indicator_spec.py            ROLLING_SPEC — single source of truth for indicators
  labeling.py                  TrailingExitLabeler, TripleBarrierLabeler
  evaluator.py  backtest.py    ModelEvaluator & lightweight vectorized backtester
  outputs/                     Trained XGBoost models & feature lists

data/                          Data ingestion & microstructure engine
  download_binance_aggtrades.py Streaming in-memory Binance aggTrades downloader
  build_microstructure_dataset.py 1m microstructure bar aggregator (OFI, CVD, VWAP)
  export_replayer_csv.py       Exporter for C++ DataReplayer format
  historical_data/             Parquet datasets & CSV replays

live_strategy/                 Live bot & IPC communications
  live_trend_bot.py            Composition shell (LiveTurtleBot.from_config)
  zmq_client.py                Binary MessagePack IPC client
  zmq_feeder.py  zmq_gateway.py Streaming market data feeder & order gateway

live_engine/                   High-performance C++ execution engine
  src/core/                    IpcServer (MessagePack), RiskManager, TrailingStop
  src/backtest/                DataReplayer, MockExecutor (queue simulation & slippage)
  src/live/                    Binance WebSocket & REST execution gateway

config/                        Strategy YAML definitions (ofi_momentum, trend_breakout)
scripts/                       CLI runners for Phase 1, Phase 3, Phase 3b, and run_system
tests/                         Unit & parity test suite (pytest)
```

---

## Test Suite

Run the full pytest suite:

```bash
pytest -v tests/
```

- `test_order_payload.py`: OrderPayload immutability, serialization & execution spec.
- `test_strategy_wrapper.py`: StrategyWrapper state machine & risk limits.
- `test_parity.py`: Abstract Base Class compliance and formula parity.
- `test_incremental_indicators.py`: Parity between vectorized and $O(1)$ streaming indicators.
- `test_trailing_stop_parity.py`: Parity between C++ trailing stop and Python research labeler.
- `test_strategy_factory.py`: Dynamic YAML loading, registration, and bot assembly.
