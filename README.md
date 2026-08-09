# Trading Bot Project — BTCUSDT Turtle Trend-Following System

A hybrid **C++ / Python** automated trading system built on a **unified 6-phase quant architecture** with ABC contracts, bracket-order execution, and YAML-driven strategy assembly.

One strategy brain (`shared/core_logic/turtle_math.py`) powers everything. One ML pipeline (`research/` → `core/strategy_wrapper.py`) bridges research to execution.

| Mode | Engine | When to use |
|------|--------|-------------|
| **C++ Backtest** | `backtest_engine.exe` + Python `live_trend_bot.py` | Rehearse the exact live code path with friction costs |
| **Live Trading** | `live_engine.exe` + Python `live_trend_bot.py` | Paper/live on Binance Testnet |
| **Fast Research Backtest** | `research/backtest.py` (`lightweight_backtest()`) | Vectorized prototyping (~50-200x faster), parameter sweeps |

### Strategy signals

`calculate_turtle_signals(df, entry_period, exit_period, atr_period, atr_mult)` returns a signal and a stop price:

| Signal | Meaning |
|--------|---------|
| `1` | Open Long — price broke above `entry_period`-bar high |
| `-1` | Open Short — price broke below `entry_period`-bar low |
| `2` | Close Long — price broke below `exit_period`-bar low |
| `-2` | Close Short — price broke above `exit_period`-bar high |
| `0` | No action |

---

## 1. Quick Reference

```bash
# 1) Install deps + build C++
pip install -r requirements.txt
cd live_engine && mkdir build && cd build && cmake .. && cmake --build . --config Debug

# 2) Download data
cd ../../data && python download_binance_data.py --prepare-csv

# 3) Research pipeline (YAML → dataset → model → evaluation)
cd ../research && python pipeline_runner.py ../config/example_turtle_vol.yaml

# 4) C++ Backtest — two terminals
#    T1: cd live_strategy && python live_trend_bot.py --no-warmup
#    T2: cd live_engine/build/Debug && backtest_engine.exe

# 5) Live (Testnet) — two terminals
#    T1: cd live_engine/build/Debug && live_engine.exe
#    T2: cd live_strategy && python live_trend_bot.py
```

---

## 2. Setup

### 2.1 Python dependencies

```bash
pip install -r requirements.txt
```

Requirements: `pandas`, `numpy`, `xgboost`, `scipy`, `scikit-learn`, `pyyaml`, `pyzmq`, `requests`, `matplotlib`, `pyarrow`.

### 2.2 Build the C++ engine

```bash
cd live_engine && mkdir build && cd build
cmake ..
cmake --build . --config Debug
```

Produces `backtest_engine.exe` and `live_engine.exe` in `live_engine/build/Debug`. All C++ dependencies are fetched automatically by CMake `FetchContent`.

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

### 2.4 Historical data

```bash
cd data && python download_binance_data.py --prepare-csv
```

Outputs `data/historical_data/BTCUSDT_1m_full.parquet` (and `.csv` with `--prepare-csv`).

---

## 3. Phase-by-Phase Developer Guide

### Architecture Overview

```
Phase 1-3 (Research Pipeline)              Phase 4-6 (Execution Pipeline)
─────────────────────────────────         ────────────────────────────────
[Trigger] → [Features] → [Labeler]        [Sizer] → [RiskManager]
     │            │           │                │          │
     └────────────┴───────┐   │                └────┬─────┘
                          ▼   ▼                     ▼
                     [build_ml_dataset()]    [StrategyWrapper]
                          │                        │
                          ▼                        ▼
                    [ModelEvaluator]          [OrderPayload]
                    (train, IC, decile)            │
                          │                        ▼
                          ▼                  C++ Engine
                 model.json + features.json   (backtest or live)
```

**Key rule:** Research (Phase 1-3) and Execution (Phase 4-6) are decoupled. The **only** bridge: model weights + feature list files. No `research.*` imports in `core/` or `execution/`.

---

### Phase 1: Event Trigger

**Goal:** Define entry events. Filter market noise.

| File | Role |
|------|------|
| `core/trigger.py` | `BaseEventTrigger` ABC |
| `research/triggers/turtle_breakout.py` | `TurtleBreakoutTrigger` — 20-day Donchian breakout |

```python
from research.triggers.turtle_breakout import TurtleBreakoutTrigger

trigger = TurtleBreakoutTrigger(
    entry_period=28800,       # 20-day at 1m bars
    atr_period=28800,
    intensity_threshold=0.5,  # require ≥ 0.5 ATR breakout
    signed=True,
)
signals = trigger.generate_signals(df)  # pd.Series: 1=long, -1=short, 0=none
```

**Add a new trigger:** Subclass `BaseEventTrigger`, implement `generate_signals()`. Register in YAML. No core code changes.

---

### Phase 2: Features & Labeling

**Goal:** Lookahead-free features + supervised labels at event positions.

| File | Role |
|------|------|
| `core/feature.py` | `BaseFeature` ABC — `compute(df, events)` / `compute_one(df, idx)` |
| `core/labeler.py` | `BaseLabeler` ABC — `compute_labels(df, events)` |
| `research/features.py` | `add_indicators()` + `VolumeRatioFeature`, `ATRFeature`, etc. |
| `research/labeling.py` | `fixed_horizon_label()` + `FixedHorizonLabeler`, `TripleBarrierLabeler` |

```python
from research.features import add_indicators, VolumeRatioFeature, default_feature_set
from research.labeling import FixedHorizonLabeler

# Step A: Precompute indicators (ALL .shift(1) — zero lookahead guarantee)
df = add_indicators(raw_data, entry_period=28800, atr_period=28800, vol_period=20)

# Step B: Feature computation
vol = VolumeRatioFeature(vol_period=20)
features_df = vol.compute(df, signals)           # batch: DataFrame
feat_dict  = vol.compute_one(df, bar_index=150)  # live: dict

# All 8 features at once
features = default_feature_set()   # ATR + BreakoutIntensity + VolumeRatio + ...
features_df = features.compute(df, signals)

# Step C: Labeling
labeler = FixedHorizonLabeler(horizon=14400)  # 10-day forward at 1m
labels_df = labeler.compute_labels(df, signals)
# y_norm = trade-side forward return / daily ATR (volatility-normalized)
```

---

### Phase 3: Signal Evaluation & ML Training

**Goal:** Train XGBoost, evaluate Spearman IC + decile staircase.

| File | Role |
|------|------|
| `research/pipeline_runner.py` | One-command: YAML → dataset → model → evaluation |
| `research/dataset_builder.py` | `build_ml_dataset()` — raw klines → (X, y) |
| `research/evaluator.py` | `ModelEvaluator` — IC, decile, AUC-ROC, train/save |
| `config/example_turtle_vol.yaml` | Declares all components |

**One-command:**
```bash
python research/pipeline_runner.py config/example_turtle_vol.yaml
# → research/outputs/turtle_vol_filter_model.json
# → research/outputs/turtle_vol_filter_features.json
```

**Manual (notebooks):**
```python
from research.dataset_builder import build_ml_dataset
from research.features import add_indicators, default_feature_set
from research.triggers.turtle_breakout import TurtleBreakoutTrigger
from research.labeling import FixedHorizonLabeler
from research.evaluator import ModelEvaluator

df = add_indicators(raw_data)
trigger = TurtleBreakoutTrigger(entry_period=28800, atr_period=28800, signed=True)
features = default_feature_set()
labeler = FixedHorizonLabeler(horizon=14400)

X, meta = build_ml_dataset(df, trigger, features, labeler)

evaluator = ModelEvaluator()
X_np, y_np, feature_names = evaluator.prepare_features(X)
model = evaluator.train_model(X_np[:split], y_np[:split])

y_pred = model.predict(X_np[split:])
ic = evaluator.evaluate_rank_ic(y_np[split:], y_pred)
decile = evaluator.evaluate_decile_spread(y_np[split:], y_pred)

print(f"Spearman IC: {ic['ic']:.4f}  {'PASS' if ic['pass'] else 'FAIL'}")
print(f"Decile spread: {decile['spread']:+.4f}  monotonic={decile['monotonic']}")

evaluator.save("research/outputs", prefix="turtle_vol_filter")
```

---

### Phase 3b: Parameter Fine-Tuning

**Goal:** Optimize entry/exit/risk params via fast vectorized sweep before C++ backtest.

| File | Role |
|------|------|
| `research/param_sweep.py` | `run_parameter_sweep()` — multiprocessing grid search |
| `research/backtest.py` | `lightweight_backtest()` — ~50-200x faster than C++ |
| `execution/sizers.py` | `VolatilityTargetingSizer` — ABC sizer |
| `execution/risk_managers.py` | `MaxDrawdownRiskManager` — ABC risk |

```python
from research.param_sweep import run_parameter_sweep
from research.backtest import lightweight_backtest
from research.features import add_indicators
from execution.sizers import VolatilityTargetingSizer
from execution.risk_managers import MaxDrawdownRiskManager

param_grid = {
    "entry_period": [10, 20, 40],
    "atr_mult": [1.5, 2.0, 3.0, 4.0],
    "risk_pct": [0.01, 0.02, 0.03],
}

def sweep_target(entry_period, atr_mult, risk_pct, raw_data):
    df = add_indicators(raw_data, entry_period=entry_period, atr_period=entry_period)
    sizer = VolatilityTargetingSizer(risk_pct=risk_pct)
    risk = MaxDrawdownRiskManager(max_dd_pct=0.05)
    return lightweight_backtest(
        df, entry_period=entry_period, atr_mult=atr_mult,
        position_sizer=sizer, risk_manager=risk,
    )

results = run_parameter_sweep(sweep_target, param_grid, raw_data=df,
                               n_jobs=-1, rank_by="sharpe")
# Top result → write into config/live_strategy.yaml
```

---

### Phase 4: Portfolio & Risk

**Goal:** Position sizing + risk gates. Isolated from order execution.

| File | Role |
|------|------|
| `core/position_sizer.py` | `BasePositionSizer` ABC |
| `core/risk_manager.py` | `BaseRiskManager` ABC |
| `execution/sizers.py` | `VolatilityTargetingSizer` — Turtle N-value |
| `execution/risk_managers.py` | `MaxDrawdownRiskManager`, `LivePositionGate` |

```yaml
# config/live_strategy.yaml — YAML assembly, no code changes needed
position_sizer:
  type: "execution.sizers.VolatilityTargetingSizer"
  params: { risk_pct: 0.01, max_leverage: 20.0 }
risk_manager:
  type: "execution.risk_managers.MaxDrawdownRiskManager"
  params: { max_dd_pct: 0.05 }
```

```python
# Manual instantiation
from execution.sizers import VolatilityTargetingSizer
from execution.risk_managers import MaxDrawdownRiskManager, LivePositionGate

sizer = VolatilityTargetingSizer(risk_pct=0.01, max_leverage=20.0)
size = sizer.calculate_size(2.0, 500.0, 10000.0, 42000.0)
# = floor(min(equity*1%/stop_dist, equity*20/entry) / 0.001) * 0.001

MaxDrawdownRiskManager(max_dd_pct=0.05).check_risk_limits({"current_drawdown": 0.03})  # True
LivePositionGate().check_risk_limits({"current_position": 0.0})  # True
```

---

### Phase 5: Robust Backtesting (C++ Engine + StrategyWrapper)

**Goal:** Full historical backtest with friction costs. Same code path as live.

| File | Role |
|------|------|
| `core/strategy_wrapper.py` | Loads model, computes signals via `calculate_turtle_signals()`, emits `OrderPayload` |
| `core/order_payload.py` | Bracket order data contract |
| `shared/core_logic/turtle_math.py` | `calculate_turtle_signals()` — single source of truth |
| `live_strategy/zmq_feeder.py` | ZMQ → buffer → callback |
| `live_strategy/zmq_gateway.py` | Sends `OrderPayload` to C++ |
| `live_strategy/live_trend_bot.py` | Composition shell |

```bash
# Terminal 1 — Python
python live_strategy/live_trend_bot.py --no-warmup \
    --entry 28800 --exit 14400 --atr-period 28800 --atr-mult 4.0 --risk-pct 0.01 \
    --model research/outputs/turtle_vol_filter_model.json \
    --features research/outputs/turtle_vol_filter_features.json --threshold 0.0

# Terminal 2 — C++
cd live_engine/build/Debug && backtest_engine.exe
```

**Per-bar flow:**
```
C++ ZMQ PUB kline
  → BinanceZmqDataFeeder (sync state, buffer, callback)
    → StrategyWrapper.on_bar()
        ├─ WAITING_CLOSE? → skip
        ├─ risk_manager.check()? → blocked? → skip
        ├─ calculate_turtle_signals(df, ...)  ← shared brain
        ├─ feature.compute_one() → ML predict → score > threshold?
        ├─ sizer.calculate_size() → size
        └─ OrderPayload(action, qty, price, stop, trailing_exit, period)
          → BinanceZmqExecutionGateway.send_order() → ZMQ PUSH
            → C++ executes entry + arms bracket orders
              → stop/trailing exit triggers → ZMQ PUB position_closed
                → StrategyWrapper.on_position_closed() → IDLE
```

---

### Phase 6: Live Incubation

**Goal:** Live Testnet. Same code as Phase 5. Only config differs.

```bash
# Terminal 1 — C++ live engine
cd live_engine/build/Debug && live_engine.exe

# Terminal 2 — Python (same params, no --no-warmup)
python live_strategy/live_trend_bot.py \
    --entry 28800 --exit 14400 --atr-period 28800 --atr-mult 4.0 --risk-pct 0.01 \
    --model research/outputs/turtle_vol_filter_model.json \
    --features research/outputs/turtle_vol_filter_features.json --threshold 0.0
```

`BinanceZmqDataFeeder` / `BinanceZmqExecutionGateway` implement `LiveDataFeeder` / `LiveExecutionGateway` ABCs. `StrategyWrapper` is identical to Phase 5 — guaranteeing 100% logic parity.

---

## 4. Bracket Order Protocol

Python handles **entry only**. C++ owns the exit lifecycle.

```
   IDLE                          WAITING_CLOSE
   ┌──────────┐                  ┌──────────────┐
   │ detecting│──OrderPayload──→ │ entry paused │
   │ entries  │                  │ C++ manages  │
   │          │←─POSITION_CLOSED─│ exit full    │
   └──────────┘                  └──────────────┘
```

**OrderPayload** (`core/order_payload.py`): `action`, `symbol`, `quantity`, `entry_price`, `hard_stop_loss`, `trailing_exit_indicator`, `trailing_exit_period`, `take_profit` (optional).

---

## 5. IPC Protocol (C++ ↔ Python over ZMQ)

| Direction | Pattern | Port | Content |
|-----------|---------|------|---------|
| C++ → Python | PUB | 5555 | `kline`, `order_update`, `position_closed` |
| Python → C++ | PULL | 5556 | order signals + `ack` |

**position_closed** (new bracket-order message):
```json
{"type":"position_closed", "symbol":"BTCUSDT", "reason":"stop_loss",
 "entry_price":42100.0, "exit_price":41800.0, "pnl":-37.0}
```

---

## 6. Directory Structure

```
core/                          ABC contracts (Phase 1-6)
  trigger.py, feature.py, labeler.py,
  position_sizer.py, risk_manager.py,
  order_payload.py, strategy_wrapper.py,
  data_feeder.py, execution_gateway.py
execution/                     Concrete sizers + risk managers
  sizers.py, risk_managers.py
research/                      Research toolkit (Phase 1-3)
  triggers/turtle_breakout.py  TurtleBreakoutTrigger
  features.py                  add_indicators() + 8 feature classes
  labeling.py                  labelers + BaseLabeler classes
  dataset_builder.py           build_ml_dataset() (ABCs + legacy)
  backtest.py                  lightweight_backtest()
  param_sweep.py               run_parameter_sweep()
  evaluator.py                 ModelEvaluator
  pipeline_runner.py           YAML-driven DI runner
  outputs/                     X_*.parquet, model.json, features.json
live_strategy/                 Execution layer (Phase 5-6)
  zmq_client.py                BinanceZmqClient (unchanged)
  zmq_feeder.py                BinanceZmqDataFeeder
  zmq_gateway.py               BinanceZmqExecutionGateway
  live_trend_bot.py            Composition shell
config/                        YAML strategy assembly
shared/                        config.json + core_logic/turtle_math.py
live_engine/                   C++ engine (unchanged)
data/                          Data pipeline
tests/                         Unit + parity tests
requirements.txt
```

---

## 7. Dependencies

**Python:** `pandas`, `numpy`, `xgboost`, `scipy`, `scikit-learn`, `pyyaml`, `pyzmq`, `requests`, `matplotlib`, `pyarrow`  
Install: `pip install -r requirements.txt`

**C++:** nlohmann/json, IXWebSocket, libzmq + cppzmq, cpp-httplib, OpenSSL (CMake FetchContent)  
**System:** CMake ≥ 3.14, C++17, OpenSSL headers

---

## 8. Running Tests

```bash
python -m research.features         # indicator precompute + zero-lookahead
python -m research.labeling         # triple-barrier cross-validation
python -m research.dataset_builder  # build_ml_dataset() demo
python -m research.backtest         # lightweight_backtest() demo

python tests/test_parity.py              # ABC ↔ legacy parity
python tests/test_order_payload.py       # OrderPayload dataclass
python tests/test_strategy_wrapper.py    # StrategyWrapper state machine
```
