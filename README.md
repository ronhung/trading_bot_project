# Trading Bot Project — BTCUSDT Quantitative Trading System

A hybrid **C++ / Python** automated trading system built on a **6-phase quant
architecture** with ABC contracts, bracket-order execution, parameterized order
execution, and YAML-driven strategy assembly.

The split-brain design keeps a **Python "brain"** (research + entry detection)
and a **C++ engine** (exit lifecycle + order execution). The only bridges between
them are model weights, indicator formulas, and an execution spec — so a new
strategy is a config change, not a code change to the engine.

---

## Table of contents

1. [Architecture at a glance](#architecture-at-a-glance)
2. [Quick start](#quick-start)
3. [Phase-by-phase guide](#phase-by-phase-guide)
4. [Running a backtest / live (detailed)](#running-a-backtest--live-detailed)
5. [Execution parameterization](#execution-parameterization)
6. [Config files](#config-files)
7. [Adding a new strategy](#adding-a-new-strategy)
8. [Directory structure](#directory-structure)
9. [Tests](#tests)
10. [Convenience scripts](#convenience-scripts)

---

## Architecture at a glance

```
Phase 1-3  (Research, Python)          Phase 4-6 (Execution, Python + C++)
─────────────────────────────────      ───────────────────────────────────
[Trigger] → [Features] → [Labeler]     [Sizer] → [RiskManager]
     │            │           │             │          │
     └────────────┴───────┐   │             └────┬─────┘
                          ▼   ▼                  ▼
                   [build ML dataset]     [StrategyWrapper]
                          │                     │
                          ▼                     ▼
                    ModelEvaluator          [OrderPayload]
                          │                     │
                          ▼                     ▼
                  model.json + features.json    C++ engine (backtest / live)
```

| Phase | Name | Runs where | Question it answers |
|-------|------|------------|---------------------|
| 1 | Trigger | Python | *When* do we enter? |
| 2 | Features + labeling | Python | *What* do we feed the model / how do we label? |
| 3 | ML training | Python | *Does* the signal carry edge? |
| 3b | Vectorized backtest | Python | *Is* it profitable before C++? |
| 4 | Sizer + risk | Python (size) / C++ (authoritative) | *How much* do we risk? |
| 5 | C++ backtest | C++ + Python (same code path as live) | *Does* it hold with friction costs? |
| 6 | Live (Testnet) | C++ + Python | *Does* it trade for real? |

### The two validated strategies

| Strategy | Trigger (`trend_breakout`) | Frequency | ML filter | Config |
|----------|---------------------------|-----------|-----------|--------|
| **High-freq** | `entry_period=2160`, `trend_period=8640`, `trail=1080` | higher (~ weekly) | XGBoost classifier (`P(win) > threshold`) | `config/trend_breakout_highfreq.yaml` |
| **Low-freq** | `entry_period=7200`, `trend_period=28800`, `trail=3600` | lower (~ monthly) | none | `config/trend_breakout_lowfreq.yaml` |

Both are long-only trend-filtered Donchian breakouts (BTC's short side has no edge
due to upward drift). The high-freq model is at
`research/outputs/trend_breakout_highfreq_model.json` +
`trend_breakout_highfreq_features.json`.

---

## Quick start

### 1. Install Python deps

```bash
pip install -r requirements.txt
```

Requirements: `pandas`, `numpy`, `xgboost`, `scipy`, `scikit-learn`, `pyyaml`,
`pyzmq`, `requests`, `matplotlib`, `pyarrow`.

### 2. Build the C++ engines

```powershell
# Windows (MSYS2 UCRT64 toolchain required)
.\build.ps1
```

Produces `live_engine/build_cmake/backtest_engine.exe` and `live_engine.exe`.
(If PowerShell blocks `.ps1`, run `powershell -ExecutionPolicy Bypass -File build.ps1`.)

### 3. Download historical data

```bash
cd data && python download_binance_data.py --prepare-csv
```

Outputs `data/historical_data/BTCUSDT_1m_full.parquet` (Python) and
`BTCUSDT_1m_full.csv` (C++ backtest engine).

### 4. Run a backtest (Phase 5) — two terminals

```powershell
# Terminal 1 — Python brain (--no-warmup: C++ replays the CSV, no REST fetch)
python live_strategy/live_trend_bot.py --config config/trend_breakout_highfreq.yaml --no-warmup

# Terminal 2 — C++ backtest engine (prints the BACKTEST REPORT)
cd live_engine/build_cmake
backtest_engine.exe
```

### 5. Run live (Phase 6, Binance Testnet) — two terminals

```powershell
# Terminal 1 — C++ live engine
cd live_engine/build_cmake
live_engine.exe

# Terminal 2 — Python brain (no --no-warmup: REST warmup pre-fills indicators)
python live_strategy/live_trend_bot.py --config config/trend_breakout_highfreq.yaml
```

> ⚠️ Live places real (paper) orders on Binance Testnet using the API keys in
> `shared/config.json`.

---

## Phase-by-phase guide

### Phase 1 — Event trigger

**Goal:** define *when* to enter, and prove the trigger's events beat a
random-entry baseline of the same direction.

```bash
python scripts/phase1_validate.py          # compare a couple of triggers on 2020-2023
```

Trigger classes live in `research/triggers/` and subclass
`core/trigger.py:BaseEventTrigger`. Each returns `{-1, 0, 1}` (short / none / long).
`research/trigger_analysis.py:analyze_trigger()` measures each event's forward
outcome against a random baseline across horizons.

### Phase 2 — Features & labeling

**Goal:** lookahead-free features + supervised labels at each event position.

| File | Role |
|------|------|
| `core/feature.py` | `BaseFeature` ABC |
| `core/labeler.py` | `BaseLabeler` ABC (the "barrier") |
| `research/features.py` | `add_indicators()` + feature classes |
| `research/labeling.py` | `FixedHorizonLabeler`, `TripleBarrierLabeler`, `TrailingExitLabeler` |

No standalone command — features/labeling are invoked by the Phase 3 scripts.

### Phase 3 — ML training

**Goal:** train XGBoost and check the IC / decile gates (Spearman IC > 0.02,
monotonic decile spread).

```bash
python scripts/phase3_train.py            # demo run over candidate strategies
python scripts/train_ml_highfreq.py       # trains the actual high-freq classifier
#   → research/outputs/trend_breakout_highfreq_model.json + _features.json
```

### Phase 3b — Vectorized backtest

**Goal:** a fast profit check before paying for a C++ backtest.

```bash
python scripts/phase3b_backtest.py        # lightweight_backtest() over train/test
```

`research/backtest.py:lightweight_backtest()` is ~50-200× faster than C++ and
used for parameter sweeps. It does **not** model fees/slippage/execution — that is
Phase 5's job.

### Phase 4 — Position sizing & risk

**Goal:** how much to risk per trade, and the risk gates.

| File | Role |
|------|------|
| `core/position_sizer.py` | `BasePositionSizer` ABC |
| `core/risk_manager.py` | `BaseRiskManager` ABC |
| `execution/sizers.py` | `FixedRiskSizer`, `VolatilityTargetingSizer` |
| `execution/risk_managers.py` | `MaxDrawdownRiskManager`, `LivePositionGate` |

> Note: in the live/backtest path, the **C++ `RiskManager` is authoritative** for
> size — Python computes a provisional size only to decide whether to send the
> signal; C++ recomputes the actual quantity from `risk_pct` / `max_leverage`.

### Phase 5 — C++ backtest (same code path as live)

**Goal:** full historical replay with fee + slippage, through the exact live path.

```powershell
python live_strategy/live_trend_bot.py --config config/trend_breakout_highfreq.yaml --no-warmup
cd live_engine/build_cmake && backtest_engine.exe
```

The C++ engine replays `data/historical_data/BTCUSDT_1m_full.csv`, publishes each
bar to Python over ZMQ, Python decides entries, C++ executes and manages the
bracket exit. Fee/slippage come from `shared/config.json` (see
[Config files](#config-files)).

### Phase 6 — Live incubation (Binance Testnet)

**Goal:** the same code trading live on Testnet. Identical to Phase 5 except the
data source (real-time WS) and the REST warmup.

```powershell
cd live_engine/build_cmake && live_engine.exe
python live_strategy/live_trend_bot.py --config config/trend_breakout_highfreq.yaml
```

---

## Running a backtest / live (detailed)

The Python brain is `live_strategy/live_trend_bot.py`. It supports either a YAML
config (recommended) or individual CLI flags.

### `--config` (recommended)

```powershell
python live_strategy/live_trend_bot.py --config config/trend_breakout_highfreq.yaml --no-warmup
```

`--config` reads the **same YAML** the research pipeline uses — trigger, periods,
model, and execution all in one file. This is the "one file per strategy" workflow:
switching strategies is a config change, never a code change.

### Without `--config` (CLI flags)

```powershell
python live_strategy/live_trend_bot.py --no-warmup \
  --trigger trend_breakout --period 2160 --trend-period 8640 --trail-period 1080 \
  --classifier \
  --model research/outputs/trend_breakout_highfreq_model.json \
  --features research/outputs/trend_breakout_highfreq_features.json \
  --threshold 0.5
```

Full flag list: run `python live_strategy/live_trend_bot.py --help`.

### `--no-warmup` — what it means

`--no-warmup` turns **off the REST warmup**. It is used only for the **C++
backtest**.

| | With `--no-warmup` (backtest) | Without (live) |
|---|---|---|
| Data source | C++ replays the CSV over ZMQ | Binance REST warmup + real-time WS |
| Indicator warmup | indicators fill naturally as bars replay | REST pre-fills them so the first signal fires immediately |
| kline printing | off (avoids slowing a multi-million-bar replay) | on (`📈 [Feeder] #N kline …`) |

In **live**, the warmup window is `max(period, trail_period, trend_period)` — e.g.
8640 bars (6 days) for high-freq, 28800 bars (20 days) for low-freq. Without warmup
you would wait that long for the first signal.

> `--no-warmup` does **not** mean "no warmup at all" — it means "don't warm from a
> REST fetch". Indicators still warm; only the source changes.

---

## Execution parameterization

How an order is placed and how an unfilled order is handled is a strategy decision,
so it lives in the strategy YAML (and travels with each order from Python to C++).
There is no per-strategy execution code in C++.

```yaml
execution:
  entry:
    order_type: market          # market | limit | limit_maker
    time_in_force: gtc          # gtc | ioc | fok | gtx   (limit orders only)
    timeout_ms: 0               # 0 = never chase (correct for market)
    unfilled_policy: cancel     # cancel | reprice | market
    max_reprice_attempts: 2     # only used by reprice
  exit:
    order_type: market          # stops should be aggressive
    time_in_force: gtc
    timeout_ms: 3000
    unfilled_policy: market     # if the stop close doesn't fill, hard-eat at market
    max_reprice_attempts: 2
```

Field meanings:

| Field | Values | Meaning |
|-------|--------|---------|
| `order_type` | `market` / `limit` / `limit_maker` | `limit_maker` = post-only (GTX) |
| `time_in_force` | `gtc` / `ioc` / `fok` / `gtx` | only applies to limit orders |
| `timeout_ms` | int | how long an open order may sit before `unfilled_policy` fires |
| `unfilled_policy` | `cancel` / `reprice` / `market` | on timeout: give up / chase (re-price) / hard-eat at market |
| `max_reprice_attempts` | int | chase re-prices before giving up |

Defaults (aggressive, right for breakout): **entry = market**, **exit = market with
a 3 s timeout → market**. A mean-reversion or maker-rebate strategy would override
to `limit` / `limit_maker` + `reprice`.

> **Breakout + limit = adverse selection.** A resting limit on a breakout only
> fills when price comes back (the weak breakout) and misses the strong ones that
> run away. Use `market` for momentum entries.

---

## Config files

### `config/*.yaml` — strategy definition (research + live)

One file per strategy. Declares `trigger`, `indicators`, `features`, `labeler`,
`model` (optional), and `execution`:

```yaml
trigger:
  type: "research.triggers.trend_breakout.TrendFilteredBreakoutTrigger"
  params: { entry_period: 2160, trend_period: 8640, long_only: true }
indicators: { entry_period: 2160, exit_period: 1080, atr_period: 2160, ma_period: 8640 }
features: [ ... ]
labeler: { type: "research.labeling.TrailingExitLabeler", params: { trail_period: 1080 } }
model: { type: "xgboost_classifier", output_prefix: "trend_breakout_highfreq" }
execution: { entry: { ... }, exit: { ... } }
```

### `shared/config.json` — API keys + ports + backtest costs

```json
{
  "api_key": "…", "secret_key": "…",
  "zmq": { "market_feed_port": 5555, "signal_port": 5556 },
  "backtest": {
    "initial_balance": 100000.0,
    "fee_rate": 0.0005,
    "slippage_bps": 1.0,
    "risk_pct": 0.01,
    "max_leverage": 20.0
  }
}
```

`fee_rate` (per side) and `slippage_bps` (per side) are read by the C++ backtest —
change them here to reprice friction costs without rebuilding.

---

## Adding a new strategy

A strategy is a set of **pluggable objects** wired together. You write (or reuse)
components; you never edit the engine.

### The pluggable objects

| Object | ABC | Write / edit | File |
|--------|-----|--------------|------|
| Trigger (entry) | `BaseEventTrigger.generate_signals()` | usually write | `research/triggers/<name>.py` |
| Labeler (barrier / exit) | `BaseLabeler.compute_labels()` | usually write | `research/labeling.py` |
| Feature set (ML input) | `BaseFeature.compute()` / `compute_one()` | usually reuse | `research/features.py` |
| Sizer (size) | `BasePositionSizer.calculate_size()` | usually reuse | `execution/sizers.py` |
| Risk manager (gate) | `BaseRiskManager.check_risk_limits()` | usually reuse | `execution/risk_managers.py` |
| Execution spec | — | config | `config/<name>.yaml` `execution:` block |

### The three research → execution bridges

Only three artifacts cross from research into execution:

1. **Model + feature list** — `model.json` + `features.json` (the ML filter).
2. **Indicator formulas** — `research/indicator_spec.py` `ROLLING_SPEC` is the single
   source of truth; both the vectorized `add_indicators` and the streaming
   `IncrementalIndicators` interpret it.
3. **Execution spec** — the `execution:` block in the strategy YAML, serialized into
   `OrderPayload` and honored by C++.

### Checklist

1. **Write the trigger** — a new `BaseEventTrigger` in `research/triggers/`.
2. **Write the labeler** (the barrier) — a new `BaseLabeler` in
   `research/labeling.py`. This drives research labeling *and*
   `lightweight_backtest`.
3. **Add indicators if needed** — one line in `ROLLING_SPEC`
   (`research/indicator_spec.py`); both backends pick it up automatically.
4. **C++ exit** — if your exit is in the fixed set (fixed stop / trailing
   Donchian low·high / MA), set `bracket`/`trailing_exit_indicator` in config — **no
   C++ change**. If it's a brand-new exit mechanism, add one branch in
   `live_engine/src/core/trailing_stop.cpp` and rebuild.
5. **Set the execution spec** — `execution:` block in the YAML (or `--exec-*` flags).
6. **Wire it in config** — point `trigger` / `features` / `labeler` / `model` /
   `execution` at your components in `config/<name>.yaml`.
7. **Validate** — Phase 1 (trigger) → Phase 3 (ML) → Phase 3b (vectorized) → Phase 5
   (C++ backtest).

---

## Directory structure

```
core/                          ABC contracts + the execution data contract
  trigger.py  feature.py  labeler.py            (Phase 1-3 ABCs)
  position_sizer.py  risk_manager.py            (Phase 4 ABCs)
  order_payload.py  execution_spec.py           (Python → C++ order + execution)
  strategy_wrapper.py  data_feeder.py  execution_gateway.py
execution/                     Concrete sizers + risk managers
  sizers.py  risk_managers.py
research/                      Research toolkit (Phase 1-3)
  triggers/                    BaseEventTrigger implementations
  features.py  features_incremental.py          (vectorized + streaming indicators)
  indicator_spec.py            ROLLING_SPEC — single source of truth
  labeling.py  dataset_builder.py  evaluator.py  backtest.py
  trigger_analysis.py  param_sweep.py  pipeline_runner.py
  outputs/                     model.json + features.json artifacts
live_strategy/                 Execution layer (Phase 5-6)
  live_trend_bot.py            composition shell + --config + CLI
  zmq_client.py  zmq_feeder.py  zmq_gateway.py
config/                        YAML strategy assembly
shared/                        config.json (keys, ports, backtest costs)
live_engine/                   C++ engine
  src/core/                    risk manager, trailing stop, IPC, executor interface
  src/backtest/                CSV replayer + mock executor
  src/live/                    Binance WS + live executor + order tracker
scripts/                       Phase 1/3/3b run scripts
tests/                         unit + parity tests
```

---

## Tests

```bash
python tests/test_order_payload.py          # OrderPayload + execution spec
python tests/test_strategy_wrapper.py       # StrategyWrapper state machine
python tests/test_parity.py                 # ABC ↔ legacy parity
python tests/test_trailing_stop_parity.py   # C++ trailing stop ↔ Python labeler
```

---

## Convenience scripts

PowerShell helpers in the repo root wrap the build + two-terminal flows.

| Script | Purpose |
|--------|---------|
| `build.ps1` | Rebuild `live_engine.exe` + `backtest_engine.exe` (CMake + Ninja). |
| `run_backtest.ps1` | Launch Python brain + `backtest_engine.exe` in two windows. |
| `run_live.ps1` | Launch Python brain + `live_engine.exe` in two windows. |

```powershell
.\build.ps1
.\run_backtest.ps1     # or .\run_live.ps1
```

> ⚠️ `run_backtest.ps1` / `run_live.ps1` currently launch the brain with
> `--no-warmup` and **no strategy flags** (the default Adam strategy). To run a
> specific strategy through them, either edit the script to pass
> `--config config/<name>.yaml`, or run the two terminals manually as shown in
> [Quick start](#quick-start).
