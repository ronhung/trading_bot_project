"""
Multi-Parameter Sweep Runner — grid-search with multiprocessing.

Core API:
    run_parameter_sweep(target_func, param_grid, raw_data, n_jobs=-1)
    -> DataFrame of all parameter combinations sorted by a key metric.

Includes ready-made target functions for:
    - ML dataset building (_dataset_target)
    - Lightweight backtesting (_backtest_target)

Non-intrusive: wraps existing functions, modifies no existing files.
Uses concurrent.futures.ProcessPoolExecutor for parallelism.
"""

import os
import sys
import itertools
import time
import traceback
import warnings
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ============================================================
# 1. Grid expansion
# ============================================================

def expand_param_grid(param_grid: Dict[str, list]) -> List[Dict[str, Any]]:
    """
    Expand a parameter grid dict into a list of individual param combinations.

    Parameters
    ----------
    param_grid : dict
        {"entry_period": [10, 20, 30], "intensity_threshold": [0.0, 0.1, 0.2]}
        Non-list values are treated as single-element lists.

    Returns
    -------
    list of dict
        One dict per parameter combination (cartesian product).
        Empty grid returns [{}].
    """
    if not param_grid:
        return [{}]

    keys = list(param_grid.keys())
    values = [
        v if isinstance(v, (list, tuple, np.ndarray)) else [v]
        for v in param_grid.values()
    ]

    combos = []
    for combo in itertools.product(*values):
        combos.append(dict(zip(keys, combo)))

    return combos


# ============================================================
# 2. Multiprocessing runner
# ============================================================

# Module-level worker function (required for Windows spawn pickling)
def _worker(payload: dict) -> dict:
    """
    Execute one parameter combination.

    Parameters
    ----------
    payload : dict with keys:
        target_func : callable (module-level, picklable)
        raw_data : pd.DataFrame
        combo : dict of param_name -> value
        fixed_kwargs : dict
        job_timeout : int

    Returns
    -------
    dict: {**combo, **metrics} or {**combo, "error": str}
    """
    target_func = payload["target_func"]
    raw_data = payload["raw_data"]
    combo = payload["combo"]
    fixed_kwargs = payload.get("fixed_kwargs", {})
    job_timeout = payload.get("job_timeout", 300)

    all_kwargs = {**combo, **fixed_kwargs}

    try:
        metrics = target_func(raw_data, **all_kwargs)
        result = {**combo}
        if isinstance(metrics, dict):
            result.update(metrics)
        else:
            result["_raw_result"] = str(metrics)[:200]
        return result
    except Exception:
        return {**combo, "error": traceback.format_exc()}


def run_parameter_sweep(
    target_func: Callable,
    param_grid: Dict[str, list],
    raw_data: pd.DataFrame,
    fixed_kwargs: Optional[Dict] = None,
    n_jobs: int = -1,
    rank_by: Optional[str] = None,
    ascending: bool = False,
    job_timeout: int = 300,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Grid-search parameter combinations with multiprocessing.

    Parameters
    ----------
    target_func : callable
        Module-level function: target_func(raw_data, **params) -> dict.
        Must return a dict of scalar metrics.
    param_grid : dict
        {"entry_period": [10, 20, 30], "atr_mult": [1.5, 2.0, 2.5]}
    raw_data : pd.DataFrame
        OHLCV data passed to every invocation of target_func.
    fixed_kwargs : dict, optional
        Fixed parameters merged into every call (not swept).
    n_jobs : int, default -1
        Number of parallel workers. -1 = all CPU cores, 1 = sequential.
    rank_by : str, optional
        Metric column to sort results by. Default: first metric not in param_grid.
    ascending : bool, default False
        Sort direction. False = best first (for Sharpe, win_rate, etc.).
    job_timeout : int, default 300
        Seconds before a single job is killed.
    verbose : bool, default True
        Print progress.

    Returns
    -------
    pd.DataFrame
        One row per parameter combination, sorted by rank_by.
        Columns: all swept params + all metric keys from target_func.
    """
    fixed_kwargs = fixed_kwargs or {}
    combos = expand_param_grid(param_grid)
    n_combos = len(combos)

    if n_combos == 0:
        return pd.DataFrame()

    if verbose:
        print(f"[sweep] {n_combos} parameter combinations, n_jobs={n_jobs}")

    # Build payloads
    payloads = [
        {
            "target_func": target_func,
            "raw_data": raw_data,
            "combo": combo,
            "fixed_kwargs": fixed_kwargs,
            "job_timeout": job_timeout,
        }
        for combo in combos
    ]

    # Determine n_jobs
    if n_jobs is None or n_jobs <= 0:
        import os as _os
        n_workers = _os.cpu_count() or 4
    else:
        n_workers = min(n_jobs, n_combos)

    results: List[dict] = []

    # --- Sequential path ---
    if n_workers == 1 or n_combos == 1:
        if verbose:
            print("[sweep] Running sequentially...")
        t0 = time.perf_counter()
        for i, payload in enumerate(payloads):
            if verbose and n_combos > 5 and i % max(1, n_combos // 10) == 0:
                print(f"  [{i + 1}/{n_combos}]")
            result = _worker(payload)
            results.append(result)
        elapsed = time.perf_counter() - t0

    # --- Parallel path ---
    else:
        if verbose:
            print(f"[sweep] Running with {n_workers} workers...")
        t0 = time.perf_counter()
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {executor.submit(_worker, p): i for i, p in enumerate(payloads)}
                completed = 0
                for future in futures:
                    try:
                        result = future.result(timeout=job_timeout + 30)
                        results.append(result)
                        completed += 1
                        if verbose and n_combos > 5 and completed % max(1, n_combos // 5) == 0:
                            print(f"  [{completed}/{n_combos}] completed")
                    except FutureTimeoutError:
                        combo = payloads[futures[future]]["combo"]
                        results.append({**combo, "error": "timeout"})
                        if verbose:
                            print(f"  [TIMEOUT] {combo}")
                    except Exception as e:
                        combo = payloads[futures[future]]["combo"]
                        results.append({**combo, "error": str(e)})
                        if verbose:
                            print(f"  [ERROR] {combo}: {e}")
        except Exception as e:
            warnings.warn(f"ProcessPoolExecutor failed ({e}), falling back to sequential")
            # Fallback to sequential
            results = []
            for i, payload in enumerate(payloads):
                if verbose and n_combos > 5 and i % max(1, n_combos // 10) == 0:
                    print(f"  [{i + 1}/{n_combos}]")
                result = _worker(payload)
                results.append(result)

        elapsed = time.perf_counter() - t0

    if verbose:
        print(f"[sweep] Completed in {elapsed:.1f}s ({elapsed / max(n_combos, 1):.2f}s per combo)")

    # --- Assemble DataFrame ---
    df = pd.DataFrame(results)

    # Determine sort column
    if rank_by is None:
        param_names = set(param_grid.keys())
        metric_cols = [c for c in df.columns if c not in param_names and c != "error"]
        if metric_cols:
            rank_by = metric_cols[0]

    if rank_by and rank_by in df.columns:
        df = df.sort_values(rank_by, ascending=ascending).reset_index(drop=True)

    return df


# ============================================================
# 3. Ready-made target functions (module-level for pickling)
# ============================================================

def _dataset_target(
    raw_data: pd.DataFrame,
    entry_period: int = 20,
    exit_period: int = 10,
    atr_period: int = 20,
    atr_mult: float = 2.0,
    intensity_threshold: float = 0.0,
    horizon: int = 288,
    tp: float = 0.02,
    sl: float = -0.01,
) -> dict:
    """
    Target function: runs build_ml_dataset and returns label metrics.

    Suitable for run_parameter_sweep(target_func=_dataset_target, ...)
    """
    from research.dataset_builder import build_ml_dataset, make_turtle_breakout_trigger
    from research.features import add_indicators, default_feature_pipeline

    df = add_indicators(raw_data, entry_period=entry_period, exit_period=exit_period,
                         atr_period=atr_period)
    trigger = make_turtle_breakout_trigger(
        entry_period=entry_period, atr_period=atr_period,
        atr_mult=atr_mult, intensity_threshold=intensity_threshold, signed=True,
    )
    pipeline = default_feature_pipeline()
    labeling_config = {"upper_barrier": tp, "lower_barrier": sl, "horizon": horizon}

    _, meta = build_ml_dataset(df, trigger, pipeline, labeling_config, verbose=False)

    return {
        "n_events": meta["n_events"],
        "label_1_rate": meta["label_rates"].get(1, 0.0),
        "label_neg1_rate": meta["label_rates"].get(-1, 0.0),
        "label_0_rate": meta["label_rates"].get(0, 0.0),
        "mean_return": meta["mean_actual_return"],
        "mean_bars_held": meta["mean_n_bars_held"],
    }


def _backtest_target(
    raw_data: pd.DataFrame,
    entry_period: int = 20,
    exit_period: int = 10,
    atr_period: int = 20,
    atr_mult: float = 2.0,
    intensity_threshold: float = 0.0,
    cash: float = 10000.0,
    risk_pct: float = 0.02,
    max_leverage: float = 20.0,
    position_sizer=None,
    risk_manager=None,
) -> dict:
    """
    Target function: runs lightweight_backtest and returns performance metrics.

    Suitable for run_parameter_sweep(target_func=_backtest_target, ...)

    Accepts optional position_sizer and risk_manager ABC instances for
    config-driven sweeps (Phase 3b parameter fine-tuning).
    """
    from research.backtest import lightweight_backtest

    result = lightweight_backtest(
        raw_data,
        entry_period=entry_period,
        exit_period=exit_period,
        atr_period=atr_period,
        atr_mult=atr_mult,
        intensity_threshold=intensity_threshold,
        initial_capital=cash,
        risk_pct=risk_pct,
        max_leverage=max_leverage,
        verbose=False,
        position_sizer=position_sizer,
        risk_manager=risk_manager,
    )

    return {
        "sharpe": result["sharpe"],
        "win_rate": result["win_rate"],
        "total_return_pct": result["total_return_pct"],
        "max_dd_pct": result["max_dd_pct"],
        "n_trades": result["n_trades"],
        "profit_factor": result["profit_factor"] if result["profit_factor"] != "inf" else 999.0,
        "avg_trade_pnl": result["avg_trade_pnl"],
    }


# ============================================================
# 4. Export utility
# ============================================================

def export_sweep_results(results: pd.DataFrame, filepath: str):
    """
    Export sweep results to CSV or Parquet.

    Parameters
    ----------
    results : pd.DataFrame
        Output from run_parameter_sweep().
    filepath : str
        Output path. Format inferred from extension (.csv or .parquet).
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    if filepath.endswith(".parquet"):
        results.to_parquet(filepath, index=False)
    else:
        results.to_csv(filepath, index=False)

    print(f"[export] Results written to {filepath} ({len(results)} rows)")


# ============================================================
# __main__ demo
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Testing research.param_sweep -- Multi-Parameter Sweep Runner")
    print("=" * 60)

    # Generate data
    from research.dataset_builder import make_synthetic_ohlcv
    print("\n[1] Generating synthetic OHLCV data (6,000 bars for speed)...")
    df = make_synthetic_ohlcv(n_bars=6000, seed=11)

    # --- Sweep 1: ML dataset target ---
    print("\n[2] Sweep 1: ML dataset labeling (3x2x2 = 12 combos)...")
    param_grid_ml = {
        "entry_period": [10, 20, 40],
        "horizon": [120, 288],
        "tp": [0.01, 0.02],
    }
    fixed_ml = {"sl": -0.01}

    results_ml = run_parameter_sweep(
        target_func=_dataset_target,
        param_grid=param_grid_ml,
        raw_data=df,
        fixed_kwargs=fixed_ml,
        n_jobs=2,
        rank_by="label_1_rate",
        ascending=False,
    )
    print(f"\n    ML sweep results ({len(results_ml)} rows, top 5):")
    pd.set_option("display.max_columns", 12)
    pd.set_option("display.width", 160)
    print(results_ml.head(5).to_string())

    # --- Sweep 2: Backtest target ---
    print(f"\n[3] Sweep 2: Lightweight backtest (3x3 = 9 combos)...")
    param_grid_bt = {
        "entry_period": [10, 20, 40],
        "atr_mult": [1.5, 2.0, 3.0],
    }
    fixed_bt = {"exit_period": 10, "atr_period": 20, "cash": 10000.0}

    results_bt = run_parameter_sweep(
        target_func=_backtest_target,
        param_grid=param_grid_bt,
        raw_data=df,
        fixed_kwargs=fixed_bt,
        n_jobs=2,
        rank_by="sharpe",
    )
    print(f"\n    Backtest sweep results ({len(results_bt)} rows, sorted by Sharpe):")
    print(results_bt.to_string())

    # Export to CSV
    print(f"\n[4] Exporting results...")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    export_sweep_results(results_bt, os.path.join(out_dir, "sweep_backtest.csv"))

    # Grid expansion check
    print(f"\n[5] Grid expansion check:")
    test_grid = {"a": [1, 2], "b": [10, 20, 30]}
    expanded = expand_param_grid(test_grid)
    print(f"    {test_grid} -> {len(expanded)} combos: {expanded}")

    print("\n" + "=" * 60)
    print("Parameter sweep demo completed successfully")
    print(f"(Results exported to research/outputs/)")
    print("=" * 60)
