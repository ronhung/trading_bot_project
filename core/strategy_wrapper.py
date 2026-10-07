"""
Phase 5-6: StrategyWrapper — bridges research (Phase 3) to execution (Phase 5/6).

Loads a trained ML model and assembles trigger + features + sizer + risk
from dependency-injected components. Called per-bar by the execution engine.

RESPONSIBILITY BOUNDARY:
  Python handles ENTRY only. Once an OrderPayload is sent, C++ manages the
  full bracket-order lifecycle. Python waits for POSITION_CLOSED callback
  before re-enabling entry detection.
"""

from enum import Enum, auto
from typing import Optional, Dict, Any, List
import logging

import numpy as np
import pandas as pd

from core.order_payload import OrderPayload, Action, TrailingExitIndicator, BracketExit
from core.execution_spec import (
    ExecutionSpec,
    DEFAULT_ENTRY_EXECUTION,
    DEFAULT_EXIT_EXECUTION,
)
from core.trigger import BaseEventTrigger
from core.feature import BaseFeature
from core.position_sizer import BasePositionSizer
from core.risk_manager import BaseRiskManager
from research.features_incremental import IncrementalIndicators

logger = logging.getLogger(__name__)


class StrategyState(Enum):
    """Python-side position tracking state."""
    IDLE = auto()            # detecting entries
    WAITING_CLOSE = auto()   # position open; C++ manages exit


class StrategyWrapper:
    """
    Bridge: Phase 3 trained model → Phase 5/6 execution engine.

    On each bar:
      1. If WAITING_CLOSE → return None (C++ is managing the exit).
      2. Check risk limits → block if exceeded.
      3. Compute signal via trigger.
      4. If entry signal → compute feature vector → ML model predict.
      5. If prediction > threshold → calculate size via sizer.
      6. Assemble OrderPayload with bracket exit parameters.
      7. Set state = WAITING_CLOSE, return OrderPayload.

    C++ sends POSITION_CLOSED → on_position_closed() → state = IDLE.
    """

    def __init__(
        self,
        trigger: BaseEventTrigger,
        feature: BaseFeature,
        model: Optional[Any],
        feature_names: List[str],
        sizer: BasePositionSizer,
        risk_manager: BaseRiskManager,
        bracket_exit: Optional[BracketExit] = None,
        indicator_params: Optional[Dict[str, Any]] = None,
        signal_threshold: float = 0.0,
        symbol: str = "BTCUSDT",
        sync_close_from_state: bool = False,
        entry_execution: Optional[ExecutionSpec] = None,
        exit_execution: Optional[ExecutionSpec] = None,
    ):
        """
        Args:
            trigger: Event trigger for entry detection (pluggable).
            feature: Feature computer for ML input vector.
            model: Trained ML model with .predict(X) method.
            feature_names: Ordered list of feature names matching model input.
            sizer: Position size calculator.
            risk_manager: Risk gate (False = block all entries).
            bracket_exit: BracketExit exit spec (trailing rule + hard stop mode).
            indicator_params: dict passed to add_indicators() for feature + exit.
            signal_threshold: Minimum model prediction to fire.
            symbol: Trading pair.
        """
        self._trigger = trigger
        self._feature = feature
        self._model = model
        self._feature_names = feature_names
        self._sizer = sizer
        self._risk_manager = risk_manager
        self._signal_threshold = signal_threshold
        self._symbol = symbol
        self._sync_close_from_state = sync_close_from_state
        self._bracket_exit = bracket_exit or BracketExit()
        self._indicator_params = indicator_params or {}

        self._state: StrategyState = StrategyState.IDLE
        self._saw_position = False
        # How this strategy's orders are placed + how unfilled orders are handled.
        self._entry_execution = entry_execution or DEFAULT_ENTRY_EXECUTION
        self._exit_execution = exit_execution or DEFAULT_EXIT_EXECUTION
        # O(1) streaming indicators — replaces recomputing add_indicators over
        # the whole lookback buffer on every bar (was O(n^2) total).
        self._incremental = IncrementalIndicators(**self._indicator_params)

    # -- public API -------------------------------------------------------

    @property
    def state(self) -> StrategyState:
        return self._state

    def on_bar(
        self, bar_data: Dict[str, Any], portfolio_state: Dict[str, Any]
    ) -> Optional[OrderPayload]:
        """
        Called each time step by the execution engine.

        Args:
            bar_data: Dict with OHLCV fields (open, high, low, close, volume,
                      open_time, close_time, ...). Matches C++ KLineData struct.
            portfolio_state: Dict with C++ RiskManager state:
                current_position, available_balance, stop_price.

        Returns:
            OrderPayload if an entry should be placed, None otherwise.
        """
        # --- Always update rolling indicators (even while WAITING_CLOSE) so the
        #     lookback stays full-history once a position closes ---
        row = self._incremental.update(bar_data)

        # --- Gate: waiting for C++ to close position ---
        cur_pos = portfolio_state.get("current_position", 0.0)
        if self._state == StrategyState.WAITING_CLOSE:
            # The backtest engine has no POSITION_CLOSED message; it syncs
            # current_position into every kline. When sync_close_from_state is
            # enabled (backtest only — fills are synchronous there), a zero
            # position means the exit has fired and we can re-arm. Live keeps
            # using on_position_closed() and must NOT use this (async fills).
            #
            # NOTE: the entry bar's kline is published BEFORE the fill, so it
            # carries current_position==0. We must only re-arm after we have
            # actually OBSERVED a non-zero position (a real open→closed
            # transition), otherwise the brain re-enters on every bar while a
            # position is open (duplicate orders).
            if cur_pos != 0.0:
                self._saw_position = True
            if self._sync_close_from_state and self._saw_position and cur_pos == 0.0:
                self._state = StrategyState.IDLE
                self._saw_position = False
                logger.debug("Position closed (synced from C++); re-arming entries")
            else:
                return None

        # --- Gate: risk limits ---
        if not self._risk_manager.check_risk_limits(portfolio_state):
            logger.debug("Risk limits blocked entry")
            return None

        # --- Entry signal via injected trigger (pluggable) ---
        # Pass the raw indicator dict to avoid building a DataFrame on every bar
        # (a ~0.5ms pandas overhead that dominates the streaming hot loop).
        signal = self._trigger.generate_signals(row)
        if isinstance(signal, pd.Series):
            signal = int(signal.iloc[-1])
        else:
            signal = int(signal)
        if signal not in (1, -1):
            return None  # no entry event at this bar

        action = Action.BUY if signal == 1 else Action.SELL

        # --- Compute features at current bar (only on entry events) ---
        ind = pd.DataFrame([row])
        feat_dict = self._feature.compute_one(ind, len(ind) - 1)

        # --- ML filter ---
        ml_score: float = 0.0
        if self._model is not None:
            feature_vec = np.array(
                [[feat_dict.get(name, 0.0) for name in self._feature_names]],
                dtype=np.float32,
            )
            if hasattr(self._model, "predict_proba"):
                # classifier: score = P(win); filter keeps only high-prob entries
                ml_score = float(self._model.predict_proba(feature_vec)[0][1])
            else:
                ml_score = float(self._model.predict(feature_vec)[0])
            if ml_score <= self._signal_threshold:
                logger.debug(
                    "ML filter suppressed %s: pred=%.4f <= threshold=%.2f",
                    action.value, ml_score, self._signal_threshold,
                )
                return None

        # --- Compute position size ---
        close = float(bar_data["close"])
        atr_val = feat_dict.get("atr", 0.0)
        if atr_val <= 0:
            atr_val = 1.0  # defensive fallback

        trailing_low = float(ind["exit_low"].iloc[-1]) if "exit_low" in ind.columns else None
        trailing_high = float(ind["exit_high"].iloc[-1]) if "exit_high" in ind.columns else None

        equity = float(portfolio_state.get("available_balance", 0.0))
        stop_dist = self._bracket_exit.stop_distance(
            signal, close, atr_val, trailing_low, trailing_high
        )
        size = self._sizer.calculate_size(
            signal_strength=self._bracket_exit.atr_mult,
            current_atr=atr_val,
            account_equity=equity,
            entry_price=close,
            stop_distance=stop_dist,
        )
        if size <= 0.0:
            logger.debug("Sizer returned zero size; skipping entry")
            return None

        # --- Compute bracket exit parameters from the exit spec ---
        hard_stop = self._bracket_exit.initial_stop(
            signal, close, atr_val, trailing_low, trailing_high
        )

        # --- Assemble OrderPayload ---
        # The trailing indicator is side-specific: a long trails the Donchian
        # low (exit when price < N-bar low), a short trails the Donchian high
        # (exit when price > N-bar high). moving_average is side-agnostic.
        trailing_indicator = self._bracket_exit.trailing_exit_indicator
        if action == Action.SELL and trailing_indicator == TrailingExitIndicator.DONCHIAN_LOW:
            trailing_indicator = TrailingExitIndicator.DONCHIAN_HIGH
        elif action == Action.BUY and trailing_indicator == TrailingExitIndicator.DONCHIAN_HIGH:
            trailing_indicator = TrailingExitIndicator.DONCHIAN_LOW

        order = OrderPayload(
            action=action,
            symbol=self._symbol,
            quantity=size,
            entry_price=close,
            hard_stop_loss=hard_stop if hard_stop is not None else 0.0,
            trailing_exit_indicator=trailing_indicator,
            trailing_exit_period=self._bracket_exit.trailing_exit_period,
            entry_execution=self._entry_execution,
            exit_execution=self._exit_execution,
        )

        # --- Transition to WAITING_CLOSE ---
        self._state = StrategyState.WAITING_CLOSE
        logger.info(
            "ENTRY %s: price=%.2f size=%.4f stop=%.2f ml_score=%.4f",
            action.value, close, size, hard_stop, ml_score,
        )

        return order

    def on_position_closed(self, close_info: Dict[str, Any]) -> None:
        """
        Callback from execution engine: position has been fully closed.

        Args:
            close_info: Dict with reason, entry_price, exit_price, pnl.
        """
        reason = close_info.get("reason", "unknown")
        pnl = close_info.get("pnl", 0.0)
        logger.info("POSITION_CLOSED: reason=%s pnl=%.2f", reason, pnl)
        self._state = StrategyState.IDLE

    def on_order_update(self, update: Dict[str, Any]) -> None:
        """
        Callback from execution engine on each order status update.

        Handles the case where an ENTRY order is abandoned before filling
        (timeout/expire/reject). C++ only manages the exit lifecycle once a
        position exists, so a never-filled entry would otherwise leave the brain
        stuck in WAITING_CLOSE forever. Here we reset to IDLE so entry detection
        resumes.
        """
        status = update.get("status", "")
        reduce_only = update.get("reduce_only", False)
        filled_qty = float(update.get("filled_quantity", 0.0))
        reason = update.get("reason", "")

        # Only a non-reduce-only (entry) order that terminal'd WITHOUT filling
        # matters here. FILLED is the normal open; a partial fill leaves a live
        # position (handled by sync_close_from_state); a cancel that is followed
        # by a reprice/market keeps the entry alive — ignore those.
        if reduce_only or self._state != StrategyState.WAITING_CLOSE:
            return
        if status not in ("CANCELED", "EXPIRED", "REJECTED"):
            return
        if filled_qty > 0.0:
            return
        if reason in ("timeout_reprice", "timeout_market"):
            return  # another order follows; entry still pending

        logger.info(
            "ENTRY abandoned (status=%s reason=%s); re-arming entries",
            status, reason,
        )
        self._state = StrategyState.IDLE
        self._saw_position = False

    def reset(self) -> None:
        """Reset state and clear indicator history (e.g., for backtest restart)."""
        self._state = StrategyState.IDLE
        self._saw_position = False
        self._incremental = IncrementalIndicators(**self._indicator_params)

    def warmup(self, bars: List[Dict[str, Any]]) -> None:
        """Feed historical bars into the incremental indicators (live cold-start).

        No entry detection is performed — this only warms the rolling lookback
        so the first live signal fires immediately instead of waiting for the
        full indicator window.
        """
        for bar in bars:
            self._incremental.update(bar)

    def warmup_bars_needed(self) -> int:
        """Number of historical bars required to fully warm the indicators."""
        return max(self._indicator_params.values()) if self._indicator_params else 0

    @classmethod
    def from_yaml(cls, config_path: str) -> "StrategyWrapper":
        """
        Factory: instantiate StrategyWrapper from a YAML config file.

        The config must contain: trigger, features, position_sizer,
        risk_manager, model (optional), bracket, execution sections.

        Uses the same component resolution as pipeline_runner.py
        (importlib-based dynamic instantiation from type paths).
        """
        import importlib
        import json

        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        def _instantiate(spec: dict):
            type_path = spec["type"]
            params = spec.get("params", {})
            module_path, class_name = type_path.rsplit(".", 1)
            module = importlib.import_module(module_path)
            cls_ = getattr(module, class_name)
            return cls_(**params)

        # Trigger (pluggable entry rule)
        trigger = _instantiate(cfg["trigger"])

        # Features
        feature_specs = cfg["features"]
        if isinstance(feature_specs, list):
            from research.features import CompositeFeature
            feature = CompositeFeature([_instantiate(s) for s in feature_specs])
        else:
            feature = _instantiate(feature_specs)

        # Sizer + risk
        sizer = _instantiate(cfg["position_sizer"])
        risk_manager = _instantiate(cfg["risk_manager"])

        # Model (optional)
        model = None
        feature_names: list = []
        model_cfg = cfg.get("model", {})
        if model_cfg.get("path"):
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(model_cfg["path"])
            with open(model_cfg["feature_list"], "r") as f:
                feature_names = json.load(f)

        # Execution
        exec_cfg = cfg.get("execution", {})
        bracket_cfg = cfg.get("bracket", {})

        trailing_map = {
            "donchian_low": TrailingExitIndicator.DONCHIAN_LOW,
            "donchian_high": TrailingExitIndicator.DONCHIAN_HIGH,
            "moving_average": TrailingExitIndicator.MOVING_AVERAGE,
        }

        bracket_exit = BracketExit(
            trailing_exit_indicator=trailing_map.get(
                bracket_cfg.get("trailing_exit_indicator", "donchian_low"),
                TrailingExitIndicator.DONCHIAN_LOW,
            ),
            trailing_exit_period=bracket_cfg.get("trailing_exit_period", 14400),
            hard_stop_mode=bracket_cfg.get("hard_stop_mode", "donchian"),
            atr_mult=cfg["trigger"]["params"].get("atr_mult", 2.0),
        )
        indicator_params = {
            "entry_period": cfg["trigger"]["params"].get("entry_period", 20),
            "exit_period": bracket_cfg.get("trailing_exit_period", 14400),
            "atr_period": cfg["trigger"]["params"].get("atr_period", 20),
        }

        # Execution spec (entry/exit order handling) — optional; defaults to the
        # aggressive market defaults when absent.
        entry_execution = DEFAULT_ENTRY_EXECUTION
        exit_execution = DEFAULT_EXIT_EXECUTION
        if exec_cfg.get("entry"):
            entry_execution = ExecutionSpec.from_dict(exec_cfg["entry"])
        if exec_cfg.get("exit"):
            exit_execution = ExecutionSpec.from_dict(exec_cfg["exit"])

        return cls(
            trigger=trigger,
            feature=feature,
            model=model,
            feature_names=feature_names,
            sizer=sizer,
            risk_manager=risk_manager,
            bracket_exit=bracket_exit,
            indicator_params=indicator_params,
            signal_threshold=model_cfg.get("threshold", 0.0),
            symbol=exec_cfg.get("symbol", "BTCUSDT"),
            entry_execution=entry_execution,
            exit_execution=exit_execution,
        )
