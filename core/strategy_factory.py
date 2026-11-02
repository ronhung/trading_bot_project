"""
Strategy Factory & Dynamic Reflection Engine.

Implements the Config-Driven Philosophy (Rule 1):
  - Every Alpha strategy can be assembled and configured purely via YAML.
  - Adding a new Alpha trigger requires only subclassing BaseEventTrigger
    and decorating with @register_trigger("name").
  - Zero modification to StrategyWrapper or live_trend_bot.py.
"""

import os
import sys
import json
import logging
import importlib
from typing import Dict, Any, Type, Callable, Optional, Tuple, List

import yaml

from core.execution_spec import ExecutionSpec, DEFAULT_ENTRY_EXECUTION, DEFAULT_EXIT_EXECUTION
from core.order_payload import BracketExit, TrailingExitIndicator
from core.trigger import BaseEventTrigger
from core.feature import BaseFeature
from core.position_sizer import BasePositionSizer
from core.risk_manager import BaseRiskManager
from core.strategy_wrapper import StrategyWrapper

logger = logging.getLogger(__name__)

TRIGGER_REGISTRY: Dict[str, Type[BaseEventTrigger]] = {}
SIZER_REGISTRY: Dict[str, Type[BasePositionSizer]] = {}
RISK_REGISTRY: Dict[str, Type[BaseRiskManager]] = {}


def register_trigger(name: str):
    """Decorator to register an event trigger class."""
    def decorator(cls: Type[BaseEventTrigger]):
        TRIGGER_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def register_sizer(name: str):
    """Decorator to register a position sizer class."""
    def decorator(cls: Type[BasePositionSizer]):
        SIZER_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def register_risk(name: str):
    """Decorator to register a risk manager class."""
    def decorator(cls: Type[BaseRiskManager]):
        RISK_REGISTRY[name.lower()] = cls
        return cls
    return decorator


_BUILTINS_REGISTERED = False


def _lazy_register_builtins():
    """Import and register standard builtin components."""
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    _BUILTINS_REGISTERED = True

    # Triggers
    from research.triggers.trend_breakout import TrendFilteredBreakoutTrigger
    from research.triggers.adam_breakout import AdamBreakoutTrigger
    from research.triggers.turtle_breakout import TurtleBreakoutTrigger
    from research.triggers.bollinger_mean_reversion import BollingerMeanReversionTrigger

    register_trigger("trend_breakout")(TrendFilteredBreakoutTrigger)
    register_trigger("adam_breakout")(AdamBreakoutTrigger)
    register_trigger("adam")(AdamBreakoutTrigger)
    register_trigger("turtle_breakout")(TurtleBreakoutTrigger)
    register_trigger("turtle")(TurtleBreakoutTrigger)
    register_trigger("bollinger_mean_reversion")(BollingerMeanReversionTrigger)

    from research.triggers.ofi_momentum import OFIMomentumTrigger
    from research.triggers.absorption_divergence import AbsorptionDivergenceTrigger
    from research.triggers.liquidation_climax import LiquidationClimaxTrigger
    from research.triggers.dynamic_vwap_band import DynamicVWAPBandTrigger
    from research.triggers.kalman_pairs import KalmanPairsTrigger

    register_trigger("ofi_momentum")(OFIMomentumTrigger)
    register_trigger("absorption_divergence")(AbsorptionDivergenceTrigger)
    register_trigger("liquidation_climax")(LiquidationClimaxTrigger)
    register_trigger("dynamic_vwap_band")(DynamicVWAPBandTrigger)
    register_trigger("kalman_pairs")(KalmanPairsTrigger)

    # Sizers
    from execution.sizers import FixedRiskSizer, VolatilityTargetingSizer, HalfKellySizer
    register_sizer("fixed_risk")(FixedRiskSizer)
    register_sizer("volatility_targeting")(VolatilityTargetingSizer)
    register_sizer("half_kelly")(HalfKellySizer)

    # Risk Managers
    from execution.risk_managers import MaxDrawdownRiskManager, AllowAllRiskManager, LivePositionGate
    register_risk("max_drawdown")(MaxDrawdownRiskManager)
    register_risk("allow_all")(AllowAllRiskManager)
    register_risk("live_position_gate")(LivePositionGate)


def _resolve_class(path_or_name: str, registry: Dict[str, Any]) -> Any:
    """Resolve a class by registry name or dotted module classpath."""
    clean_name = path_or_name.strip()
    if clean_name.lower() in registry:
        return registry[clean_name.lower()]
    if "." in clean_name:
        module_name, class_name = clean_name.rsplit(".", 1)
        mod = importlib.import_module(module_name)
        return getattr(mod, class_name)
    raise ValueError(f"Could not resolve class '{path_or_name}'. Available in registry: {list(registry.keys())}")


def parse_execution_spec(cfg: Optional[Dict[str, Any]]) -> ExecutionSpec:
    """Parse execution spec from a YAML dict."""
    if not cfg:
        return DEFAULT_ENTRY_EXECUTION
    return ExecutionSpec(
        order_type=cfg.get("order_type", "market"),
        time_in_force=cfg.get("time_in_force", "gtc"),
        timeout_ms=int(cfg.get("timeout_ms", 0)),
        unfilled_policy=cfg.get("unfilled_policy", "cancel"),
        max_reprice_attempts=int(cfg.get("max_reprice_attempts", 2)),
    )


def build_strategy_from_config(config_input: str | Dict[str, Any]) -> Tuple[StrategyWrapper, Dict[str, Any]]:
    """
    Build a fully assembled StrategyWrapper and metadata from a YAML file or dict.

    Returns:
        (strategy_wrapper, metadata_dict)
    """
    _lazy_register_builtins()

    if isinstance(config_input, str):
        with open(config_input, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = config_input

    # 1. Strategy identity & symbol
    strategy_cfg = cfg.get("strategy", {})
    symbol = strategy_cfg.get("symbol", cfg.get("symbol", "BTCUSDT"))

    # 2. Trigger instantiation
    raw_trigger = cfg.get("trigger", "trend_breakout")
    if isinstance(raw_trigger, dict):
        trigger_type_str = raw_trigger.get("type", "trend_breakout")
        trigger_params = raw_trigger.get("params", {})
    else:
        trigger_type_str = str(raw_trigger)
        trigger_params = cfg.get("trigger_params", {})

    trigger_cls = _resolve_class(trigger_type_str, TRIGGER_REGISTRY)

    if not trigger_params:
        entry_period = cfg.get("entry_period", cfg.get("period", 2160))
        trend_period = cfg.get("trend_period", entry_period * 4)
        if "trend" in trigger_type_str.lower():
            trigger_params = {"entry_period": entry_period, "trend_period": trend_period}
        else:
            trigger_params = {"period": entry_period}

    trigger = trigger_cls(**trigger_params)

    # 3. Features & Indicators
    from research.features import default_feature_set, CompositeFeature
    feature_list_cfg = cfg.get("features", [])
    features_list: List[BaseFeature] = []
    if isinstance(feature_list_cfg, list) and len(feature_list_cfg) > 0 and isinstance(feature_list_cfg[0], dict):
        for f_item in feature_list_cfg:
            f_type = f_item.get("type")
            f_params = f_item.get("params", {})
            if f_type:
                f_cls = _resolve_class(f_type, {})
                features_list.append(f_cls(**f_params))
    if features_list:
        features = CompositeFeature(features_list)
    else:
        features = default_feature_set()

    # 4. Exit / Bracket configuration & Indicator params
    ind_cfg = cfg.get("indicators", {})
    exit_cfg = cfg.get("exit", {})
    labeler_cfg = cfg.get("labeler", {}).get("params", {})
    
    trail_period = int(
        exit_cfg.get("trail_period")
        or labeler_cfg.get("trail_period")
        or ind_cfg.get("exit_period")
        or cfg.get("trail_period", 1080)
    )
    
    trail_indicator_str = exit_cfg.get("indicator", "donchian_low")
    try:
        trail_indicator = TrailingExitIndicator(trail_indicator_str.lower())
    except Exception:
        trail_indicator = TrailingExitIndicator.DONCHIAN_LOW

    bracket_exit = BracketExit(
        trailing_exit_indicator=trail_indicator,
        trailing_exit_period=trail_period,
        hard_stop_mode=exit_cfg.get("hard_stop_mode", "donchian"),
        atr_mult=float(exit_cfg.get("atr_mult", 2.0)),
    )

    indicator_params = {
        "entry_period": int(ind_cfg.get("entry_period", trigger_params.get("entry_period", trigger_params.get("period", 2160)))),
        "exit_period": trail_period,
        "atr_period": int(ind_cfg.get("atr_period", cfg.get("atr_period", 1440))),
        "vol_period": int(ind_cfg.get("vol_period", cfg.get("vol_period", 1440))),
        "ma_period": int(trigger_params.get("trend_period", ind_cfg.get("ma_period", 8640))),
    }

    # 5. Position Sizer & Risk Manager
    sizing_cfg = cfg.get("sizing", {})
    risk_pct = float(sizing_cfg.get("risk_pct", cfg.get("risk_pct", 0.02)))
    max_leverage = float(sizing_cfg.get("max_leverage", cfg.get("max_leverage", 20.0)))
    sizer_type = sizing_cfg.get("type", "fixed_risk").lower()
    sizer_cls = SIZER_REGISTRY.get(sizer_type, SIZER_REGISTRY["fixed_risk"])
    if sizer_type == "half_kelly":
        sizer = sizer_cls(
            win_rate=float(sizing_cfg.get("win_rate", 0.45)),
            profit_factor=float(sizing_cfg.get("profit_factor", 2.0)),
            max_risk_cap=float(sizing_cfg.get("max_risk_cap", 0.05)),
            max_leverage=max_leverage,
        )
    else:
        sizer = sizer_cls(risk_pct=risk_pct, max_leverage=max_leverage)

    risk_cfg = cfg.get("risk", {})
    max_dd_pct = float(risk_cfg.get("max_dd_pct", cfg.get("max_dd_pct", 0.15)))
    risk_cls = RISK_REGISTRY.get(risk_cfg.get("type", "max_drawdown").lower(), RISK_REGISTRY["max_drawdown"])
    risk_manager = risk_cls(max_dd_pct=max_dd_pct)

    # 6. ML Model & Feature list
    ml_cfg = cfg.get("ml_filter", cfg.get("model", {}))
    prefix = ml_cfg.get("output_prefix", None)
    
    model_path = ml_cfg.get("model_path", cfg.get("model_path", None))
    feature_list_path = ml_cfg.get("features_path", cfg.get("features_path", None))
    if not model_path and prefix:
        candidate_model = f"research/outputs/{prefix}_model.json"
        if os.path.exists(candidate_model):
            model_path = candidate_model
    if not feature_list_path and prefix:
        candidate_feats = f"research/outputs/{prefix}_features.json"
        if os.path.exists(candidate_feats):
            feature_list_path = candidate_feats

    threshold = float(ml_cfg.get("threshold", cfg.get("signal_threshold", 0.0)))
    classifier = (ml_cfg.get("type") == "xgboost_classifier") or bool(cfg.get("classifier", False))

    model = None
    feature_names = []
    if model_path and os.path.exists(model_path):
        import xgboost as xgb
        model = xgb.XGBClassifier() if classifier else xgb.XGBRegressor()
        model.load_model(model_path)
        logger.info("Loaded ML model from %s", model_path)

    if feature_list_path and os.path.exists(feature_list_path):
        with open(feature_list_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)

    # 7. Execution specs
    exec_cfg = cfg.get("execution", {})
    entry_exec = parse_execution_spec(exec_cfg.get("entry", None))
    exit_exec = parse_execution_spec(exec_cfg.get("exit", None))

    # 8. Assemble StrategyWrapper
    wrapper = StrategyWrapper(
        trigger=trigger,
        feature=features,
        model=model,
        feature_names=feature_names,
        sizer=sizer,
        risk_manager=risk_manager,
        bracket_exit=bracket_exit,
        indicator_params=indicator_params,
        signal_threshold=threshold,
        symbol=symbol,
        sync_close_from_state=True,
        entry_execution=entry_exec,
        exit_execution=exit_exec,
    )

    warmup_period = max(
        indicator_params.get("entry_period", 2160),
        indicator_params.get("exit_period", 1080),
        indicator_params.get("ma_period", 1440),
    )

    metadata = {
        "symbol": symbol,
        "warmup_period": warmup_period,
        "entry_execution": entry_exec,
        "exit_execution": exit_exec,
        "indicator_params": indicator_params,
        "model_loaded": model is not None,
    }

    return wrapper, metadata
