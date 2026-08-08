"""Concrete risk manager implementations."""

from typing import Any, Dict

from core.risk_manager import BaseRiskManager


class MaxDrawdownRiskManager(BaseRiskManager):
    """
    Blocks all new entries when drawdown exceeds a threshold.

    Drawdown is computed as: (peak_equity - current_equity) / peak_equity.
    """

    def __init__(self, max_dd_pct: float = 0.05):
        """
        Args:
            max_dd_pct: Maximum allowed drawdown as decimal (0.05 = 5%).
        """
        self.max_dd_pct = max_dd_pct

    def check_risk_limits(self, current_portfolio_state: Dict[str, Any]) -> bool:
        current_drawdown = float(
            current_portfolio_state.get("current_drawdown", 0.0)
        )
        return current_drawdown < self.max_dd_pct


class AllowAllRiskManager(BaseRiskManager):
    """No risk limits — always allow entries. Useful as default in backtests."""

    def check_risk_limits(self, current_portfolio_state: Dict[str, Any]) -> bool:
        return True
