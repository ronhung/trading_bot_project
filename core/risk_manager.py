"""Phase 4: Risk Manager — system-level risk gates. Returns False to block entries."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseRiskManager(ABC):
    """
    Abstract base for risk management.

    Called before every entry decision. Returns True if new trades are allowed,
    False to block all new entries (e.g., max drawdown exceeded, daily loss limit).

    Subclasses must implement check_risk_limits().
    """

    @abstractmethod
    def check_risk_limits(self, current_portfolio_state: Dict[str, Any]) -> bool:
        """
        Evaluate risk limits against current portfolio state.

        Args:
            current_portfolio_state: Dict with keys like:
                equity           — current account equity
                peak_equity      — highest equity seen
                current_drawdown — current drawdown as decimal (0.05 = 5%)
                open_positions   — number of open positions
                available_balance— free margin
                position         — current position size (0 = flat)

        Returns:
            True if new entries are allowed, False to block.
        """
        ...
