"""
ExecutionSpec — how an order is placed and how unfilled orders are handled.

Single source of truth for execution parameters. A spec is serialized into
OrderPayload and sent over ZMQ so the C++ engine places/manages orders the way
the strategy wants — without any per-strategy code in C++.

Two specs ride on every entry order:
  * entry execution — how the entry itself is placed + chased.
  * exit  execution — how the trailing-stop / hard-stop close is placed.

Breakout entries are aggressive (market); stops are aggressive (market) with a
short timeout → market fallback. Mean-reversion or maker-rebate strategies can
override to LIMIT / LIMIT_MAKER + REPRICE without touching C++.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict


class OrderType(str, Enum):
    MARKET = "market"            # take liquidity immediately
    LIMIT = "limit"              # rest at a price (GTC/IOC/FOK)
    LIMIT_MAKER = "limit_maker"  # post-only (GTX), never takes liquidity


class TimeInForce(str, Enum):
    GTC = "gtc"                  # good-till-cancel
    IOC = "ioc"                  # immediate-or-cancel
    FOK = "fok"                  # fill-or-kill
    GTX = "gtx"                  # post-only (Binance synonym for LIMIT_MAKER)


class UnfilledPolicy(str, Enum):
    CANCEL = "cancel"            # cancel and give up
    REPRICE = "reprice"          # cancel + re-place at market (chase)
    MARKET = "market"            # cancel + hard-eat with a market order


@dataclass(frozen=True)
class ExecutionSpec:
    """
    Execution parameters for one side of a trade.

    order_type:             MARKET | LIMIT | LIMIT_MAKER
    time_in_force:          GTC | IOC | FOK | GTX (only used for LIMIT)
    timeout_ms:             how long an open order may sit unfilled before
                            unfilled_policy kicks in. 0 = never chase
                            (correct for MARKET, which fills immediately).
    unfilled_policy:        CANCEL | REPRICE | MARKET — action on timeout.
    max_reprice_attempts:   chase re-prices before giving up (REPRICE only).
    """
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.GTC
    timeout_ms: int = 0
    unfilled_policy: UnfilledPolicy = UnfilledPolicy.CANCEL
    max_reprice_attempts: int = 2

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_type": self.order_type.value,
            "time_in_force": self.time_in_force.value,
            "timeout_ms": self.timeout_ms,
            "unfilled_policy": self.unfilled_policy.value,
            "max_reprice_attempts": self.max_reprice_attempts,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExecutionSpec":
        if not d:
            return cls()
        return cls(
            order_type=OrderType(d.get("order_type", "market")),
            time_in_force=TimeInForce(d.get("time_in_force", "gtc")),
            timeout_ms=int(d.get("timeout_ms", 0)),
            unfilled_policy=UnfilledPolicy(d.get("unfilled_policy", "cancel")),
            max_reprice_attempts=int(d.get("max_reprice_attempts", 2)),
        )


# Recommended defaults. Breakout entries are momentum — take liquidity rather
# than getting adversely selected by a resting limit. Stops must be aggressive.
DEFAULT_ENTRY_EXECUTION = ExecutionSpec(order_type=OrderType.MARKET)

DEFAULT_EXIT_EXECUTION = ExecutionSpec(
    order_type=OrderType.MARKET,
    timeout_ms=3000,
    unfilled_policy=UnfilledPolicy.MARKET,
)
