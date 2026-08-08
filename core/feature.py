"""Phase 2: Feature — compute features at event locations with lookahead-bias prevention."""

from abc import ABC, abstractmethod
from typing import Dict

import pandas as pd


class BaseFeature(ABC):
    """
    Abstract base for feature computation.

    Features are computed at event positions (from a trigger). Subclasses MUST
    guarantee zero lookahead bias — no data from after the event bar may leak
    into the feature value.

    Two compute paths:
      - compute(): batch mode — all events at once → DataFrame.
      - compute_one(): single-bar mode — one bar index → dict (live/real-time path).
    """

    @abstractmethod
    def compute(self, data: pd.DataFrame, events: pd.Series) -> pd.DataFrame:
        """
        Compute features for all event positions.

        Args:
            data: OHLCV DataFrame with precomputed indicators.
                  All indicator columns must be .shift(1)'d so row `i`
                  only uses data from bars <= i.
            events: pd.Series from BaseEventTrigger.generate_signals().
                    Values in {-1, 0, 1} or boolean. Non-zero/True = event.

        Returns:
            pd.DataFrame indexed by event position (integer index into `data`),
            with one column per feature.
        """
        ...

    @abstractmethod
    def compute_one(self, data: pd.DataFrame, idx: int) -> Dict[str, float]:
        """
        Compute features for a single bar index.

        Used by the live/real-time path where only the current bar matters.

        Args:
            data: OHLCV DataFrame with precomputed indicators.
            idx: Integer position of the bar to compute features for.

        Returns:
            Dict mapping feature_name → float value.
        """
        ...
