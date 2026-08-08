"""Phase 2: Labeler — compute supervised labels for each event."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseLabeler(ABC):
    """
    Abstract base for label computation.

    A labeler takes OHLCV data and event positions, then computes the forward
    outcome (label) for each event. Label types supported:
      - Triple-barrier: +1 (take-profit), -1 (stop-loss), 0 (timeout).
      - Fixed-horizon: continuous y_norm = forward_return / ATR.
      - Classification: binary up/down.

    Subclasses must implement compute_labels().
    """

    @abstractmethod
    def compute_labels(self, data: pd.DataFrame, events: pd.Series) -> pd.DataFrame:
        """
        Compute labels for each event position.

        Args:
            data: OHLCV DataFrame with precomputed indicators.
            events: pd.Series with event directions. Values in {-1, 0, 1}
                    or boolean (True = long-only event).

        Returns:
            pd.DataFrame indexed by event position with columns:
              label          — int or float label value
              barrier_hit    — str: 'upper' | 'lower' | 'timeout' | 'fixed_horizon'
              exit_idx       — int: bar index of exit
              n_bars_held    — int: bars in trade
              entry_price    — float
              exit_price     — float
              actual_return  — float: trade-side log return
              truncated      — bool: event too close to data end
        """
        ...
