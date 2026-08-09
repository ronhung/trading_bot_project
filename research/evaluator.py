"""
ModelEvaluator — time-series aware model training and evaluation.

Extracted from the deleted research/ml_analysis.py. Provides:
  - Purged time-series cross-validation
  - Spearman Rank IC
  - Decile/quintile mean return analysis
  - AUC-ROC for binary classification
  - XGBoost model training + save/load

Usage:
    evaluator = ModelEvaluator()
    evaluator.train_model(X_train, y_train)
    ic = evaluator.evaluate_rank_ic(y_true, y_pred)
    decile = evaluator.evaluate_decile_spread(y_true, y_pred)
"""

import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


class ModelEvaluator:
    """
    Time-series aware model evaluation for quant finance.

    Defaults match the Phase 3 XGBoost pipeline:
      - Regression target: y_norm (forward return / daily ATR)
      - Spearman IC threshold: 0.03
      - 5-quantile decile staircase

    Also supports binary classification with AUC-ROC when labels
    are {-1, 0, 1} from triple-barrier labeling.
    """

    def __init__(
        self,
        model_params: Optional[Dict[str, Any]] = None,
        target: str = "y_norm",
        feature_exclude: Optional[Set[str]] = None,
        ic_threshold: float = 0.03,
        n_quantiles: int = 5,
    ):
        """
        Args:
            model_params: XGBoost parameters dict. Defaults to Phase 3 params.
            target: Name of the target column in the dataset.
            feature_exclude: Column names to exclude from features.
            ic_threshold: Minimum |IC| to pass the rank IC check.
            n_quantiles: Number of quantile groups for decile analysis.
        """
        self.model_params = model_params or dict(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            verbosity=0,
        )
        self.target = target
        self.feature_exclude = feature_exclude or {
            "label", "barrier_hit", "exit_idx", "n_bars_held",
            "entry_price", "exit_price", "actual_return", "return_atr",
            "raw_return", "y_norm", "truncated",
            "event_time", "side",
        }
        self.ic_threshold = ic_threshold
        self.n_quantiles = n_quantiles
        self._model: Any = None
        self._feature_names: List[str] = []

    # -- Feature preparation ------------------------------------------------

    def prepare_features(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract X (features) and y (target) from a dataset DataFrame.

        Returns:
            (X, y, feature_names) — X and y as float32 numpy arrays.
        """
        feature_cols = [
            c for c in df.columns if c not in self.feature_exclude
        ]
        self._feature_names = feature_cols

        X = np.nan_to_num(
            df[feature_cols].values.astype(np.float32), nan=0.0
        )
        y = df[self.target].values.astype(np.float32)

        return X, y, feature_cols

    # -- Time-series split --------------------------------------------------

    def time_series_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        n_splits: int = 5,
        gap: int = 0,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Purged time-series cross-validation.

        Splits data chronologically into n_splits train/val folds.
        Each fold uses all data up to a cutoff for training and the
        subsequent block for validation.

        Args:
            X: Feature matrix.
            y: Target vector.
            dates: Optional datetime index (unused currently, reserved for
                   date-aware purging).
            n_splits: Number of folds.
            gap: Bars to skip between train and val (purging).

        Returns:
            List of (X_train, X_val, y_train, y_val) tuples.
        """
        n = len(X)
        if n < n_splits * 2:
            raise ValueError(
                f"Not enough data ({n}) for {n_splits} splits"
            )

        fold_size = n // (n_splits + 1)
        folds = []
        for i in range(n_splits):
            val_start = (i + 1) * fold_size
            val_end = val_start + fold_size
            train_end = val_start - gap

            folds.append((
                X[:train_end], X[val_start:val_end],
                y[:train_end], y[val_start:val_end],
            ))
        return folds

    # -- Model training -----------------------------------------------------

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_type: str = "xgboost",
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Train and return a model.

        Args:
            X_train: Feature matrix.
            y_train: Target vector.
            model_type: "xgboost" (default) — others can be added.
            params: Override default model params.

        Returns:
            Trained model with .predict(X) method.
        """
        if model_type == "xgboost":
            import xgboost as xgb
            p = {**self.model_params, **(params or {})}
            self._model = xgb.XGBRegressor(**p)
            self._model.fit(X_train, y_train, verbose=False)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        return self._model

    def train_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Train XGBoost classifier for binary labels (AUC-ROC path)."""
        import xgboost as xgb
        p = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "objective": "binary:logistic",
            "random_state": 42,
            "verbosity": 0,
            **(params or {}),
        }
        self._model = xgb.XGBClassifier(**p)
        self._model.fit(X_train, y_train, verbose=False)
        return self._model

    # -- Evaluation metrics -------------------------------------------------

    def evaluate_rank_ic(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Spearman Rank IC between predictions and actuals.

        Returns:
            {'ic': float, 'p_value': float, 'pass': bool}
        """
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        if valid.sum() < 10:
            return {"ic": 0.0, "p_value": 1.0, "pass": False}

        ic, pval = spearmanr(y_pred[valid], y_true[valid])
        return {
            "ic": float(ic),
            "p_value": float(pval),
            "pass": abs(ic) > self.ic_threshold and float(pval) < 0.05,
        }

    def evaluate_decile_spread(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_quantiles: int | None = None,
    ) -> Dict[str, Any]:
        """
        Decile mean return analysis.

        Sorts predictions into quantiles, computes mean y_true per
        quantile, and checks for monotonic staircase.

        Returns:
            {'quantile_means': [...], 'spread': float, 'monotonic': bool}
        """
        nq = n_quantiles or self.n_quantiles
        valid = ~(np.isnan(y_true) | np.isnan(y_pred))
        yt = y_true[valid]
        yp = y_pred[valid]

        if len(yt) < nq * 5:
            return {
                "quantile_means": [],
                "spread": 0.0,
                "monotonic": False,
            }

        try:
            quantiles = pd.qcut(yp, q=nq, labels=False, duplicates="drop")
        except ValueError:
            return {
                "quantile_means": [],
                "spread": 0.0,
                "monotonic": False,
            }

        means = []
        for q in range(nq):
            mask = quantiles == q
            if mask.sum() > 0:
                means.append(float(np.mean(yt[mask])))
            else:
                means.append(0.0)

        spread = means[-1] - means[0] if len(means) >= 2 else 0.0
        monotonic = all(
            means[i] <= means[i + 1] for i in range(len(means) - 1)
        )

        return {
            "quantile_means": means,
            "spread": spread,
            "monotonic": monotonic,
        }

    def evaluate_auc_roc(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> Dict[str, float]:
        """
        AUC-ROC for binary classification.

        Args:
            y_true: Binary labels (0/1).
            y_prob: Predicted probabilities.

        Returns:
            {'auc': float}
        """
        from sklearn.metrics import roc_auc_score

        valid = ~(np.isnan(y_true) | np.isnan(y_prob))
        if valid.sum() < 5 or len(np.unique(y_true[valid])) < 2:
            return {"auc": 0.5}

        auc = float(roc_auc_score(y_true[valid], y_prob[valid]))
        return {"auc": auc, "pass": auc > 0.55}

    # -- Save / Load --------------------------------------------------------

    def save(
        self,
        out_dir: str,
        prefix: str = "xgb",
        feature_names: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Save model and feature list to disk.

        Args:
            out_dir: Output directory.
            prefix: Filename prefix (e.g., "turtle_vol_filter").
            feature_names: Feature column names. Uses stored names if None.

        Returns:
            List of saved file paths.
        """
        os.makedirs(out_dir, exist_ok=True)
        saved = []

        if self._model is not None:
            model_path = os.path.join(out_dir, f"{prefix}_model.json")
            self._model.save_model(model_path)
            saved.append(model_path)

        feats = feature_names or self._feature_names
        if feats:
            feat_path = os.path.join(out_dir, f"{prefix}_features.json")
            with open(feat_path, "w") as f:
                json.dump(feats, f)
            saved.append(feat_path)

        return saved

    @classmethod
    def load(
        cls,
        model_path: str,
        feature_list_path: str,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> "ModelEvaluator":
        """
        Load a trained model and feature list from disk.

        Returns:
            ModelEvaluator with loaded model ready for predict().
        """
        import xgboost as xgb

        evaluator = cls(model_params=model_params)
        evaluator._model = xgb.XGBRegressor()
        evaluator._model.load_model(model_path)

        with open(feature_list_path, "r") as f:
            evaluator._feature_names = json.load(f)

        return evaluator

    @property
    def model(self) -> Any:
        """Access the trained model."""
        return self._model

    @property
    def feature_names(self) -> List[str]:
        """Feature column names in model input order."""
        return self._feature_names
