"""Train the high-freq ML filter (classifier) and save model + feature list.

Target: did the trailing trade win (r_multiple > 0)?
Saves research/outputs/trend_breakout_highfreq_model.json + _features.json.
"""

import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd

from research.features import add_indicators, default_feature_set
from research.evaluator import ModelEvaluator
from research.labeling import TrailingExitLabeler
from research.triggers.trend_breakout import TrendFilteredBreakoutTrigger


def main():
    ep, tm = 2160, 4
    df = pd.read_parquet(os.path.join(_PROJECT_ROOT, "data", "historical_data", "BTCUSDT_1m_full.parquet"))
    df["datetime"] = pd.to_datetime(df["datetime"])
    train = df[df["datetime"] < "2024-01-01"].reset_index(drop=True)

    ind = add_indicators(train, entry_period=ep, exit_period=ep // 2, atr_period=ep, ma_period=ep * tm)
    events = TrendFilteredBreakoutTrigger(entry_period=ep, trend_period=ep * tm).generate_signals(ind)
    feat = default_feature_set().compute(ind, events)
    lab = TrailingExitLabeler(trail_period=ep // 2).compute_labels(ind, events)
    X = feat.join(lab, how="left")

    ev = ModelEvaluator(target="r_multiple")
    Xn, yn, fn = ev.prepare_features(X)
    y_bin = (yn > 0).astype(int)
    split = int(len(Xn) * 0.8)

    model = ev.train_classifier(Xn[:split], y_bin[:split])
    prob = model.predict_proba(Xn[split:])[:, 1]
    auc = ev.evaluate_auc_roc(y_bin[split:], prob)
    print(f"train events={len(Xn)}  AUC={auc['auc']:.3f}  pass={auc.get('pass')}")

    # Threshold trade-off: what fraction of entries survive, and their mean return
    for thr in (0.5, 0.55, 0.6, 0.65):
        keep = prob >= thr
        frac = keep.mean()
        mean_r = yn[split:][keep].mean() if keep.sum() else float("nan")
        print(f"  threshold={thr}: keep={frac:.2%}  mean_r_multiple={mean_r:+.3f}")

    # Save (trained on full train set for the live filter)
    full_model = ev.train_classifier(Xn, y_bin)
    ev._model = full_model
    ev._feature_names = fn
    saved = ev.save(os.path.join(_PROJECT_ROOT, "research", "outputs"), prefix="trend_breakout_highfreq")
    print("saved:", saved)


if __name__ == "__main__":
    main()
