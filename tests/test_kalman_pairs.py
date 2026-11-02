import pytest
import numpy as np
import pandas as pd
from research.models.kalman_filter import KalmanFilterPairs, compute_half_life
from research.triggers.kalman_pairs import KalmanPairsTrigger


def test_kalman_filter_recovers_synthetic_beta():
    np.random.seed(42)
    n = 2000
    # Synthetic asset X random walk
    px = np.cumsum(np.random.randn(n)) + 100.0
    # Dynamic beta drifting from 1.5 to 2.0
    true_beta = np.linspace(1.5, 2.0, n)
    # Asset Y = true_beta * X + stationary noise
    noise = np.random.randn(n) * 0.5
    py = true_beta * px + noise

    kf = KalmanFilterPairs(delta=1e-4, R=1e-3)
    alphas, betas, residuals, zscores = kf.fit_spread(px, py)

    # In the middle/end, recovered beta should be close to true beta
    assert abs(betas[-1] - true_beta[-1]) < 0.15
    assert len(residuals) == n


def test_compute_half_life():
    np.random.seed(42)
    # Generate stationary mean-reverting OU process: s_t = 0.9 * s_{t-1} + e_t
    # Half life should be -log(2) / (0.9 - 1.0) = -log(2) / -0.1 ~= 6.93 steps
    s = [0.0]
    for _ in range(2000):
        s.append(0.9 * s[-1] + np.random.randn())
    s = np.array(s)
    hl = compute_half_life(s)
    assert 5.0 < hl < 9.0


def test_kalman_pairs_trigger_signals():
    np.random.seed(42)
    n = 500
    px = pd.Series(np.cumsum(np.random.randn(n)) + 100.0)
    py = pd.Series(px.values * 1.5 + np.random.randn(n) * 2.0)

    trig = KalmanPairsTrigger(entry_z=1.5, exit_z=0.2, stop_z=4.0, zscore_window=60, cooldown=10)
    res = trig.generate_pair_signals(py, px)

    assert "signal_y" in res.columns
    assert "signal_x" in res.columns
    assert "beta" in res.columns
    assert "zscore" in res.columns
    # Check that short Y corresponds to long X
    shorts_y = (res["signal_y"] == -1)
    if shorts_y.any():
        assert (res.loc[shorts_y, "signal_x"] == 1).all()
