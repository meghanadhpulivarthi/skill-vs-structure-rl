import numpy as np
from src import metrics


def test_expected_shortfall_picks_worst_tail():
    r = np.array([-0.10, -0.05, 0.0, 0.01, 0.02] * 20)  # 100 obs
    es = metrics.expected_shortfall(r, alpha=0.99)  # worst 1% => the single -0.10
    assert es <= -0.09


def test_max_drawdown_is_negative_for_a_dip():
    r = np.array([0.1, -0.5, 0.1])
    assert metrics.max_drawdown(r) < -0.4


def test_sharpe_zero_mean_is_zero():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, size=10_000)
    assert abs(metrics.sharpe(r)) < 0.15


def test_turnover_zero_for_constant_weights():
    w = np.tile(np.array([0.5, 0.5]), (10, 1))
    assert metrics.turnover(w) == 0.0


def test_turnover_full_switch():
    w = np.array([[1.0, 0.0], [0.0, 1.0]])
    # 0.5 * (|0-1| + |1-0|) = 1.0 on the single transition
    assert abs(metrics.turnover(w) - 1.0) < 1e-9
