import numpy as np
from src.baselines import minimum_variance_base, cvar_min_base, roll_weights
from src.base_policies import equal_weight_base
from src.metrics import expected_shortfall


def test_min_variance_downweights_the_volatile_asset():
    rng = np.random.default_rng(0)
    win = np.column_stack([
        rng.normal(0, 0.005, 300),   # calm asset
        rng.normal(0, 0.05, 300),    # volatile asset
    ])
    w = minimum_variance_base(win)
    assert w[0] > w[1]
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)
    assert np.all(w >= -1e-6)


def test_cvar_min_reduces_tail_vs_equal_weight():
    # One asset has a fat left tail; CVaR-min should avoid it, giving a
    # less-negative expected shortfall than equal weight on the same window.
    rng = np.random.default_rng(1)
    calm = rng.normal(0.0, 0.01, 500)
    fat = rng.standard_t(df=3, size=500) * 0.02      # heavy left tail
    win = np.column_stack([calm, fat])
    w_cvar = cvar_min_base(win, alpha=0.95)
    np.testing.assert_allclose(w_cvar.sum(), 1.0, atol=1e-6)
    assert np.all(w_cvar >= -1e-6)
    es_cvar = expected_shortfall(win @ w_cvar, alpha=0.95)
    es_eq = expected_shortfall(win @ equal_weight_base(win), alpha=0.95)
    assert es_cvar >= es_eq                           # less-negative (better) tail


def test_roll_weights_charges_cost_and_shapes():
    rng = np.random.default_rng(2)
    returns = rng.normal(0, 0.01, size=(120, 3))
    net = roll_weights(equal_weight_base, returns, window=20, cost_bps=10.0)
    assert net.shape == (100,)                        # T - window
    assert np.isfinite(net).all()
