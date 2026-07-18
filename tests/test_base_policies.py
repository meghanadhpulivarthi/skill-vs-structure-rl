import numpy as np
from src.base_policies import equal_weight_base, vol_scaled_base, BASE_POLICIES


def test_equal_weight_is_uniform_and_on_simplex():
    win = np.zeros((20, 4))
    w = equal_weight_base(win)
    np.testing.assert_allclose(w, np.full(4, 0.25))
    np.testing.assert_allclose(w.sum(), 1.0)


def test_vol_scaled_downweights_high_vol_asset():
    rng = np.random.default_rng(0)
    win = np.column_stack([
        rng.normal(0, 0.005, 60),   # low-vol asset
        rng.normal(0, 0.05, 60),    # high-vol asset
    ])
    w = vol_scaled_base(win)
    assert w[0] > w[1]                      # low-vol gets more weight
    np.testing.assert_allclose(w.sum(), 1.0)
    assert np.all(w >= 0)


def test_registry_contains_both():
    assert set(BASE_POLICIES.keys()) == {"equal_weight", "vol_scaled"}
