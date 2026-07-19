import numpy as np
from src.real_market import build_real_market


def test_shapes_and_keys():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, size=(500, 4))
    m = build_real_market(returns, safe_asset_index=3, window=20)
    assert m["returns"].shape == (500, 4)
    assert m["signal"].shape == (500,)
    assert m["safe_asset_index"] == 3
    assert np.all((m["signal"] >= 0.0) & (m["signal"] <= 1.0))


def test_signal_is_causal():
    # THE NO-LOOKAHEAD TEST. Altering returns at/after time t must not change
    # signal[t] — the signal at t may depend only on data strictly before t.
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.01, size=(300, 3))
    base = build_real_market(returns, safe_asset_index=0, window=20)["signal"]

    perturbed = returns.copy()
    perturbed[150:] += 5.0                       # huge change from t=150 onward
    after = build_real_market(perturbed, safe_asset_index=0, window=20)["signal"]

    np.testing.assert_allclose(base[:151], after[:151], atol=1e-12)  # <= t=150 unchanged


def test_signal_rises_in_high_vol_window():
    # Calm then turbulent: signal should be higher in the turbulent tail.
    rng = np.random.default_rng(2)
    calm = rng.normal(0, 0.005, size=(400, 3))
    turbulent = rng.normal(0, 0.05, size=(400, 3))
    returns = np.vstack([calm, turbulent])
    signal = build_real_market(returns, safe_asset_index=0, window=20)["signal"]
    assert signal[600:].mean() > signal[100:380].mean()
