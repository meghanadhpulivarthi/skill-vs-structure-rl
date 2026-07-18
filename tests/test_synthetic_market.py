import numpy as np
from src.synthetic_market import generate_market


def test_shapes_and_keys():
    m = generate_market(n_assets=5, n_steps=1000, seed=1, signal_strength=0.8)
    assert m["returns"].shape == (1000, 5)
    assert m["signal"].shape == (1000,)
    assert m["regime"].shape == (1000,)
    assert set(np.unique(m["regime"])).issubset({0, 1})


def test_crisis_has_higher_volatility_than_calm():
    m = generate_market(n_assets=5, n_steps=20_000, seed=2, signal_strength=0.8)
    calm_vol = m["returns"][m["regime"] == 0].std()
    crisis_vol = m["returns"][m["regime"] == 1].std()
    assert crisis_vol > 1.5 * calm_vol


def test_signal_predicts_next_regime_only_when_strength_positive():
    # With signal, current signal should correlate with NEXT-step crisis.
    m_on = generate_market(n_assets=3, n_steps=20_000, seed=3, signal_strength=0.9)
    next_crisis = m_on["regime"][1:]
    corr_on = np.corrcoef(m_on["signal"][:-1], next_crisis)[0, 1]
    assert corr_on > 0.2

    m_off = generate_market(n_assets=3, n_steps=20_000, seed=3, signal_strength=0.0)
    corr_off = abs(np.corrcoef(m_off["signal"][:-1], m_off["regime"][1:])[0, 1])
    assert corr_off < 0.05


def test_determinism():
    a = generate_market(n_assets=4, n_steps=500, seed=7, signal_strength=0.5)
    b = generate_market(n_assets=4, n_steps=500, seed=7, signal_strength=0.5)
    np.testing.assert_array_equal(a["returns"], b["returns"])
