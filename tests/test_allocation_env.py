import numpy as np
import pytest
from src.synthetic_market import generate_market
from src.allocation_env import AllocationEnv


def _make_env(base_name="vol_scaled"):
    market = generate_market(n_assets=4, n_steps=300, seed=5, signal_strength=0.8)
    return AllocationEnv(market, base_name=base_name, window=20, cost_bps=10.0)


def test_reset_returns_obs_of_declared_shape():
    env = _make_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape


def test_action_space_is_unit_interval_scalar():
    env = _make_env()
    assert env.action_space.shape == (1,)
    assert env.action_space.low[0] == 0.0 and env.action_space.high[0] == 1.0


def test_zero_gate_reproduces_base_and_gives_zero_reward():
    # g = 0 => weights == base => structure-baselined reward == 0.
    env = _make_env()
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.array([0.0]))
    np.testing.assert_allclose(info["weights"], info["base_weights"], atol=1e-8)
    assert info["gate"] == 0.0
    assert abs(reward) < 1e-8


def test_full_gate_goes_fully_to_safe_asset():
    # g = 1 => weights == safe one-hot (default safe = last asset, index 3).
    env = _make_env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.array([1.0]))
    expected = np.array([0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(info["weights"], expected, atol=1e-8)
    assert info["gate"] == 1.0


def test_weights_stay_on_simplex_for_any_gate():
    env = _make_env()
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    for _ in range(50):
        _, _, term, trunc, info = env.step(rng.random(1))  # g in [0,1)
        w = info["weights"]
        assert np.all(w >= -1e-9)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-8)
        if term or trunc:
            break


def test_gate_is_clipped_to_unit_interval():
    # actions outside [0,1] must be clipped (defensive; SB3 clips too).
    env = _make_env()
    env.reset(seed=0)
    _, _, _, _, info_hi = env.step(np.array([5.0]))
    assert info_hi["gate"] == 1.0


def test_episode_terminates_at_end_of_series():
    env = _make_env()
    env.reset(seed=0)
    steps = 0
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(np.array([0.0]))
        done = term or trunc
        steps += 1
    assert steps == 300 - 20  # n_steps - window


def test_unknown_base_name_raises_valueerror():
    market = generate_market(n_assets=4, n_steps=300, seed=5, signal_strength=0.8)
    with pytest.raises(ValueError):
        AllocationEnv(market, base_name="not_a_real_base", window=20, cost_bps=10.0)


def test_safe_asset_index_out_of_range_raises():
    market = generate_market(n_assets=4, n_steps=300, seed=5, signal_strength=0.8)
    with pytest.raises(ValueError):
        AllocationEnv(market, base_name="vol_scaled", safe_asset_index=9)
