# tests/test_allocation_env_tilt.py
import numpy as np
from src.synthetic_market import generate_market
from src.allocation_env import AllocationEnv


def _tilt_env(n_assets=5, n_steps=400, max_tilt=0.15):
    market = generate_market(n_assets=n_assets, n_steps=n_steps, seed=5, signal_strength=0.8)
    return AllocationEnv(market, base_name="equal_weight", window=20, cost_bps=10.0,
                         action_mode="tilt", max_tilt=max_tilt)


def test_tilt_reset_obs_matches_declared_shape():
    env = _tilt_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    # enriched obs: window*n + 4*n + 1
    assert env.observation_space.shape[0] == 20 * 5 + 4 * 5 + 1


def test_tilt_action_space_is_per_asset_and_bounded():
    env = _tilt_env()
    assert env.action_space.shape == (5,)
    assert np.isfinite(env.action_space.low).all() and np.isfinite(env.action_space.high).all()


def test_tilt_zero_action_reproduces_base_and_zero_reward():
    env = _tilt_env()
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros(5))
    np.testing.assert_allclose(info["weights"], info["base_weights"], atol=1e-8)
    assert abs(reward) < 1e-8


def test_tilt_weights_stay_on_simplex_for_arbitrary_actions():
    env = _tilt_env()
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    for _ in range(50):
        _, _, term, trunc, info = env.step(rng.normal(size=5))
        w = info["weights"]
        assert np.all(w >= -1e-9)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-8)
        if term or trunc:
            break


def test_tilt_bounded_by_max_tilt():
    # each executed weight is within max_tilt of the base before projection renormalizes;
    # deviation per asset cannot exceed max_tilt by more than projection slack.
    env = _tilt_env(max_tilt=0.15)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.full(5, 100.0))  # saturates tanh -> +max_tilt each
    dev = np.abs(info["weights"] - info["base_weights"])
    assert dev.max() <= 0.15 + 1e-6


def test_tilt_episode_length_uses_long_window():
    env = _tilt_env(n_steps=400)
    env.reset(seed=0)
    steps = 0
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(np.zeros(5))
        done = term or trunc
        steps += 1
    assert steps == 400 - 40  # n_steps - long_window (2*window)


def test_tilt_info_gate_is_activity_scalar():
    env = _tilt_env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.full(5, 100.0))
    expected = 0.5 * np.abs(info["weights"] - info["base_weights"]).sum()
    assert abs(info["gate"] - expected) < 1e-9
