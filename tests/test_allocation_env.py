import numpy as np
from src.synthetic_market import generate_market
from src.allocation_env import AllocationEnv


def _make_env(base_name="vol_scaled"):
    market = generate_market(n_assets=4, n_steps=300, seed=5, signal_strength=0.8)
    return AllocationEnv(market, base_name=base_name, window=20, cost_bps=10.0)


def test_reset_returns_obs_of_declared_shape():
    env = _make_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape


def test_zero_action_reproduces_base_and_gives_zero_reward():
    # residual = 0 => weights == base => structure-baselined reward == 0.
    env = _make_env()
    env.reset(seed=0)
    action = np.zeros(env.action_space.shape[0])
    _, reward, _, _, info = env.step(action)
    np.testing.assert_allclose(info["weights"], info["base_weights"], atol=1e-8)
    assert abs(reward) < 1e-8


def test_weights_stay_on_simplex_for_arbitrary_action():
    env = _make_env()
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    for _ in range(50):
        _, _, term, trunc, info = env.step(rng.normal(size=env.action_space.shape[0]))
        w = info["weights"]
        assert np.all(w >= -1e-9)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-8)
        if term or trunc:
            break


def test_episode_terminates_at_end_of_series():
    env = _make_env()
    env.reset(seed=0)
    steps = 0
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(np.zeros(env.action_space.shape[0]))
        done = term or trunc
        steps += 1
    assert steps == 300 - 20  # n_steps - window
