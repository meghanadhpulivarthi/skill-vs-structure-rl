import numpy as np
from src.placebo import phase_randomize, placebo_null


def test_phase_randomize_preserves_variance_and_shape_and_is_deterministic():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, size=(512, 3))
    surrogate = phase_randomize(returns, seed=7)
    assert surrogate.shape == returns.shape
    np.testing.assert_allclose(surrogate.std(axis=0), returns.std(axis=0), rtol=0.05)
    again = phase_randomize(returns, seed=7)
    np.testing.assert_allclose(surrogate, again)               # deterministic in seed


def test_phase_randomize_destroys_autocorrelation():
    # A strongly autocorrelated series should lose its lag-1 autocorrelation.
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, size=2000)
    ar = np.zeros(2000)
    for t in range(1, 2000):
        ar[t] = 0.9 * ar[t - 1] + noise[t]                     # strong AR(1)
    returns = ar.reshape(-1, 1)
    surrogate = phase_randomize(returns, seed=3).ravel()

    def lag1(x):
        return np.corrcoef(x[:-1], x[1:])[0, 1]

    assert lag1(ar) > 0.7
    assert abs(lag1(surrogate)) < 0.3                          # structure destroyed


def test_placebo_null_returns_requested_number_of_skills(tmp_path):
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0003, 0.01, size=(900, 3))
    config = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
              "safe_asset_index": 2, "total_timesteps": 1500, "seed": 0,
              "initial_train": 400, "test_block": 250}
    null = placebo_null(returns, config, n_placebo=2, seed=0, run_dir=tmp_path)
    assert len(null["placebo_skills"]) == 2
    assert np.isfinite(null["mean"]) and np.isfinite(null["std"])
