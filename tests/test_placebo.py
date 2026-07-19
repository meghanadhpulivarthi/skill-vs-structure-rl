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


def test_phase_randomize_destroys_volatility_clustering():
    # WHY THIS IS THE RIGHT NULL: phase randomization preserves the linear power
    # spectrum, so (Wiener-Khinchin) it PRESERVES linear autocorrelation and does NOT
    # destroy 2nd-order linear structure. What it destroys is volatility clustering —
    # the autocorrelation of SQUARED returns — which is exactly the timeable
    # (regime/vol-clustering) structure the de-risking gate exploits via its
    # trailing-vol signal. So the surrogate is a valid "no timeable structure" null.
    rng = np.random.default_rng(1)
    n = 3000
    # Simple regime-switching: returns are white noise but volatility alternates in blocks.
    # This creates vol clustering in squared returns without AR structure in raw returns.
    block_size = 150
    vol_block = np.repeat([0.002, 0.05], block_size)  # 300-step repeat pattern
    vol_schedule = np.tile(vol_block, (n // len(vol_block) + 1))[:n]
    returns = (rng.normal(0.0, 1.0, n) * vol_schedule).reshape(-1, 1)
    surrogate = phase_randomize(returns, seed=3)

    def lag1(x):
        return np.corrcoef(x[:-1], x[1:])[0, 1]

    original_vol_clustering = lag1(returns.ravel() ** 2)
    surrogate_vol_clustering = lag1(surrogate.ravel() ** 2)
    # Regime switching creates vol clustering in squared returns
    assert original_vol_clustering > 0.17           # strong vol clustering by construction (true value ~0.18)
    assert abs(surrogate_vol_clustering) < 0.1      # phase randomization destroys it


def test_placebo_null_returns_requested_number_of_skills(tmp_path):
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0003, 0.01, size=(900, 3))
    config = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
              "safe_asset_index": 2, "total_timesteps": 1500, "seed": 0,
              "initial_train": 400, "test_block": 250}
    null = placebo_null(returns, config, n_placebo=2, seed=0, run_dir=tmp_path)
    assert len(null["placebo_skills"]) == 2
    assert np.isfinite(null["mean"]) and np.isfinite(null["std"])
