import numpy as np
from src.rq1_real_data import compute_metrics_bundle, run_rq1


def test_metrics_bundle_keys_and_finite():
    rng = np.random.default_rng(0)
    net = rng.normal(0.0003, 0.01, size=1000)
    bundle = compute_metrics_bundle(net)
    assert set(bundle) == {"es_99", "max_drawdown", "tail_ratio", "skewness", "sharpe", "sortino"}
    assert all(np.isfinite(v) for v in bundle.values())


def test_rq1_no_skill_on_structureless_input(tmp_path):
    # INTENT TEST (real-data analog of RQ2's signal-OFF): on i.i.d. noise there is
    # no timeable structure, so the agent's OOS skill must not exceed the placebo
    # null in a way the method would call real. skill_net should be ~0 (within CI).
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0003, 0.01, size=(900, 3))
    config = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
              "safe_asset_index": 2, "total_timesteps": 1500, "seed": 0,
              "initial_train": 400, "test_block": 250, "n_placebo": 2}
    result = run_rq1(config, returns=returns, run_dir=tmp_path)

    assert "skill_net" in result and np.isfinite(result["skill_net"])
    assert "skill_net_ci" in result and len(result["skill_net_ci"]) == 2
    # metrics table must include the agent, the base, and the literature baselines
    assert {"agent", "base", "one_over_n", "min_variance", "cvar_min"}.issubset(result["metrics_table"].keys())
    # On structureless noise the net skill CI must straddle ~0 (no manufactured skill).
    low, high = result["skill_net_ci"]
    assert low <= 5e-5 and high >= -5e-5
