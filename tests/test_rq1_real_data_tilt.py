import numpy as np
from src.rq1_real_data import run_rq1


def test_rq1_runs_end_to_end_with_tilt_action(tmp_path):
    # Plumbing check: the RQ1 pipeline runs with the expressive tilt action and
    # produces the same result shape (skill_net + CI + metrics table). On i.i.d.
    # noise there is no timeable structure, so skill_net must not be positively
    # distinguishable from the placebo null (CI straddles ~0) — the tilt action
    # must not manufacture skill from noise any more than the gate did.
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0003, 0.01, size=(900, 3))
    config = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
              "safe_asset_index": 2, "action_mode": "tilt", "max_tilt": 0.15,
              "total_timesteps": 1500, "seed": 0,
              "initial_train": 400, "test_block": 250, "n_placebo": 2}
    result = run_rq1(config, returns=returns, run_dir=tmp_path)

    assert "skill_net" in result and np.isfinite(result["skill_net"])
    assert len(result["skill_net_ci"]) == 2
    assert {"agent", "base", "one_over_n", "min_variance", "cvar_min"}.issubset(result["metrics_table"].keys())
    low, high = result["skill_net_ci"]
    assert low <= 5e-5 and high >= -5e-5                   # no manufactured skill from noise
