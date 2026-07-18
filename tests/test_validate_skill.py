from src.validate_skill import run_skill_validation


def test_skill_vanishes_without_signal_and_appears_with_signal():
    # THE RQ2 GROUND-TRUTH TEST. The skill measure (mean structure-baselined
    # reward on held-out data) must be ~0 with no timeable signal, and clearly
    # positive when the signal exists. This is the paper's core validity claim.
    config = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
              "safe_asset_index": 1, "total_timesteps": 100_000, "n_steps": 6000}
    result = run_skill_validation(config, signal_strengths=(0.0, 0.95), n_seeds=2)

    skill_off = result["by_strength"]["0.0"]["mean_baselined_reward"]
    skill_on = result["by_strength"]["0.95"]["mean_baselined_reward"]

    assert skill_on > 0.0                 # timeable structure => positive skill
    assert abs(skill_off) < 5e-5          # no structure => skill ~ 0 (noise floor)
    assert skill_on > 3 * abs(skill_off)  # signal-driven skill dominates the floor
