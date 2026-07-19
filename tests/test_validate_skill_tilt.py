from src.validate_skill import run_skill_validation


def test_tilt_skill_vanishes_without_signal_and_appears_with_signal():
    # THE RQ2 GATE FOR THE TILT MODEL. The skill measure must be ~0 with no timeable
    # signal (the tilt agent must learn tilt~0) and clearly positive when the signal
    # exists (it times a tilt toward the safe block). Same validity claim as the gate,
    # now for the expressive action.
    config = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
              "action_mode": "tilt", "max_tilt": 0.15, "market": "multi_regime",
              "n_risky": 3, "n_safe": 2, "total_timesteps": 120_000, "n_steps": 6000}
    result = run_skill_validation(config, signal_strengths=(0.0, 0.95), n_seeds=3)

    skill_off = result["by_strength"]["0.0"]["mean_baselined_reward"]
    skill_on = result["by_strength"]["0.95"]["mean_baselined_reward"]

    # Floor 5e-5: comfortably above the ~1e-5 noise floor, below observed on-skill.
    # If skill_on falls below this, do NOT weaken the assertion — recalibrate max_tilt
    # (spec §3/§6: RQ2 is where max_tilt is set) and re-run. See context/decisions.md.
    assert skill_on > 5e-5
    assert abs(skill_off) < 5e-5
    assert skill_on > 3 * abs(skill_off)
