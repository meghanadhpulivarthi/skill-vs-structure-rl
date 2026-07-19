from src.validate_skill import run_skill_validation


def test_tilt_skill_vanishes_without_signal_and_appears_with_signal():
    # THE RQ2 GATE FOR THE TILT MODEL. The skill measure must be ~0 with no timeable
    # signal (the tilt agent must learn tilt~0) and clearly positive when the signal
    # exists (it times a tilt toward the safe block). Same validity claim as the gate,
    # now for the expressive action.
    # Judged NET-OF-NULL, consistent with the real-data placebo-net-of-null method:
    # the expressive tilt agent cannot reach a clean do-nothing floor — signal-off it
    # over-churns and LOSES (~-6e-5), signal-on it adds skill (~+1.4e-4). So we require
    # (1) skill NET of the signal-off null is clearly positive (skill appears only when
    # timeable structure exists), and (2) signal-off does not manufacture POSITIVE
    # skill (one-sided; the negative churn is the null that gets netted out).
    # Validated on LSF (max_tilt=0.15, 5 seeds, 150k): skill_off=-6.4e-5, skill_on=+1.4e-4,
    # skill_net=+2.0e-4 (4.09x floor). See context/decisions.md.
    config = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
              "action_mode": "tilt", "max_tilt": 0.15, "market": "multi_regime",
              "n_risky": 3, "n_safe": 2, "total_timesteps": 150_000, "n_steps": 6000}
    result = run_skill_validation(config, signal_strengths=(0.0, 0.95), n_seeds=5)

    skill_off = result["by_strength"]["0.0"]["mean_baselined_reward"]
    skill_on = result["by_strength"]["0.95"]["mean_baselined_reward"]
    skill_net = skill_on - skill_off

    # Do NOT weaken these — they encode the validity claim (Rule 9). If skill_net
    # falls below the floor, recalibrate max_tilt (spec §3/§6), not the threshold.
    assert skill_net > 5e-5          # net of the matched signal-off null: skill appears only with structure
    assert skill_off < 5e-5          # signal-off must not manufacture POSITIVE skill (one-sided)
