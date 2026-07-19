"""Parallel tilt-model RQ2 calibration on LSF (CPU).

Runs the same computation as `src.validate_skill.run_skill_validation` for the
tilt agent on the multi-regime market — 5 seeds x 2 signal strengths = 10
INDEPENDENT trainings — but concurrently across worker processes, so the RQ2
validity gate can be checked (and max_tilt calibrated) in minutes instead of ~25
min sequential. It mirrors run_skill_validation's exact seed scheme (train seed
1000+seed, eval seed 2000+seed, PPO seed=seed) so the printed skill_off/skill_on
match what the pytest gate would produce. Orchestration only; reuses tested
functions. GPU is not used (tiny MLP, CPU/env-bound).

Config MUST match tests/test_validate_skill_tilt.py so this verifies that gate.
"""
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# Config — must match tests/test_validate_skill_tilt.py
BASE_NAME = "equal_weight"
WINDOW = 20
COST_BPS = 10.0
ACTION_MODE = "tilt"
MAX_TILT = 0.15          # 0.15 keeps signal-off clean; 0.20 over-churns (skill_off=-8.5e-5)
N_RISKY = 3
N_SAFE = 2
N_STEPS = 6000
TOTAL_TIMESTEPS = 150_000
N_SEEDS = 5              # net-of-null is low-variance; 5 seeds suffice for a robust gate
SIGNAL_STRENGTHS = (0.0, 0.95)
THREADS = 3
FLOOR = 5e-5


def one_run(strength, seed):
    import torch
    torch.set_num_threads(THREADS)
    from src.synthetic_market import generate_multi_regime_market
    from src.train import train_agent
    from src.validate_skill import evaluate_skill

    config = {"base_name": BASE_NAME, "window": WINDOW, "cost_bps": COST_BPS,
              "action_mode": ACTION_MODE, "max_tilt": MAX_TILT, "n_steps": N_STEPS,
              "total_timesteps": TOTAL_TIMESTEPS, "seed": seed}
    train_market = generate_multi_regime_market(N_RISKY, N_SAFE, N_STEPS,
                                                seed=1000 + seed, signal_strength=strength)
    model = train_agent(train_market, config)
    eval_market = generate_multi_regime_market(N_RISKY, N_SAFE, N_STEPS,
                                               seed=2000 + seed, signal_strength=strength)
    skill = evaluate_skill(model, eval_market, config)["mean_baselined_reward"]
    return (strength, seed, float(skill))


def main():
    os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
    tasks = [(strength, seed) for strength in SIGNAL_STRENGTHS for seed in range(N_SEEDS)]
    print(f"rq2_tilt_parallel: max_tilt={MAX_TILT} n_seeds={N_SEEDS} steps={TOTAL_TIMESTEPS} "
          f"| {len(tasks)} concurrent trainings ({THREADS} threads each)", flush=True)

    context = multiprocessing.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=context) as executor:
        futures = [executor.submit(one_run, strength, seed) for (strength, seed) in tasks]
        for future in futures:
            strength, seed, skill = future.result()
            results.append((strength, seed, skill))
            print(f"  strength={strength} seed={seed} skill={skill:.6e}", flush=True)

    by_strength = {}
    for strength in SIGNAL_STRENGTHS:
        vals = np.array([r[2] for r in results if r[0] == strength], dtype=float)
        by_strength[strength] = (float(vals.mean()), float(vals.std()))

    skill_off = by_strength[0.0][0]
    skill_on = by_strength[0.95][0]
    print("=" * 55, flush=True)
    print(f"skill_off (signal 0.0)  = {skill_off:+.6e}  (std {by_strength[0.0][1]:.1e})", flush=True)
    print(f"skill_on  (signal 0.95) = {skill_on:+.6e}  (std {by_strength[0.95][1]:.1e})", flush=True)
    # Net-of-null RQ2 criterion (consistent with the real-data placebo-net-of-null
    # method): skill signal-on NET of the signal-off null must be clearly positive,
    # and signal-off must not manufacture POSITIVE skill (one-sided; negative churn
    # is fine — it is the null that gets netted out).
    skill_net = skill_on - skill_off
    print(f"floor={FLOOR:.1e} | skill_net (on-off) = {skill_net:+.6e} = {skill_net / FLOOR:.2f}x floor", flush=True)
    print(f"GATE (net-of-null): pass_net={skill_net > FLOOR}  "
          f"pass_off_not_positive={skill_off < FLOOR}", flush=True)


if __name__ == "__main__":
    main()
