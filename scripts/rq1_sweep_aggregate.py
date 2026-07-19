"""Aggregate the RQ1 robustness sweep into a per-base verdict.

For each base: the agent's OOS skill across N_SEEDS seeds vs. the placebo (luck)
null across N_PLACEBO surrogate draws -> skill_net + small-sample-t CI +
placebo-exceedance. Reads the cached folds the array tasks produced (loads npz,
no training), so it runs in seconds. Run locally after the array job finishes.
Config is imported from scripts.rq1_sweep_task so the two cannot drift.
"""
import json

import numpy as np
from scipy import stats

from scripts.rq1_sweep_task import (
    BASES, N_SEEDS, N_PLACEBO, WINDOW, COST_BPS, TOTAL_TIMESTEPS,
    INITIAL_TRAIN, TEST_BLOCK, PLACEBO_PHASE_BASE, PLACEBO_TRAIN_BASE,
    RUN_DIR, SAFE_TICKER_PREFERENCE, ACTION_MODE, MAX_TILT,
)


def main():
    from src.data import load_etf_panel
    from src.walk_forward import walk_forward_gate
    from src.placebo import phase_randomize

    panel = load_etf_panel()
    returns = panel["returns"]
    tickers = panel["tickers"]
    safe_ticker = next((t for t in SAFE_TICKER_PREFERENCE if t in tickers), None)
    if safe_ticker is None:
        raise ValueError(f"no safe sleeve {SAFE_TICKER_PREFERENCE} in panel tickers {tickers}")
    safe_index = tickers.index(safe_ticker)
    base_cfg = {"window": WINDOW, "cost_bps": COST_BPS, "safe_asset_index": safe_index,
                "total_timesteps": TOTAL_TIMESTEPS, "initial_train": INITIAL_TRAIN,
                "test_block": TEST_BLOCK, "action_mode": ACTION_MODE, "max_tilt": MAX_TILT}

    summary = {}
    for base in BASES:
        agent_skills = []
        for seed in range(N_SEEDS):
            run_dir = RUN_DIR / base / f"agent_seed{seed}"
            result = walk_forward_gate(returns, {**base_cfg, "base_name": base, "seed": seed},
                                       run_dir=run_dir)
            agent_skills.append(result["mean_skill"])
        placebo_skills = []
        for draw in range(N_PLACEBO):
            surrogate = phase_randomize(returns, seed=PLACEBO_PHASE_BASE + draw)
            run_dir = RUN_DIR / base / f"placebo_{draw:02d}"
            result = walk_forward_gate(surrogate,
                                       {**base_cfg, "base_name": base, "seed": PLACEBO_TRAIN_BASE + draw},
                                       run_dir=run_dir)
            placebo_skills.append(result["mean_skill"])

        agent = np.asarray(agent_skills, dtype=float)
        placebo = np.asarray(placebo_skills, dtype=float)
        skill_net = float(agent.mean() - placebo.mean())
        combined_se = float(np.sqrt(agent.std(ddof=1) ** 2 / len(agent)
                                    + placebo.std(ddof=1) ** 2 / len(placebo)))
        dof = max(1, min(len(agent), len(placebo)) - 1)
        t_crit = float(stats.t.ppf(0.975, dof))
        skill_net_ci = [skill_net - t_crit * combined_se, skill_net + t_crit * combined_se]
        # fraction of placebo (luck) runs whose skill meets/exceeds the mean agent skill
        placebo_exceedance = float(np.mean(placebo >= agent.mean()))

        summary[base] = {
            "agent_skills": [float(x) for x in agent],
            "agent_mean": float(agent.mean()),
            "agent_std": float(agent.std(ddof=1)),
            "placebo_skills": [float(x) for x in placebo],
            "placebo_mean": float(placebo.mean()),
            "placebo_std": float(placebo.std(ddof=1)),
            "skill_net": skill_net,
            "skill_net_ci": skill_net_ci,
            "placebo_exceedance": placebo_exceedance,
        }
        print(f"{base:>13}: agent={agent.mean():+.3e}±{agent.std(ddof=1):.1e}  "
              f"placebo={placebo.mean():+.3e}  skill_net={skill_net:+.3e}  "
              f"CI=[{skill_net_ci[0]:+.2e},{skill_net_ci[1]:+.2e}]  exceed={placebo_exceedance:.2f}",
              flush=True)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {RUN_DIR / 'summary.json'}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = list(summary)
    nets = [summary[b]["skill_net"] for b in labels]
    lower = [summary[b]["skill_net"] - summary[b]["skill_net_ci"][0] for b in labels]
    upper = [summary[b]["skill_net_ci"][1] - summary[b]["skill_net"] for b in labels]
    plt.figure()
    plt.axhline(0.0, color="grey", linewidth=0.8)
    plt.bar(labels, nets, yerr=[lower, upper], capsize=6)
    plt.ylabel("skill net of placebo null")
    plt.title("RQ1 robustness: learned skill above structure, per base")
    plt.savefig(RUN_DIR / "skill_net_by_base.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"wrote {RUN_DIR / 'skill_net_by_base.png'}", flush=True)


if __name__ == "__main__":
    main()
