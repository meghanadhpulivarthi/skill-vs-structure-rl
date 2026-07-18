"""RQ2: validate the skill measure against synthetic ground truth.

The mean structure-baselined reward on held-out markets IS the operational skill
measure. It must vanish when signal_strength=0 (no timeable structure => skill
impossible) and be positive when signal_strength>0. Passing this calibrates the
measure so real-data results (Plan 2) are interpretable.
"""
import json
import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.synthetic_market import generate_risky_safe_market
from src.train import build_env, train_agent
from src.metrics import expected_shortfall, turnover


def evaluate_skill(model, market: dict, config: dict) -> dict:
    env = build_env(market, config)
    obs, _ = env.reset(seed=0)
    baselined_rewards, port_returns, base_returns, gates, weights_path = [], [], [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        baselined_rewards.append(reward)
        port_returns.append(info["port_return"])
        base_returns.append(info["base_return"])
        gates.append(info["gate"])
        weights_path.append(info["weights"])
        done = term or trunc

    es_agent = expected_shortfall(np.array(port_returns))
    es_base = expected_shortfall(np.array(base_returns))
    return {
        "mean_baselined_reward": float(np.mean(baselined_rewards)),
        "mean_gate": float(np.mean(gates)),
        "residual_turnover": float(turnover(np.array(weights_path))),
        "es_agent": es_agent,
        "es_base": es_base,
        "es_reduction": float(es_agent - es_base),  # >0 means agent has the better (less negative) tail
    }


def run_skill_validation(config: dict, signal_strengths=(0.0, 0.95), n_seeds: int = 5) -> dict:
    now = datetime.datetime.now()
    print("=" * 60)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Config      : {config} | strengths={signal_strengths} | n_seeds={n_seeds}")
    print("=" * 60)

    by_strength = {}
    for strength in signal_strengths:
        seed_rewards = []
        for seed in tqdm(range(n_seeds), desc=f"signal={strength}"):
            train_market = generate_risky_safe_market(config["n_steps"], seed=1000 + seed,
                                                       signal_strength=strength)
            model = train_agent(train_market, {**config, "seed": seed})
            eval_market = generate_risky_safe_market(config["n_steps"], seed=2000 + seed,
                                                     signal_strength=strength)
            seed_rewards.append(evaluate_skill(model, eval_market, config)["mean_baselined_reward"])
        by_strength[str(strength)] = {
            "mean_baselined_reward": float(np.mean(seed_rewards)),
            "std_baselined_reward": float(np.std(seed_rewards)),
            "per_seed": [float(x) for x in seed_rewards],
        }
        print(f"signal={strength}: mean skill = {by_strength[str(strength)]['mean_baselined_reward']:.6e}")

    out_dir = Path(__file__).resolve().parent.parent / "outputs" / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_skill-validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump({"config": config, "signal_strengths": list(signal_strengths), "n_seeds": n_seeds}, f, indent=2)
    result = {"by_strength": by_strength}
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    labels = [str(s) for s in signal_strengths]
    means = [by_strength[s_label]["mean_baselined_reward"] for s_label in labels]
    stds = [by_strength[s_label]["std_baselined_reward"] for s_label in labels]
    plt.figure()
    plt.bar(labels, means, yerr=stds, capsize=4)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("signal_strength (timeable structure)")
    plt.ylabel("mean structure-baselined reward (skill)")
    plt.title("RQ2: skill vanishes without timeable structure")
    plt.savefig(out_dir / "skill_vs_signal.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved run to: {out_dir}")
    return result


if __name__ == "__main__":
    _config = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
               "safe_asset_index": 1, "total_timesteps": 150_000, "n_steps": 6000}
    run_skill_validation(_config, signal_strengths=(0.0, 0.5, 0.95), n_seeds=5)
