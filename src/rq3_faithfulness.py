"""RQ3 / MC2 — causal-faithfulness experiment.

Probes a trained gate agent on the synthetic risky+safe market, where the
`signal` feature is the known ground-truth driver. Compares the CAUSAL track
(feature-group ablation) against the ATTRIBUTION track (saliency + KernelSHAP)
to test whether post-hoc attribution identifies the true mechanism (H3).

Run: uv run python -m src.rq3_faithfulness
"""
import json
import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from src.interventions import (feature_groups, rollout_observations, make_gate_fn,
                               causal_effect)
from src.attribution import (make_gate_mean_fn, saliency_importance, shap_importance)

# Config — edit these directly
CONFIG = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
          "safe_asset_index": 1, "action_mode": "gate",
          "total_timesteps": 150_000, "n_steps": 6000, "signal_strength": 0.95}
N_SEEDS = 5
SHAP_BACKGROUND = 40
SHAP_EXPLAIN = 60


def _normalized_group_shares(per_feature: np.ndarray, groups: dict) -> dict:
    """Normalize a per-feature importance vector to sum 1 (share of total), then sum
    the shares within each group. Normalization does NOT remove a group's feature-count
    advantage (it is a monotone rescale; the argmax group is unchanged) — a group that
    collectively drives more output legitimately scores higher. Its purpose is to put all
    four methods on ONE comparable scale (each sums to 1). The cardinality artifact was
    fixed separately, by computing the causal group verdict from the SAME per-feature
    vector the attribution methods use (see run_probe) instead of the old joint-group
    ablation, so causal and attribution share one basis (final-review C2)."""
    per_feature = np.abs(np.asarray(per_feature, dtype=float))
    total = per_feature.sum()
    if total <= 0.0:
        # No signal to attribute; report uniform-by-cardinality shares rather than divide by zero.
        raise ValueError("per-feature importance sums to zero; cannot form shares")
    shares = per_feature / total
    return {name: float(shares[indices].sum()) for name, indices in groups.items()}


def _per_feature_causal(gate_fn, observations, obs_dim, mode, seed):
    """Causal effect of ablating each single feature -> a [obs_dim] vector, for
    the per-feature Spearman comparison against attribution."""
    return np.array([causal_effect(gate_fn, observations, [i], mode, seed=seed)
                     for i in range(obs_dim)], dtype=float)


def _top(group_importance: dict) -> str:
    return max(group_importance, key=group_importance.get)


def gate_response_to_vol_shock(model, market: dict, config: dict, t0: int,
                              width: int, multiplier: float) -> dict:
    """Gate trajectory on the original vs. vol-shocked market, aligned on the same
    decision timeline. Evidence the agent is causally responsive to volatility."""
    from src.interventions import inject_vol_shock  # local import keeps the top clean
    gate_fn = make_gate_fn(model)
    baseline_obs = rollout_observations(model, market, config)
    shocked_market = inject_vol_shock(market, t0=t0, width=width, multiplier=multiplier)
    shocked_obs = rollout_observations(model, shocked_market, config)
    return {"baseline": [float(x) for x in gate_fn(baseline_obs)],
            "shocked": [float(x) for x in gate_fn(shocked_obs)]}


def run_probe(gate_fn, gate_mean_fn, observations, groups, seed: int = 0) -> dict:
    """Pure verdict computation over one agent's replayed decisions. No training,
    no IO. See the module plan for the returned schema."""
    observations = np.asarray(observations, dtype=np.float32)
    obs_dim = observations.shape[1]

    saliency_vec = saliency_importance(gate_mean_fn, observations)
    shap_vec = shap_importance(gate_fn, observations, n_background=SHAP_BACKGROUND,
                               n_explain=SHAP_EXPLAIN, seed=seed)

    causal_vecs = {mode: _per_feature_causal(gate_fn, observations, obs_dim, mode, seed)
                   for mode in ("freeze", "permute")}

    causal_group = {mode: _normalized_group_shares(causal_vecs[mode], groups)
                    for mode in ("freeze", "permute")}
    attribution_group = {
        "saliency": _normalized_group_shares(saliency_vec, groups),
        "shap": _normalized_group_shares(shap_vec, groups),
    }

    spearman = {}
    for mode in ("freeze", "permute"):
        for attr_name, attr_vec in (("saliency", saliency_vec), ("shap", shap_vec)):
            rho, _ = spearmanr(causal_vecs[mode], attr_vec)
            spearman[f"{attr_name}_{mode}"] = float(rho) if np.isfinite(rho) else 0.0

    top_group = {
        "causal_freeze": _top(causal_group["freeze"]),
        "causal_permute": _top(causal_group["permute"]),
        "saliency": _top(attribution_group["saliency"]),
        "shap": _top(attribution_group["shap"]),
    }
    return {"causal": causal_group, "attribution": attribution_group,
            "spearman": spearman, "top_group": top_group}


def _summarize_across_seeds(per_seed: list) -> dict:
    """Across-seed faithfulness summary shared by the gate and tilt experiments: fraction of
    seeds where each method ranks `signal` top (premise from the causal track), plus mean/std
    per-feature Spearman agreement, plus the interpretation caveats."""
    def signal_top_fraction(method_key):
        return float(np.mean([v["top_group"][method_key] == "signal" for v in per_seed]))

    return {
        "signal_is_causal_driver_fraction": signal_top_fraction("causal_freeze"),
        "saliency_signal_top_fraction": signal_top_fraction("saliency"),
        "shap_signal_top_fraction": signal_top_fraction("shap"),
        "spearman_mean": {k: float(np.mean([v["spearman"][k] for v in per_seed]))
                          for k in per_seed[0]["spearman"]},
        "spearman_std": {k: float(np.std([v["spearman"][k] for v in per_seed]))
                         for k in per_seed[0]["spearman"]},
        "caveats": [
            "saliency uses the pre-clip Gaussian mean while causal/SHAP use the clipped/projected "
            "behavioral object; where the policy is decisive a saliency-vs-causal divergence is not "
            "necessarily unfaithfulness.",
            "importances are put on comparable units (saliency=grad x std) and all four methods share "
            "one aggregation basis (per-feature shares summed per group), so a causal-vs-attribution "
            "disagreement reflects mechanism, not feature scale or a joint-vs-per-feature basis mismatch.",
        ],
    }


def run_experiment(config: dict, n_seeds: int) -> dict:
    """Train n_seeds gate agents at the configured signal strength, probe each on
    a held-out market, aggregate the verdict across seeds, and save the run."""
    from src.train import train_agent
    from src.synthetic_market import generate_risky_safe_market

    now = datetime.datetime.now()
    print("=" * 60)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Config      : {config} | n_seeds={n_seeds}")
    print("=" * 60)

    groups = feature_groups(config["window"], n_assets=2)
    per_seed = []
    for seed in range(n_seeds):
        train_market = generate_risky_safe_market(config["n_steps"], seed=1000 + seed,
                                                  signal_strength=config["signal_strength"])
        model = train_agent(train_market, {**config, "seed": seed})
        eval_market = generate_risky_safe_market(config["n_steps"], seed=2000 + seed,
                                                 signal_strength=config["signal_strength"])
        observations = rollout_observations(model, eval_market, config)
        gate_fn = make_gate_fn(model)
        gate_mean_fn = make_gate_mean_fn(model)
        verdict = run_probe(gate_fn, gate_mean_fn, observations, groups, seed=seed)
        print(f"seed={seed}: causal_top={verdict['top_group']['causal_freeze']} "
              f"saliency_top={verdict['top_group']['saliency']} "
              f"shap_top={verdict['top_group']['shap']}", flush=True)
        if seed == 0:
            first_model, first_eval = model, eval_market
        per_seed.append(verdict)

    summary = _summarize_across_seeds(per_seed)

    out_dir = (Path(__file__).resolve().parent.parent / "outputs"
               / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_rq3-faithfulness")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Legibility figure: gate response to a vol shock on the first agent (spec §7).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        curves = gate_response_to_vol_shock(first_model, first_eval, config,
                                            t0=config["n_steps"] // 2, width=5,
                                            multiplier=6.0)
        plt.figure()
        plt.plot(curves["baseline"], label="baseline", linewidth=0.8)
        plt.plot(curves["shocked"], label="vol shock", linewidth=0.8)
        plt.xlabel("decision step"); plt.ylabel("de-risking gate g"); plt.legend()
        plt.title("Gate response to an injected volatility shock")
        plt.savefig(out_dir / "gate_response_vol_shock.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"wrote {out_dir / 'gate_response_vol_shock.png'}")
    except Exception as exc:
        print(f"WARNING: gate-response figure failed (non-fatal): {exc}")
    with open(out_dir / "config.json", "w") as f:
        json.dump({"config": config, "n_seeds": n_seeds}, f, indent=2)
    with open(out_dir / "results.json", "w") as f:
        json.dump({"summary": summary, "per_seed": per_seed}, f, indent=2)
    print(f"Summary: {summary}")
    print(f"Saved run to: {out_dir}")
    return {"summary": summary, "per_seed": per_seed, "out_dir": str(out_dir)}


TILT_CONFIG = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
               "action_mode": "tilt", "max_tilt": 0.15, "market": "multi_regime",
               "n_risky": 3, "n_safe": 2, "total_timesteps": 150_000,
               "n_steps": 6000, "signal_strength": 0.95}


def run_tilt_experiment(config: dict, n_seeds: int) -> dict:
    """Probe the expressive tilt agent's SAFE-BLOCK WEIGHT (directional de-risking) on the
    multi-regime market. Same verdict pipeline as run_experiment; different agent + scalar object."""
    from src.train import train_agent
    from src.synthetic_market import generate_multi_regime_market
    from src.interventions import feature_groups_tilt, make_safe_weight_fn
    from src.attribution import make_safe_weight_mean_fn

    now = datetime.datetime.now()
    print("=" * 60)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__} (tilt)")
    print(f"Config      : {config} | n_seeds={n_seeds}")
    print("=" * 60)

    n_assets = config["n_risky"] + config["n_safe"]
    groups = feature_groups_tilt(config["window"], n_assets)
    base_obs_idx = groups["base_weights"]                       # obs indices of the base block
    safe_asset_idx = list(range(config["n_risky"], n_assets))   # safe cols in the weight vector

    per_seed = []
    for seed in range(n_seeds):
        train_market = generate_multi_regime_market(config["n_risky"], config["n_safe"],
                                                    config["n_steps"], seed=1000 + seed,
                                                    signal_strength=config["signal_strength"])
        model = train_agent(train_market, {**config, "seed": seed})
        eval_market = generate_multi_regime_market(config["n_risky"], config["n_safe"],
                                                   config["n_steps"], seed=2000 + seed,
                                                   signal_strength=config["signal_strength"])
        observations = rollout_observations(model, eval_market, config)
        gate_fn = make_safe_weight_fn(model, base_obs_idx, safe_asset_idx, config["max_tilt"])
        gate_mean_fn = make_safe_weight_mean_fn(model, base_obs_idx, safe_asset_idx, config["max_tilt"])
        verdict = run_probe(gate_fn, gate_mean_fn, observations, groups, seed=seed)
        print(f"seed={seed}: causal_top={verdict['top_group']['causal_freeze']} "
              f"saliency_top={verdict['top_group']['saliency']} "
              f"shap_top={verdict['top_group']['shap']}", flush=True)
        per_seed.append(verdict)

    summary = _summarize_across_seeds(per_seed)
    out_dir = (Path(__file__).resolve().parent.parent / "outputs"
               / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_rq3-faithfulness-tilt")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump({"config": config, "n_seeds": n_seeds}, f, indent=2)
    with open(out_dir / "results.json", "w") as f:
        json.dump({"summary": summary, "per_seed": per_seed}, f, indent=2)
    print(f"Summary: {summary}")
    print(f"Saved run to: {out_dir}")
    return {"summary": summary, "per_seed": per_seed, "out_dir": str(out_dir)}


if __name__ == "__main__":
    run_experiment(CONFIG, N_SEEDS)
