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


def _activity_diagnostics(gate_fn, observations, groups: dict, seed: int = 0) -> dict:
    """Absolute-magnitude view of how much the agent actually ACTS, so a near-inactive real
    agent (RQ1 mean gate ~0.04) is VISIBLE. The normalized group-shares in run_probe always
    sum to 1 — they look identical for a real mechanism and for pure noise. This reports the
    gate mean/std over the rollout and the raw (un-normalized) |freeze causal effect| per
    group; if gate_std and every causal magnitude are ~0 the agent barely acts and any
    method-agreement number is attribution of noise."""
    observations = np.asarray(observations, dtype=np.float32)
    gate = np.asarray(gate_fn(observations), dtype=float)
    causal_magnitude = {name: causal_effect(gate_fn, observations, indices, "freeze", seed=seed)
                        for name, indices in groups.items()}
    return {"gate_mean": float(gate.mean()), "gate_std": float(gate.std()),
            "causal_magnitude": causal_magnitude}


def _probe_or_null(gate_fn, gate_mean_fn, observations, groups: dict, seed: int = 0) -> dict:
    """run_probe, but if importances collapse to zero (a constant / inactive agent, which
    makes the normalized group-shares undefined) record a LOGGED degenerate null instead of
    crashing. This is the honest 'no measurable mechanism' path for the real agent."""
    try:
        verdict = run_probe(gate_fn, gate_mean_fn, observations, groups, seed=seed)
    except ValueError as exc:
        # Expected, documented case: a near-inactive agent has ~zero causal/attribution
        # importance, so per-feature shares cannot be formed. Surface it loudly and record it.
        print(f"_probe_or_null: degenerate agent (no measurable mechanism): {exc}", flush=True)
        return {"degenerate": True, "reason": str(exc)}
    return {**verdict, "degenerate": False}


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


# "market" here is documentation-only: run_tilt_experiment calls generate_multi_regime_market
# directly (build_env/train_agent do not read config["market"]). It is NOT a dispatch switch.
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


# --- Real-data probe (no ground truth; method-agreement + activity diagnostics) ---
# RQ1's headline config (risk_parity base, full real panel). safe_asset_index and n_assets
# are resolved at runtime from the loaded panel (columns are alphabetical — never hardcode).
REAL_CONFIG = {"base_name": "risk_parity", "window": 20, "cost_bps": 10.0,
               "total_timesteps": 150_000}


def _summarize_real_across_seeds(per_seed: list) -> dict:
    """Real-data summary. There is NO ground-truth 'signal is the true driver' premise on real
    data, so this reports METHOD AGREEMENT (per-feature causal-vs-attribution Spearman) over the
    non-degenerate seeds, the fraction of seeds with no measurable mechanism, and the activity
    diagnostics that tell the reader whether the agreement reflects a real mechanism or noise."""
    n = len(per_seed)
    live = [v for v in per_seed if not v["degenerate"]]
    degenerate_fraction = float(np.mean([v["degenerate"] for v in per_seed]))

    spearman_mean, spearman_std = {}, {}
    if live:
        keys = live[0]["spearman"].keys()
        spearman_mean = {k: float(np.mean([v["spearman"][k] for v in live])) for k in keys}
        spearman_std = {k: float(np.std([v["spearman"][k] for v in live])) for k in keys}

    diags = [v["diagnostics"] for v in per_seed]   # every seed carries diagnostics, degenerate or not
    group_names = list(diags[0]["causal_magnitude"].keys())
    diagnostics = {
        "gate_mean": float(np.mean([d["gate_mean"] for d in diags])),
        "gate_std_mean": float(np.mean([d["gate_std"] for d in diags])),
        "causal_magnitude_mean": {g: float(np.mean([d["causal_magnitude"][g] for d in diags]))
                                  for g in group_names},
    }
    # Descriptive only (no truth to score against): which group each live seed's causal track ranks top.
    causal_top_group_by_seed = [v["top_group"]["causal_freeze"] for v in live]

    return {
        "n_seeds": n,
        "degenerate_fraction": degenerate_fraction,
        "spearman_mean": spearman_mean,
        "spearman_std": spearman_std,
        "diagnostics": diagnostics,
        "causal_top_group_by_seed": causal_top_group_by_seed,
        "caveats": [
            "no ground truth on real data: the `signal` feature is a causal no-lookahead crisis "
            "heuristic, not the known driver, so only METHOD AGREEMENT (causal vs attribution) is "
            "adjudicable here — not faithfulness against truth.",
            "in-sample rollout: the agent is probed on the series it trained on because this "
            "attributes the DECISION MECHANISM (not out-of-sample skill, which RQ1 already settled).",
            "read the Spearman agreement TOGETHER with `diagnostics`: if gate_std and the raw "
            "causal_magnitude are ~0 the agent barely acts, so any agreement is attribution of noise.",
            "seeds vary PPO training only (real data is a single history), unlike the synthetic "
            "probes which draw independent train/eval markets per seed.",
        ],
    }


def run_real_experiment(config: dict, n_seeds: int) -> dict:
    """Probe the REAL-DATA gate agent's decision mechanism. No ground truth exists on real data,
    so this measures METHOD AGREEMENT (causal vs SHAP/saliency) reported alongside activity
    diagnostics. See docs/design_2026-07-20_rq3-real-faithfulness.md."""
    from src.train import train_agent
    from src.data import load_etf_panel
    from src.real_market import build_real_market

    now = datetime.datetime.now()
    print("=" * 60)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__} (real)")
    print(f"Config      : {config} | n_seeds={n_seeds}")
    print("=" * 60)

    panel = load_etf_panel()
    returns = panel["returns"]
    tickers = panel["tickers"]
    print(f"run_real_experiment: loaded real panel {returns.shape} tickers={tickers}")

    # Resolve the safe sleeve by NAME (yfinance columns are alphabetical — never hardcode the
    # index), matching src/rq1_real_data.py.__main__. Fail loud if no Treasury sleeve survived.
    if "IEF" in tickers:
        safe_ticker = "IEF"
    elif "TLT" in tickers:
        safe_ticker = "TLT"
    else:
        raise ValueError(f"no Treasury safe sleeve (IEF/TLT) in panel tickers {tickers}")
    safe_asset_index = tickers.index(safe_ticker)
    n_assets = returns.shape[1]
    print(f"run_real_experiment: safe sleeve = {safe_ticker} at column {safe_asset_index}, n_assets={n_assets}")

    market = build_real_market(returns, safe_asset_index, window=config["window"])
    groups = feature_groups(config["window"], n_assets)

    per_seed = []
    for seed in range(n_seeds):
        seed_config = {**config, "seed": seed, "safe_asset_index": safe_asset_index}
        model = train_agent(market, seed_config)
        observations = rollout_observations(model, market, seed_config)
        gate_fn = make_gate_fn(model)
        gate_mean_fn = make_gate_mean_fn(model)
        verdict = _probe_or_null(gate_fn, gate_mean_fn, observations, groups, seed=seed)
        verdict["diagnostics"] = _activity_diagnostics(gate_fn, observations, groups, seed=seed)
        diag = verdict["diagnostics"]
        mags = {k: round(v, 5) for k, v in diag["causal_magnitude"].items()}
        print(f"seed={seed}: degenerate={verdict['degenerate']} gate_mean={diag['gate_mean']:.4f} "
              f"gate_std={diag['gate_std']:.4f} causal_mag={mags}", flush=True)
        per_seed.append(verdict)

    summary = _summarize_real_across_seeds(per_seed)
    out_dir = (Path(__file__).resolve().parent.parent / "outputs"
               / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_rq3-faithfulness-real")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump({"config": config, "n_seeds": n_seeds, "safe_ticker": safe_ticker,
                   "safe_asset_index": safe_asset_index, "n_assets": n_assets}, f, indent=2)
    with open(out_dir / "results.json", "w") as f:
        json.dump({"summary": summary, "per_seed": per_seed}, f, indent=2)
    print(f"Summary: {summary}")
    print(f"Saved run to: {out_dir}")
    return {"summary": summary, "per_seed": per_seed, "out_dir": str(out_dir)}


if __name__ == "__main__":
    run_experiment(CONFIG, N_SEEDS)
