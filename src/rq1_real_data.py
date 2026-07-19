"""RQ1: how much of the gate agent's OOS tail/risk benefit is learned skill vs.
inherited structure — on real ETFs, cost-aware, walk-forward.

Headline statistic (open-questions.md): skill NET of the phase-randomized placebo
null, with a confidence interval. A verdict of "skill_net ~ 0" (residual adds
nothing above the structural base after costs) is a valid, publishable RQ1 answer
(spec §9), not a failure. The metrics table places the agent beside its base and
the literature baselines (1/N, min-variance, CVaR-min) on the same OOS window.
"""
import csv
import json
import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_etf_panel
from src.walk_forward import walk_forward_gate, make_folds
from src.placebo import placebo_null
from src.baselines import minimum_variance_base, cvar_min_base, roll_weights
from src.base_policies import equal_weight_base, vol_scaled_base, risk_parity_base
from src.metrics import (expected_shortfall, max_drawdown, tail_ratio,
                         skewness, sharpe, sortino)

def compute_metrics_bundle(net_returns: np.ndarray) -> dict:
    net_returns = np.asarray(net_returns, dtype=float)
    return {
        "es_99": expected_shortfall(net_returns, alpha=0.99),
        "max_drawdown": max_drawdown(net_returns),
        "tail_ratio": tail_ratio(net_returns),
        "skewness": skewness(net_returns),
        "sharpe": sharpe(net_returns),
        "sortino": sortino(net_returns),
    }


def _oos_start(n_steps: int, config: dict) -> int:
    # The first index of the stitched OOS region (start of the first test block).
    return make_folds(n_steps, config["initial_train"], config["test_block"])[0][1].start


def run_rq1(config: dict, returns: np.ndarray = None, run_dir=None) -> dict:
    now = datetime.datetime.now()
    print("=" * 60)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Config      : {config}")
    print("=" * 60)

    if returns is None:
        panel = load_etf_panel()
        returns = panel["returns"]
        print(f"run_rq1: loaded real panel {returns.shape}")

    if run_dir is None:
        run_dir = Path(__file__).resolve().parent.parent / "outputs" / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_rq1-real-data"
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    n_folds_estimate = len(make_folds(returns.shape[0], config["initial_train"], config["test_block"]))
    print(f"run_rq1: ~{n_folds_estimate} agent folds + {config['n_placebo']} x {n_folds_estimate} placebo "
          f"folds = ~{n_folds_estimate * (1 + config['n_placebo'])} PPO trainings at "
          f"{config['total_timesteps']} steps each — expect a long run on the real panel")

    # Agent: per-fold retrained gate, stitched OOS.
    agent = walk_forward_gate(returns, config, run_dir=run_dir / "agent")
    null = placebo_null(returns, config, n_placebo=config["n_placebo"],
                        seed=config["seed"] + 100, run_dir=run_dir / "placebo")

    # skill net of the luck/overfitting floor, with a CI combining agent-fold and
    # placebo spread (added in quadrature; both are estimates of a mean skill).
    from scipy import stats
    skill_net = agent["mean_skill"] - null["mean"]
    fold_skills = np.asarray(agent["fold_mean_skill"], dtype=float)
    placebo_arr = np.asarray(null["placebo_skills"], dtype=float)
    agent_se = fold_skills.std(ddof=1) / np.sqrt(len(fold_skills)) if len(fold_skills) > 1 else 0.0
    placebo_se = placebo_arr.std(ddof=1) / np.sqrt(len(placebo_arr)) if len(placebo_arr) > 1 else 0.0
    combined_se = float(np.sqrt(agent_se ** 2 + placebo_se ** 2))
    # Small-sample t critical value on the more conservative (smaller) arm's dof — the
    # placebo arm has few draws, so a normal 1.96 would understate the interval width.
    dof = max(1, min(len(fold_skills), len(placebo_arr)) - 1)
    t_crit = float(stats.t.ppf(0.975, dof))
    half_width = t_crit * combined_se
    skill_net_ci = [skill_net - half_width, skill_net + half_width]
    # Distribution-free robustness check: fraction of placebo (luck) runs whose
    # manufactured skill meets or exceeds the agent's OOS skill.
    placebo_exceedance = float(np.mean(placebo_arr >= agent["mean_skill"]))

    oos_start = _oos_start(returns.shape[0], config)
    window = config["window"]
    cost_bps = config["cost_bps"]
    # Roll baselines over the SAME OOS calendar as the stitched agent series: prepend
    # `window` days of real history so the first scored day is exactly oos_start,
    # matching the agent (whose per-fold warm-up is also drawn from prior history).
    baseline_returns = returns[oos_start - window:]
    agent_oos_len = len(agent["oos_port_return"])
    baseline_series = {
        "one_over_n": roll_weights(equal_weight_base, baseline_returns, window, cost_bps),
        "min_variance": roll_weights(minimum_variance_base, baseline_returns, window, cost_bps),
        "cvar_min": roll_weights(cvar_min_base, baseline_returns, window, cost_bps),
    }
    for _name, _series in baseline_series.items():
        assert len(_series) == agent_oos_len, (
            f"OOS length mismatch: baseline {_name} has {len(_series)} steps, "
            f"agent has {agent_oos_len} — agent and baselines must span the same OOS window")
    metrics_table = {
        "agent": compute_metrics_bundle(agent["oos_port_return"]),
        "base": compute_metrics_bundle(agent["oos_base_return"]),
        "one_over_n": compute_metrics_bundle(baseline_series["one_over_n"]),
        "min_variance": compute_metrics_bundle(baseline_series["min_variance"]),
        "cvar_min": compute_metrics_bundle(baseline_series["cvar_min"]),
    }

    result = {
        "mean_skill": agent["mean_skill"],
        "placebo_mean": null["mean"],
        "placebo_exceedance": placebo_exceedance,
        "skill_net": float(skill_net),
        "skill_net_ci": [float(skill_net_ci[0]), float(skill_net_ci[1])],
        "fold_mean_skill": agent["fold_mean_skill"],
        "placebo_skills": null["placebo_skills"],
        "mean_gate": float(agent["oos_gate"].mean()),
        "metrics_table": metrics_table,
    }

    # Persist config + results + a flat metrics CSV.
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(run_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(run_dir / "metrics_table.csv", "w", newline="") as f:
        metric_names = list(next(iter(metrics_table.values())).keys())
        writer = csv.writer(f)
        writer.writerow(["strategy"] + metric_names)
        for strategy, bundle in metrics_table.items():
            writer.writerow([strategy] + [bundle[name] for name in metric_names])

    # Figure 1: skill net of the null, with CI.
    plt.figure()
    plt.axhline(0.0, color="grey", linewidth=0.8)
    plt.bar(["skill_net"], [skill_net],
            yerr=[[skill_net - skill_net_ci[0]], [skill_net_ci[1] - skill_net]], capsize=6)
    plt.ylabel("OOS skill net of placebo null")
    plt.title("RQ1: learned skill above structure (real ETFs)")
    plt.savefig(run_dir / "skill_net.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Figure 2: cumulative wealth, agent vs base.
    plt.figure()
    plt.plot(np.cumprod(1.0 + agent["oos_port_return"]), label="agent (gate)")
    plt.plot(np.cumprod(1.0 + agent["oos_base_return"]), label="base")
    plt.legend()
    plt.ylabel("cumulative wealth (OOS)")
    plt.title("Agent vs structural base, net of costs")
    plt.savefig(run_dir / "cumulative_wealth.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Figure 3: gate over time (when does the agent de-risk?).
    is_tilt = config.get("action_mode", "gate") == "tilt"
    activity_label = "residual tilt activity (0.5*sum|w-base|)" if is_tilt else "de-risking gate g"
    activity_title = ("Learned residual tilt activity over time" if is_tilt
                      else "Learned de-risking gate over time")
    plt.figure()
    plt.plot(agent["oos_gate"])
    plt.ylabel(activity_label)
    plt.xlabel("OOS step")
    plt.title(activity_title)
    plt.savefig(run_dir / "gate_timeseries.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"run_rq1: mean_skill={agent['mean_skill']:.6e} placebo={null['mean']:.6e} "
          f"skill_net={skill_net:.6e} CI={skill_net_ci}")
    print(f"Saved run to: {run_dir}")
    return result


if __name__ == "__main__":
    from src.data import load_etf_panel

    panel = load_etf_panel()
    tickers = panel["tickers"]
    # Resolve the safe sleeve by NAME: yfinance returns columns alphabetically, so a
    # hardcoded integer index is fragile. Prefer intermediate Treasuries (IEF), else
    # long-duration (TLT). Fail loudly if neither survived the universe.
    if "IEF" in tickers:
        safe_ticker = "IEF"
    elif "TLT" in tickers:
        safe_ticker = "TLT"
    else:
        raise ValueError(f"no Treasury safe sleeve (IEF/TLT) in panel tickers {tickers}")
    safe_index = tickers.index(safe_ticker)
    print(f"rq1 __main__: safe sleeve = {safe_ticker} at column {safe_index} of {tickers}")

    _config = {"base_name": "risk_parity", "window": 20, "cost_bps": 10.0,
               "safe_asset_index": safe_index, "total_timesteps": 150_000, "seed": 0,
               "initial_train": 1260, "test_block": 252, "n_placebo": 5}
    run_rq1(_config, returns=panel["returns"])
