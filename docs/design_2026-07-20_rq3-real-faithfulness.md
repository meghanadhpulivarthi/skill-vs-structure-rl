# RQ3 on the real-data agent — causal-faithfulness probe (design)

**Date:** 2026-07-20
**Status:** approved (build it; honest null acceptable)

## Question

The synthetic RQ3 probes (gate, tilt) asked: *when an agent has a known mechanism,
do post-hoc explanations (SHAP, saliency) identify it?* On synthetic data the
`signal` feature is the true driver **by construction**, so faithfulness is
adjudicable against ground truth.

This phase asks the same machinery of the **real-data** gate agent. Two structural
differences change what is answerable, and the design is built around them.

## What is (and is not) adjudicable here

1. **No ground truth.** On real ETFs the crisis `signal` is a *causal, no-lookahead
   heuristic* (trailing equal-weight realized vol, `src/real_market.py`) — one input
   among 232, with no established "true importance". There is no oracle. So we cannot
   ask "did attribution find the true driver". We can only measure **method agreement**:
   - causal ablation ↔ SHAP, causal ablation ↔ saliency (per-feature Spearman),
   - and implicitly SHAP ↔ saliency.
   This is a **transfer / robustness check** on the synthetic finding, not a
   faithfulness verdict. The write-up must frame it that way.

2. **The real agent barely acts.** RQ1 found mean gate ≈ 0.044 — it stays ~96 % on the
   base (no learned skill). A near-constant decision surface means freeze/permute
   ablations may move the gate negligibly for *every* feature, and the normalized
   group-shares (which always sum to 1) would then be attributing **pure noise** while
   looking like a clean result. A null is the *expected* outcome and is a legitimate
   finding — it reinforces the RQ1 "no mechanism" verdict.

## Design decisions

- **Agent probed.** One gate agent trained on the **full real panel** with RQ1's
  headline config: `base_name="risk_parity"`, `window=20`, `cost_bps=10`,
  `total_timesteps=150_000`. The safe sleeve is resolved by **ticker** (IEF, else TLT;
  never a hardcoded index — yfinance columns are alphabetical), exactly as
  `src/rq1_real_data.py.__main__`. This is a faithful representative of RQ1's per-fold
  agents (same env, base, reward, budget). It is **not** literally an RQ1 walk-forward
  agent (there is no single such agent — RQ1 stitches per-fold models).

- **In-sample rollout, stated as a caveat.** We roll out and probe on the same real
  series the agent trained on. This is acceptable because the probe attributes the
  agent's **decision mechanism**, not out-of-sample skill (which RQ1 already settled).
  A generalization split is unnecessary and would only shorten the rollout.

- **Seeds vary PPO training only.** Real data is a single history, so the 5-seed spread
  reflects PPO training stochasticity on fixed data — not data resampling (unlike the
  synthetic probes, which draw independent train/eval markets per seed).

- **Reuse.** `feature_groups(20, 11)`, `make_gate_fn`, `make_gate_mean_fn`,
  `rollout_observations`, and `run_probe` all work unchanged (obs layout is the same
  `[returns | short_vol | signal]` structure, only larger). No new probe math.

## The honest addition: activity / magnitude diagnostics

Normalized group-shares hide inactivity — they sum to 1 whether the agent has a real
mechanism or is responding to noise. So the real runner additionally reports, per seed
and aggregated:

- **`gate_mean`, `gate_std`** over the rollout — how much the agent actually moves the
  gate (compare to RQ1's 0.044). A tiny `gate_std` ⇒ the agent does almost nothing.
- **`causal_magnitude`** — the *raw, un-normalized* `|causal effect|` (freeze) per
  feature group, i.e. the absolute gate change under ablation. If all groups are ~0,
  there is no mechanism to attribute, and the Spearman agreement numbers are noise.

These make a null **visible** instead of dressing it as a result.

## Degenerate guard

If a seed's importances collapse to exactly zero (a constant agent),
`_normalized_group_shares` raises. The runner catches this **loudly** (prints the
condition, records the seed as `degenerate: true` with a `null` verdict) rather than
crashing the whole run or silently swallowing it — consistent with the no-silent-
failures rule.

## Outputs

`outputs/<ts>_rq3-faithfulness-real/` with `config.json` and `results.json`
(`summary` incl. the diagnostics block + `per_seed`). LSF CPU driver
`scripts/rq3_faithfulness_real.sh` (thread-capped; the panel cache must already exist
under `data/` — it does, from the RQ1 runs).

## Success criteria

- The probe runs to completion on the real panel across 5 seeds without crashing on the
  degenerate case.
- The report states, honestly: (a) the method-agreement Spearman numbers, and (b) the
  activity diagnostics that say whether those numbers reflect a real mechanism or noise.
- A null ("agent too inactive to have a probeable mechanism; method agreement is
  therefore uninformative") is an acceptable, correctly-framed verdict.
