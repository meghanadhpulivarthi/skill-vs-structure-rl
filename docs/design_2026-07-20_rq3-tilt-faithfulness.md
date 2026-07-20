# RQ3 Faithfulness Probe — Tilt Agent Extension: Design Spec

**Date:** 2026-07-20
**Status:** approved design (key fork answered), pre-plan
**Extends:** `docs/design_2026-07-19_mc2-causal-probing.md` (the gate-agent faithfulness probe).
Depends on the MC2 harness (`src/interventions.py`, `src/attribution.py`,
`src/rq3_faithfulness.py`) and the expressive tilt agent (Plan 3).

---

## 1. Question and motivation

The gate-agent RQ3 verdict (2026-07-20) found attribution *largely faithful* in a clean,
low-dimensional, single-mechanism setting, and flagged that reliability frays as the mechanism
becomes more distributed. The expressive **tilt** agent is exactly the harder case: a per-asset
action on a richer multi-regime market, where the agent over-churns and its de-risking is spread
across assets. **Does post-hoc attribution stay faithful for the more capable agent, or does it
break where the gate result predicted it would?** Same H3, harder testbed. Falsifiable either way.

## 2. Testbed and the behavioral object

- **Agent / market:** the tilt agent (`action_mode="tilt"`, `max_tilt=0.15`) trained on the
  multi-regime market (`generate_multi_regime_market`, `n_risky=3`, `n_safe=2`), `base_name=
  "equal_weight"`, `signal_strength=0.95`, `window=20`, `total_timesteps=150_000` — the exact
  config the tilt RQ2 gate was validated on. The `signal` feature remains the known ground-truth
  driver (it predicts the next crisis).
- **Scalar behavioral object = safe-block weight** `sum(w over the 2 safe assets)` — the agent's
  *directional de-risking* (user-approved). This is the tilt analog of the gate's `g`: it is what
  the signal→crisis→move-to-safety mechanism should drive. The probe attributes this scalar, so
  the whole gate-agent machinery (causal ablation, saliency, SHAP, Spearman, ground-truth anchor)
  transfers unchanged; only the scalar it reads is different.
- Safe assets are the LAST `n_safe` columns (indices 3,4 for 3 risky + 2 safe).

## 3. Observation layout and feature groups

Tilt obs on this market (`n_assets=5`, `window=20`) is 121-dim, laid out (per
`src/allocation_env.py`) as:

```
[ win.flatten() | short_vol | long_vol | momentum | base_weights | signal ]
[   20*5 = 100  |    5      |    5     |    5     |      5       |   1    ]
```

Feature groups (6):

| group | indices | role |
|---|---|---|
| `returns` | 0–99 | correlate |
| `short_vol` | 100–104 | correlate |
| `long_vol` | 105–109 | correlate |
| `momentum` | 110–114 | correlate |
| `base_weights` | 115–119 | correlate (the base the tilt departs from) |
| `signal` | 120 | **true driver** |

## 4. What is new vs. reused

**Reused UNCHANGED** (obs-dim- and scalar-agnostic): `rollout_observations`, `freeze_group`,
`permute_group`, `causal_effect`, `saliency_importance`, `shap_importance`,
`aggregate_to_groups`, `_normalized_group_shares`, `run_probe`. The confound fixes (grad×std,
normalized per-feature group shares) carry over automatically.

**New, small, isolated:**
1. `feature_groups_tilt(window, n_assets, n_safe)` — the 6-group layout above. (Keep the existing
   `feature_groups` for the gate untouched.)
2. `project_to_simplex_torch(v)` — a batched, differentiable torch mirror of the numpy Duchi
   projection in `src/simplex.py` (needed so saliency can backprop through the tilt→weights map).
3. `make_safe_weight_fn(model, base_idx, safe_idx, max_tilt)` (numpy) and
   `make_safe_weight_mean_fn(model, base_idx, safe_idx, max_tilt)` (torch, differentiable) — the
   tilt analogs of `make_gate_fn`/`make_gate_mean_fn`. Both map an observation to the safe-block
   weight of the executed portfolio:
   - read `base_weights` from the obs (indices `base_idx`),
   - `tilt = max_tilt * tanh(action_mean)` (numpy: deterministic `predict`; torch: policy Gaussian
     mean, pre-clip),
   - `w = project_to_simplex(base_weights + tilt)`,
   - return `w[safe_idx].sum()`.
   Reading `base_weights` from the obs (rather than recomputing) keeps the summary a pure function
   of (obs, policy) and treats `base_weights` as the input channel it is — consistent with how the
   probe treats every derived feature as an independent input.
4. `run_tilt_experiment(config, n_seeds)` in `src/rq3_faithfulness.py` — mirrors `run_experiment`
   but trains tilt agents on the multi-regime market, uses `feature_groups_tilt` + the safe-weight
   adapters, and saves to `outputs/<ts>_rq3-faithfulness-tilt/`. The premise gate becomes
   "`signal` is the top causal driver of the SAFE-WEIGHT decision"; verdict schema is identical.

## 5. Verdict (identical structure to the gate probe)

Per seed: `top_group` per method (causal_freeze/permute, saliency, shap) + per-feature Spearman
(causal vs each attribution). Across 5 seeds: `signal_is_causal_driver_fraction` (premise, from the
causal track), `saliency_signal_top_fraction`, `shap_signal_top_fraction`, and mean±std Spearman.
Same clip vs pre-clip caveat carries in `results.json` (saliency uses the pre-clip Gaussian mean;
causal/SHAP use the clipped/projected behavioral safe-weight).

**Interpretation frame:** compare directly to the gate verdict. If attribution stays faithful
(signal top, positive Spearman), that *extends* the boundary condition to a capable agent. If it
frays (signal not top, low/negative Spearman, high variance), that *confirms* the gate result's
predicted failure mode — attribution misleads exactly where the mechanism is distributed and the
agent over-churns. Either way it is a real, honest result.

## 6. Testing (Rule 9)

- `project_to_simplex_torch`: matches the numpy `project_to_simplex` on random batches (allclose),
  and its output lies on the simplex (nonneg, sums to 1). A known-answer gate on the instrument.
- Safe-weight adapters: on a stub policy with a known constant action, the numpy and torch safe-weight
  agree with a hand-computed `project_to_simplex(base + max_tilt*tanh(action))[safe].sum()`.
- `feature_groups_tilt`: exactly partitions the 121-dim obs (no overlap, full cover).
- End-to-end (fast, no training): a stub policy whose action depends ONLY on the signal feature must
  make all four methods rank the `signal` group top for the safe-weight object (mirrors the gate
  `run_probe` gate). Keep stochastic-training assertions as smoke/ground-truth, not exact values.

## 7. Compute

Heavier than the gate probe (5-asset, 121-dim obs, 150k steps × 5 seeds, plus KernelSHAP over 121
features). Runs on **LSF, thread-capped** (heavy PPO → not the login node), via a driver mirroring
`scripts/rq3_faithfulness.sh`. GPU intentionally unused.

## 8. Success criteria

Instrument tests pass (torch projection matches numpy; adapters match hand-computed safe-weight);
the probed tilt agents pass the premise (signal is the top causal driver of safe-weight); a
faithfulness verdict (signal-dominance per method + Spearman, across-seed CIs) is produced, saved,
and interpreted against the gate result — in whichever direction the data falls.
