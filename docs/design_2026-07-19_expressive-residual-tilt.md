# Design Spec — Expressive Residual-Tilt Agent (gives the RL model a fair shot at skill)

**Status:** Design (approved in brainstorming 2026-07-19, pre-plan).
**Extends:** `design_2026-07-18_skill-vs-structure-residual-rl.md` (MC1). Same
research question and the same measurement invariants; this document only changes
the *model* (the agent's action + observation), not the method that measures it.
**Relation to results so far:** Plan 1 (synthetic RQ2) validated the skill measure;
Plan 2 (real RQ1 + robustness sweep) found the minimal **de-risking-gate** agent adds
no skill above structure (skill_net < 0, exceedance 1.00, across 3 bases × 5 seeds).

---

## 1. Motivation

The gate agent's only lever is a scalar `g∈[0,1]` that blends the base with a safe
asset — it can *de-risk or not*, nothing else. So "it adds ~no skill" is partly a
property of how little we let it do. To make the verdict about *RL capability*
rather than *action-space poverty*, give the agent a genuinely expressive action
(tilt toward/away from any asset) while keeping every invariant that makes the skill
measurement trustworthy. The result is meaningful in both directions:

- If the expressive agent beats structure **net of the placebo null** (and passes the
  synthetic RQ2 gate), that is *real learned skill* — a positive finding.
- If it still does not, that is a **stronger H1**: even a capable, well-fed agent
  cannot beat transparent structure after costs.

## 2. What stays fixed (the measurement invariants — do NOT change)

- **Structure-baselined reward:** `r_t = agent_net_log_return_t − base_net_log_return_t`
  on the same path, each charged its own turnover cost. This is what isolates skill.
- **Long-only simplex** executed weights (`w ≥ 0`, `sum w = 1`), via exact projection.
- **Free "do-nothing" floor:** the zero action must reproduce the base exactly, so the
  agent only deviates when deviating pays.
- **Structural-null base ladder:** equal-weight / vol-scaled / risk-parity (ERC).
- **Skill reported net of the phase-randomization placebo null**, with a CI, on real data.
- **Synthetic ground-truth (RQ2) re-validation before any real-data claim.**

## 3. Action — bounded residual tilt (the one change to the policy output)

The agent outputs a per-asset tilt `a ∈ ℝⁿ`; executed weights are

```
w = project_to_simplex( base_weights + max_tilt * tanh(a) )
```

- `max_tilt` is a config cap (default **0.15**) on how far each asset can move from base.
- `a = 0  ⇒  w = base_weights` (do-nothing floor, free).
- `tanh` bounds each component to `(−max_tilt, +max_tilt)` before projection.

**Why this and not the earlier N-dim tilt:** the original design used an *unbounded*
tilt (`w = project_simplex(base + a)`); PPO over-tilted (mean|a|≈0.7), churned turnover
every step, and scored negative skill even with a near-perfect signal — it never learned
the do-nothing floor. The bound (`tanh`), the small cap (`max_tilt`), and the existing
per-step turnover cost together make parsimony cheap and over-tilting expensive, so the
agent can learn to sit at the base and deviate only for a real edge. This action subsumes
de-risking (the agent can tilt toward the Treasury sleeve itself), so no separate
safe-asset blend is needed in tilt mode.

`max_tilt` is the key hyperparameter; it is **tuned via the RQ2 gate** (§6): too loose ⇒
spurious skill with the signal off; too tight ⇒ cannot express skill with the signal on.

## 4. Observation — enriched (tilt mode only)

The gate obs (flattened return window + per-asset trailing vol + scalar crisis signal)
is thin for cross-asset selection. In **tilt mode** add, per asset:

- **momentum:** trailing mean return over the window,
- **a second, longer volatility** horizon (e.g., 2×window),
- **the current base weights** (so the agent knows its reference point).

Concretely the tilt-mode observation is the concatenation of: flattened return window
`(window*n_assets)`, short-horizon per-asset vol `(n_assets)`, long-horizon per-asset vol
`(n_assets)`, per-asset momentum `(n_assets)`, base weights `(n_assets)`, and the crisis
signal `(1)`. All features are **causal** (computed only from data strictly before the
decision step) — the no-lookahead rule is a hard constraint and is unit-tested.

**Gate-mode observation is unchanged**, so the already-validated gate results remain
reproducible; enrichment is scoped to the new mode.

## 5. Architecture

Parameterize the existing `AllocationEnv` with `action_mode ∈ {"gate", "tilt"}`
(plus `max_tilt` and the tilt-mode feature set). The step/reward logic, base-policy
selection, turnover accounting, and `last_info` are shared; only the action→weights map,
the `action_space`, and the observation vector differ by mode. This keeps the tested gate
env intact and lets the same training/eval/walk-forward/placebo machinery run either model,
enabling a direct gate-vs-tilt comparison with zero duplication of the reward core.

Everything downstream (`train`, `validate_skill`, `walk_forward`, `placebo`,
`rq1_real_data`, the LSF drivers) is reused unchanged except for passing `action_mode`
(and `max_tilt`) through the config.

## 6. Validation flow (the safety gate — order matters)

1. **RQ2 re-validation FIRST (synthetic ground truth).** Train the tilt agent on the
   regime-switching market with the signal ON vs OFF. Require: mean structure-baselined
   skill ≈ 0 with signal OFF (the agent must learn tilt≈0 — no manufactured skill) and
   clearly positive with signal ON. This re-calibrates the measure *for the new model*.
   Failure signal-off ⇒ the action is too loose: lower `max_tilt` / raise cost, and repeat.
   No real-data claim is made until this passes.
2. **RQ1 real-data + robustness sweep.** Only after (1) passes: run the walk-forward on
   real ETFs across the base ladder and seeds, report **skill net of the placebo null**
   with CIs and placebo-exceedance — exactly as in Plan 2 — now for the tilt agent, with
   the gate agent reported alongside as the reference point.

## 7. Success criteria

A trustworthy verdict for the *expressive* agent: (a) it passes the RQ2 ground-truth gate
(the measure is valid for this model), and (b) a clean real-data statement of how much
skill it adds net of the null. Either outcome is a result: real learned skill (positive,
validated) or a stronger "structure dominates even a capable agent" (H1). Success is **not**
beating the market.

## 8. Testing

- `action=0` reproduces the base exactly and yields reward 0 (do-nothing floor).
- Weights stay on the simplex (`w≥0`, `sum=1`) for arbitrary tilt actions.
- New observation features are causal (no-lookahead): altering `returns[t:]` never changes
  the observation at step `t`.
- **RQ2 ground-truth gate for the tilt model** (skill vanishes signal-off, appears signal-on)
  — the load-bearing validity test; thresholds encode intent (do not weaken to pass).
- Tilt agent beats base out-of-sample when a strong signal exists (end-to-end skill detection).
- Gate-mode behavior is unchanged (regression: existing gate tests still pass).

## 9. Scope / YAGNI

- PPO only; long-only simplex only; same universe, costs, and walk-forward protocol as Plan 2.
- No new RL algorithms, no alternative reward, no market-cap/diversity data.
- MC2 causal probing (RQ3) is deferred to a later plan (staged so a slip degrades gracefully
  to the MC1-synthetic + RQ1-real + expressive-agent paper).

## 10. Risks

- **Tilt instability / over-churn returns** → mitigated by the bounded tanh + `max_tilt` cap
  + turnover cost, and *caught* by the RQ2 signal-off gate before any real-data claim.
- **Spurious skill from a looser action** (higher overfitting floor) → this is exactly why
  the placebo null and RQ2 gate exist; report skill net of the null, and tune `max_tilt` on RQ2.
- **Observation enrichment leaks lookahead** → causal-by-construction + explicit no-lookahead test.
