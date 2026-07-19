# Decisions

Choices made and why. Newest first.

## 2026-07-19 — Tilt RQ2 calibration: max_tilt=0.15 passes on multi-regime market

**Chosen `max_tilt`: 0.15** (the brief's initial value — no sweep needed).

Tilt RQ2 test on multi_regime market (n_risky=3, n_safe=2, 120k steps, 3 seeds):
- `skill_off` (signal=0.0): **-9.5e-6** (|skill_off| = 9.5e-6 < 5e-5 ✓)
- `skill_on` (signal=0.95): **9.38e-5** (> 5e-5 ✓; > 3×|skill_off| ✓)

All three assertions pass at max_tilt=0.15. No sweep of {0.20, 0.25} was needed.
Run artifact: `outputs/2026-07-19_11-24-05_skill-validation/`.

The tilt agent on the multi-regime market (5 assets: 3 risky + 2 safe) exhibits the
same RQ2 shape as the gate agent on the 2-asset risky_safe market: skill vanishes
without timeable structure and is clearly positive when the signal leads crises.
This validates the skill measure for the expressive tilt action.

## 2026-07-19 — RQ1 headline statistics: OOS-window alignment + small-sample CI

Two decisions locked while resolving the final whole-branch review of Plan 2:

1. **Agent and baselines are scored on the SAME OOS calendar.** The walk-forward
   agent is evaluated per fold on the test slice *with `window` days of real preceding
   history prepended*, so the env's warm-up window comes from genuine prior data and the
   stitched agent series covers exactly `[oos_start, end)` with no per-fold day drop and
   no cold-started signal. Baselines are rolled over `returns[oos_start-window:]` so they
   span the identical calendar; `run_rq1` asserts equal OOS lengths. Rationale: the RQ1
   skill-vs-structure comparison is only defensible if both sides are measured on matching
   days and denominators (spec §8). This fixed a real bias (dropped agent days clustered
   at block starts, right after each retrain).

2. **skill_net CI uses a small-sample t critical value, not 1.96.** The CI on
   `skill_net = agent_mean_skill − placebo_mean` combines the agent's cross-fold SE and the
   placebo's across-draw SE in quadrature, scaled by `t.ppf(0.975, dof)` with
   `dof = min(n_folds, n_placebo) − 1` (the placebo arm has few draws, so a normal 1.96
   understates width). A distribution-free `placebo_exceedance` (fraction of placebo runs
   whose manufactured skill ≥ the agent's) is also reported as a robustness check.
   **Open for the real run:** the exact CI construction (quadrature-of-SEs vs. a bootstrap
   over the stitched OOS skill against the placebo distribution) should be revisited once
   the real-data verdict's magnitude is known — it is less load-bearing for a
   skill_net≈0 verdict, more so for a positive one. See [[open-questions]].

## 2026-07-19 — Phase-randomization placebo destroys volatility clustering (not linear autocorr)

Phase randomization preserves the linear power spectrum and hence linear autocorrelation
(by Wiener-Khinchin theorem) — it does NOT destroy 2nd-order linear structure. What it
destroys is volatility clustering (squared-return autocorrelation), which is exactly the
timeable regime/vol-clustering structure the de-risking gate exploits via its trailing-vol
signal. The original Plan-2 Task-6 test asserting linear-autocorrelation destruction was
scientifically incorrect and has been replaced with a volatility-clustering-destruction test.
The null remains valid: it removes exactly the timeable regime/vol structure the gate
exploits while preserving the linear structure the base already harvests.

## 2026-07-19 — Base policy: §5.1(3) realized as ERC (not diversity-weighted SPT)

Spec §5.1(3) names an "SPT / diversity-weighted (growth-optimal-oriented)" base to span growth.
Diversity-weighting requires market-cap weights we do not have in the ETF universe.
More fundamentally, its growth benefit (rebalancing premium) is already spanned by `equal_weight`.
The unspanned structural channel left open by equal-weight + inverse-vol is **cross-asset correlation**
and its covariance structure. Risk-parity (ERC: equal-risk-contribution) closes this channel
transparently and non-learned, making it the right fit for §5.1(3). Implemented with a sqrt-damped multiplicative fixed-point iteration that shares the ERC fixed point w_i·(Σw)_i=const with the naive 1/(Σw) map but converges stably for high vol-ratio inputs (the naive map oscillates). The test uses the plan-mandated independent-normals inputs (σ=0.005 vs σ=0.05, seed 0).

## 2026-07-18 — RQ2 test threshold recalibrated (5e-5, not the plan's 2e-4)

The Plan-1 RQ2 test (`tests/test_validate_skill.py`) originally specified
`skill_on > 2e-4`. After the action-design pivot (see below), the scalar
de-risking gate earns **less absolute skill** than the original N-dim tilt was
imagined to: measured signal-on skill ≈ 8e-5 (prototype, 1 seed, 120k) to
2.3e-4 (2 seeds, 100k). We recalibrated the absolute floor to `skill_on > 5e-5`
— comfortably above the ~1e-5 noise floor (measured signal-off skill) and below
observed on-skill — and kept the relative gate `skill_on > 3*|skill_off|` plus
`|skill_off| < 5e-5`. Seeds raised 2 → 3 for robustness.
**Why surfaced here:** a final-review flagged that lowering the plan's threshold
silently would violate the fail-loud / tests-encode-intent rules. The floor must
not be lowered further without re-checking the noise floor.

## 2026-07-18 — Action design: N-dim tilt → scalar de-risking gate

Original spec §5.2 had the residual policy output an unconstrained N-dim tilt on
the base (`w = project_simplex(base + a)`). Empirically PPO over-tilted
(mean|action|≈0.7), paid turnover every step, and scored **negative** skill even
against the weakest base with a near-perfect signal (150k steps, VecNormalize) —
it never learned the "do nothing" floor.
**Decision (user-approved):** replace with a scalar **de-risking gate** g∈[0,1],
`w = (1-g)·base + g·safe`. g=0 reproduces the base (zero-skill floor by
construction); the agent only raises g when timed de-risking pays.
**Why:** makes parsimony structural, kills the turnover churn, and the action
space matches the scientific object (learned de-risking timing). Prototype
confirmed the RQ2 shape (signal-on skill>0, signal-off skill≈0). Spec §5.2 updated.

## 2026-07-18 — Research direction: audit → skill-vs-structure method

Pivoted from the original `research_problem.md` framing (honest-protocol audit of
Lavko/Klein/Walther 2023) — found largely scooped (Kruthof & Müller 2025) and too
paper-derivative — to a reusable **method** contribution: skill-isolating residual
RL that measures how much of an RL allocator's tail benefit is learned skill vs.
inherited structure, validated on synthetic ground truth (MC1) with a staged
causal-probing extension (MC2). See `docs/design_2026-07-18_skill-vs-structure-residual-rl.md`
and the `context/` novelty-check trail.

## 2026-07-18 — Run scripts as modules

Entrypoints use absolute `from src...` imports, so run them as modules:
`uv run python -m src.train` and `uv run python -m src.validate_skill`
(NOT `uv run python src/validate_skill.py`, which fails with "No module named src").
