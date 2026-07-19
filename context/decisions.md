# Decisions

Choices made and why. Newest first.

## 2026-07-19 — RQ1 REAL-DATA VERDICT (TILT): expressive agent adds no skill, and loses MORE than the gate

**Result (LSF array 1123950[1-45], 45/45 done 0 fail; 5 agent seeds + 10 placebo draws per
base; `scripts/rq1_sweep_task.py` with `ACTION_MODE="tilt"`, `MAX_TILT=0.15`; cached folds in
`outputs/rq1_sweep_tilt/`, summary.json + skill_net_by_base.png):** H1 holds for the expressive
tilt agent too — it adds NO learned skill above the structural base on real ETFs, on every base,
and the loss is LARGER than the scalar gate's. skill_net (agent mean − placebo mean), all CIs
entirely below 0, placebo-exceedance = 1.00 for all three:

| base | gate agent / net (Plan 2) | tilt agent / net (Plan 3) |
|---|---|---|
| equal_weight | −8.6e-5 / −1.76e-4 | **−1.56e-4 / −2.06e-4** |
| vol_scaled   | −7.1e-5 / −1.19e-4 | **−1.26e-4 / −1.75e-4** |
| risk_parity  | −6.0e-5 / −1.01e-4 | **−1.03e-4 / −1.54e-4** |

Every agent seed is negative on every base; the placebo (luck) floor is POSITIVE everywhere
(~+5e-5), so netting stays essential; all 10 surrogates beat the agent for every base.

**Interpretation — closes the loop.** The tilt agent's real-data loss is ~1.5–1.8× the gate's
on each base. This is the **over-churn mechanism confirmed on real data**: giving the agent more
expressiveness (per-asset tilt vs. a scalar gate) lets it churn more, and on real markets — where
there is no reliably timeable structure — that extra churn is pure cost drag, not edge. Same
ordering as the gate: loss is largest for the weakest base (equal_weight → more tilt activation →
more turnover) and smallest for risk_parity. Trustworthy because the synthetic tilt RQ2 proved the
SAME measure detects skill net-of-null when timeable structure exists. **The expressive agent
STRENGTHENS H1, it does not rescue the RL allocator:** capability without timeable structure loses.

## 2026-07-19 — Tilt RQ2 judged NET-OF-NULL; the tilt agent over-churns signal-off (supersedes the entry below)

**Finding (robust, LSF, 5–8 seeds):** the expressive residual-tilt agent CANNOT reach a
clean do-nothing floor. With NO signal it still tilts and LOSES to costs:
`skill_off ≈ −6e-5 to −8e-5` (not ~0), and more training makes it worse (it overfits the
observation features to noise). This is the milder echo of the original unbounded-N-dim-tilt
failure — the tanh bound + `max_tilt` cap reduced but did not eliminate the over-churn.
`max_tilt=0.20` over-churns harder (`skill_off=−8.5e-5`) than `0.15` (`≈−7e-5`); **0.15 is the
validated cap** (the earlier "clean −9.5e-6 at 3 seeds" was a lucky draw — see below).

**Decision (user-approved): judge the tilt RQ2 gate NET-OF-NULL**, consistent with the
real-data placebo-net-of-null method already used for RQ1. Two criteria:
1. `skill_net = skill_on − skill_off > 5e-5` — signal-on skill NET of the signal-off churn
   null is clearly positive (skill appears only when timeable structure exists);
2. `skill_off < 5e-5` (one-sided) — signal-off must not manufacture POSITIVE skill; a
   NEGATIVE skill_off is fine, it is the null that gets netted out.
The old two-sided `|skill_off| < 5e-5` criterion was correct for the SCALAR GATE (whose
do-nothing floor is free at exactly 0) but wrong for the tilt model (which cannot reach it).
This is a methodology alignment, not threshold-weakening (thresholds/floor unchanged; Rule 9).

**Validated (LSF, max_tilt=0.15, 5 seeds, 150k):** skill_off=−6.4e-5, skill_on=+1.40e-4,
**skill_net=+2.04e-4 = 4.09× the 5e-5 floor**; `pass_net=True`, `pass_off_not_positive=True`.
Robust (net-of-null cancels common-mode seed variance). The tilt skill measure is VALIDATED:
skill appears net-of-null only when the signal exists. Tooling: `scripts/rq2_tilt_parallel.py`
(+ `rq2_tilt.sh`) runs this on LSF in ~4 min (parallel across seeds×strengths).

**Implication for real-data RQ1 (tilt):** report skill NET of the placebo null (as already
designed) — the null absorbs the tilt agent's over-churn, so the verdict stays interpretable.

## 2026-07-19 — Tilt RQ2 calibration: max_tilt=0.15 passes on multi-regime market

**SUPERSEDED by the entry above** — the single 3-seed `skill_off=-9.5e-6` below was a lucky
draw; the robust (multi-seed) signal-off skill is ~−6 to −8e-5, so the gate is judged
net-of-null, not two-sided. Kept for the record.

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
