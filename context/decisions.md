# Decisions

Choices made and why. Newest first.

## 2026-07-20 — RQ3 TILT-AGENT VERDICT: attribution stays faithful per-feature; SHAP over-credits the salient signal; group-premise is cardinality-confounded

**Result (LSF job 1130878, 5 tilt agents @ signal 0.95 on multi_regime; safe-block-weight object;
`src/rq3_faithfulness.py::run_tilt_experiment`; outputs/2026-07-20_05-22-16_rq3-faithfulness-tilt/).**

**Premise-as-coded FAILED but for an illuminating reason (cardinality, not mechanism).**
`signal_is_causal_driver_fraction = 0.0` — the `signal` is never the top causal GROUP. BUT the
signal is a SINGLE feature with causal group-share 0.32–0.44 (all seeds), while the `returns`
group's larger total (0.54–0.63) is spread across 100 features (~0.006 each — near the ~1/121
noise floor). Per-feature, the signal is the dominant driver by ~54× (0.344 vs 0.0063 avg return,
seed 0). So the tilt agent DOES key on the signal; the group-level "signal top" premise fails only
because 100 return features out-SUM one signal feature. **Methodological lesson:** the group-level
top-driver premise gate is confounded by extreme group-cardinality imbalance (1 vs 100); the
**per-feature Spearman is the robust, cardinality-free faithfulness measure** and should be the
headline for any wildly-unequal-cardinality obs. (This is the same normalization-doesn't-cancel-
cardinality property flagged in the gate final review, now biting the premise gate itself.)

**Faithfulness (per-feature Spearman, causal vs attribution):** saliency **+0.999 ± 0.0004**,
SHAP **+0.88 ± 0.009** (freeze≈permute; near-zero std across seeds). Both strongly faithful;
saliency essentially perfect. **SHAP over-attributes to the salient signal feature:** it ranks
`signal` the top group in 4/5 seeds (share 0.48 vs causal 0.34, seed 0) and deflates `returns`
(0.51 vs causal 0.63), whereas saliency matches causal's top group (`returns`) in 5/5 and its full
ranking almost exactly. So relative to the causal ground truth, **saliency is faithful; SHAP is
less faithful and biased toward the high-variance, semantically-salient signal feature.**

**INVERSION vs the gate result.** Gate (2026-07-20): SHAP was the RELIABLE one (named signal top
5/5), saliency missed once; Spearman saliency 0.79 / SHAP 0.61. Tilt: saliency near-perfect
(0.999), SHAP lower (0.88) and salience-biased. **Which attribution method is faithful is
AGENT-DEPENDENT.** The recurring SHAP failure mode is over-crediting a salient high-variance
single feature; saliency (local grad×std) tracks the actual per-feature sensitivity more robustly.
Notably the tilt per-feature Spearmans are HIGHER than the gate's — attribution is not less
faithful for the more expressive agent at the per-feature level; what frays is SHAP's group-level
top-driver call under salience + cardinality pressure.

**Verdict on H3 (tilt):** attribution remains broadly faithful to the causal mechanism for the
capable tilt agent (per-feature), EXTENDING the gate's boundary-condition rather than confirming
the strong "attribution misidentifies the mechanism" hypothesis — with the important qualifier
that SHAP's *single-top-driver* identification can mislead via salience bias, and that group-level
verdicts are unreliable under extreme cardinality imbalance. Report per-feature faithfulness.

**Caveats / notes.** (a) `base_weights` group is inert here by construction: under the equal_weight
base its obs block is a constant 0.2 vector (std≈0 → ~0 importance) — correct, not a finding.
(b) Per-feature causal/attribution VECTORS were not persisted (only group shares); the per-feature
dominance is inferred from group shares + cardinality. A paper re-run should save the 121-dim
vectors to state the per-feature max directly. (c) Clip vs pre-clip caveat carries as before
(saliency uses the pre-clip mean; causal/SHAP the projected safe-weight) — inert here (~3.9e-5).

## 2026-07-20 — RQ3/MC2 VERDICT: in this clean setting, post-hoc attribution is LARGELY FAITHFUL

**Result (LSF job 1130750, 5 gate agents @ signal_strength=0.95 on the risky+safe market;
`src/rq3_faithfulness.py`; outputs/2026-07-20_04-30-45_rq3-faithfulness/):**

- **Premise (causal ground truth): 100%.** In all 5 seeds the `signal` feature is the top
  CAUSAL driver of de-risking (per-feature freeze/permute ablation) — the agents genuinely
  time off the signal, so faithfulness IS adjudicable.
- **SHAP identifies the true driver in 5/5 seeds** (signal-group share 0.77–1.00, matching the
  causal share). **Saliency in 4/5.** The one saliency miss is seed 0 — which is ALSO the seed
  where the agent's mechanism is least concentrated (causal signal share 0.54 vs ~1.0 for the
  other four). So attribution's reliability degrades exactly where the true mechanism is more
  distributed — a coherent, non-random failure mode.
- **Per-feature Spearman agreement (causal vs attribution), mean±std over seeds:** saliency
  +0.79±0.27, SHAP +0.61±0.31 (freeze≈permute). Both positive; notable INVERSION — saliency
  tracks the full per-feature causal profile better, while SHAP is the more reliable at naming
  the single top driver. The two "standard" methods are not interchangeable.

**Verdict on H3:** the strong hypothesis ("standard attribution MISidentifies the mechanism")
is **not supported in this controlled, single-mechanism, low-dimensional setting** — attribution
is largely faithful here. Per the spec (§5), this is an explicitly legitimate, publishable
BOUNDARY-CONDITION result: it bounds the Atrey/Lu attribution-faithfulness worry rather than
confirming it, and says *when* attribution can be trusted for RL allocators (clean, concentrated
mechanism) and where it frays (distributed mechanism → seed-0; high across-seed variance ~0.3).

**Trustworthiness hinges on the confound fix.** The final whole-branch review caught that the
`signal` feature has ~35–100× the input scale of the return/vol features, and that saliency
(raw |∂g/∂o|, scale-invariant) was in different units from freeze/SHAP (scale-sensitive), plus a
40:1 group-cardinality tilt from summed aggregation. Fixed: saliency→grad×std (matched units),
all four methods→one normalized per-feature aggregation basis. WITHOUT this fix the saliency-vs-
causal comparison would have been dominated by units and we could have FALSELY concluded
"attribution is unfaithful." The verdict is only interpretable because the measure was corrected
and validated on known-answer + feature-swap + distributed-driver calibration tests (Rule 9).

**Caveat carried in results.json:** saliency reads the pre-clip Gaussian mean while causal/SHAP
read the clipped behavioral gate; where the gate is decisive a saliency-vs-causal divergence is
not necessarily unfaithfulness. **Future work (deferred):** re-run the probe on the harder cases
where attribution is more likely to fray — the expressive TILT agent (per-asset, multi-regime)
and/or the real-data agent — where the mechanism is less clean/concentrated than this gate.

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
