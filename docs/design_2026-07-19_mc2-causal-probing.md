# MC2 — Causal Mechanism Probing (RQ3): Design Spec

**Date:** 2026-07-19
**Status:** approved design, pre-plan
**Depends on:** MC1 (Plans 1–3) — the trained gate agent and the synthetic
risky+safe market with a toggleable signal. This is the staged second
contribution; MC1 stands alone if this slips.

---

## 1. Question and hypothesis

**RQ3.** Do causal interventions on a trained RL allocator show that its
de-risking is driven by what post-hoc attribution (SHAP / saliency) claims?

**H3.** Standard attribution *misidentifies* the mechanism; a causal probe does
not. Falsifiable in both directions — a clean "attribution is faithful in this
controlled setting" result is as publishable as "it is not." We do not assume
the outcome.

**Deliverable.** A reusable causal-probing protocol for RL allocators, plus a
faithfulness verdict adjudicated against **synthetic ground truth** — the one
setting where the true mechanism is known by construction. This tests the
Atrey/Lu attribution-faithfulness critique in the portfolio domain, where (to
our knowledge) it has not been tested.

## 2. Why synthetic-only, why the gate

- **Synthetic-only testbed.** A faithfulness verdict needs a *known* true
  driver. On the risky+safe market the leading `signal` feature is, by
  construction, the only genuinely predictive input; trailing returns and
  volatility are contemporaneous correlates. Real data has no ground-truth
  driver, so faithfulness cannot be adjudicated there — only method agreement,
  which is a weaker claim. RQ3 therefore lives entirely on synthetic ground
  truth, mirroring how RQ2 used synthetic ground truth to validate the skill
  measure.
- **Scalar gate agent.** The gate outputs a single de-risking scalar
  `g ∈ [0,1]` — the canonical skill object of the whole project. A scalar
  output yields one clean per-feature importance vector, so the causal-vs-
  attribution comparison is unambiguous. (The expressive tilt agent, with a
  per-asset vector output, would force an arbitrary scalar-summary choice; it is
  explicitly out of scope here.)
- **Signal on.** We probe an agent trained at `signal_strength = 0.95`, where
  the gate demonstrably learned to time de-risking off the signal (RQ2). With
  the signal off there is no mechanism to (mis)attribute, so that regime is not
  the probe target (it may appear as a contrast if cheap).

## 3. The observation space (what attribution ranks)

Gate observation on the risky+safe market (`n_assets = 2`, `window = 20`), from
`src/allocation_env.py`:

```
obs = [ win.flatten() | short_vol | signal ]
      [   20 × 2 = 40  |    2     |   1    ]   → 43 features
```

Semantic feature **groups** used throughout the probe:

| group        | indices | meaning                                   | role            |
|--------------|---------|-------------------------------------------|-----------------|
| `returns`    | 0–39    | trailing 20-step returns, both assets     | correlate       |
| `short_vol`  | 40–41   | trailing short-horizon volatility per asset | correlate     |
| `signal`     | 42      | leading crisis indicator                  | **true driver** |

Grouping tames the 40 correlated lagged-return features and makes the verdict
legible. Per-feature (43-dim) vectors are retained for the rank-agreement
statistic; the headline verdict is stated at group level.

## 4. Architecture

Four new source units, each one responsibility, plus tests. No changes to
existing MC1 modules are required (the probe consumes the trained model and the
market dict as-is).

```
src/
  interventions.py       # causal interventions (feature ablation + environment)
  attribution.py         # post-hoc attribution (gradient saliency + KernelSHAP)
  rq3_faithfulness.py    # the RQ3 experiment: train → replay → probe → verdict
tests/
  test_interventions.py
  test_attribution.py
  test_rq3_faithfulness.py
```

New dependency: **`shap`** (`uv add shap`). `torch` is already present via
Stable-Baselines3.

### 4.1 `src/interventions.py`

Pure functions; no training, no I/O.

**Deterministic gate replay.** `gate_of(model, obs_array) -> np.ndarray`:
map a stack of observations to the deterministic gate via
`model.predict(obs, deterministic=True)`, then clip to `[0,1]` exactly as the
env does. Returns one gate value per observation row. This is the behavior all
interventions perturb.

**Feature-group ablation.** For a group `G` (a set of obs indices) and a stack
of observations `O` (shape `[T, 43]`):
- **freeze-at-mean:** replace columns `G` in every row with their column mean
  over `O` — removes that group's *variation*.
- **permute:** replace columns `G` with a row-permutation of themselves (shared
  permutation across the group's columns, drawn from a seeded RNG) — breaks the
  group's temporal alignment with the target while preserving its marginal.

`causal_effect(model, O, group, mode) -> float` returns
`mean_t | gate_of(O)_t − gate_of(O^{(G)})_t |` — the mean absolute change in the
de-risking decision when group `G` is ablated. Larger ⇒ more causally
important. This is a causal test *of attribution's own claim*: attribution
asserts input→output dependence; ablation measures the actual behavior change.

**Environment interventions** (sanity + legibility layer, §7 caveat):
- `inject_vol_shock(market, t0, width, multiplier, seed) -> market`: scale the
  risky asset's returns by `multiplier` over `[t0, t0+width)` (a transient
  variance spike), leaving the safe asset and signal untouched. Returns a new
  market dict (input not mutated).
- `flip_signal(market, t0, value) -> market`: set `signal[t0]` to `value`
  (e.g. force it high) with the rest of the path fixed, to read the gate's
  response to the signal alone.

Both are used to produce **gate-response curves**: gate before vs. after the
intervention, demonstrating the agent is genuinely responsive to the crisis
mechanism (not the verdict, but the evidence that a mechanism exists to probe).

### 4.2 `src/attribution.py`

Post-hoc attribution of the deterministic gate over a stack of observations.
Both methods return a per-feature importance vector (length 43), aggregated to
groups by summing `|importance|` over each group's indices (SHAP values are
additive, so a group's summed magnitude is its attributed contribution;
saliency is aggregated the same way for comparability).

- **Gradient saliency.** `saliency_importance(model, O) -> np.ndarray`:
  the deterministic gate is the mean of the policy's action distribution;
  obtain it as a differentiable function of the observation tensor via
  `model.policy` (`get_distribution(obs).distribution.mean`, pre-clip), then
  `∂g/∂oᵢ` by autograd. Importance = `mean_t |∂g/∂oᵢ|`.
- **KernelSHAP.** `shap_importance(model, O, background, n_samples) -> np.ndarray`:
  wrap the gate as `f(O) -> g` (numpy in/out via `gate_of`), explain a sample of
  rows against a background set (a seeded sub-sample of `O`) using
  `shap.KernelExplainer`. Importance = `mean |SHAP value|` per feature.

Both take a seed and are deterministic given it (SHAP sampling and background
selection are seeded).

### 4.3 `src/rq3_faithfulness.py`

The experiment entrypoint (run as a module, per project convention). Config
block at top of file (traceability rules). Steps:

1. **Train / obtain agents.** Train `N_SEEDS` gate agents on the risky+safe
   market at `signal_strength = 0.95` via the existing `train_agent` / `build_env`
   (reuse, do not reimplement). Self-contained and reproducible; no dependence on
   cached RQ2 artifacts.
2. **Replay.** For each agent, roll out the deterministic policy on a held-out
   eval market (different seed) to collect the observation stack `O` and the
   realized gates.
3. **Causal track.** For each group ∈ {returns, short_vol, signal} and each mode
   ∈ {freeze, permute}: `causal_effect`. Plus environment-intervention response
   curves (vol shock, signal flip) on one representative agent for the figure.
4. **Attribution track.** `saliency_importance` and `shap_importance` over the
   same `O`; aggregate to groups.
5. **Verdict** (§5), aggregated across seeds.
6. **Save** (traceability): timestamped `outputs/<ts>_rq3-faithfulness/` with
   `config.json`, `results.json`, and figures — grouped-importance bars per
   method, the gate-response curves, and the per-seed agreement distribution.

## 5. The faithfulness verdict

Per seed, four importance rankings over the three groups: causal-freeze,
causal-permute, saliency, SHAP.

**Headline — ground-truth anchor.**
1. *Causal ground-truth confirmation:* the `signal` group produces the largest
   `|Δg|` (freeze and permute). This confirms the agent's true mechanism *is*
   the signal — the premise the whole verdict rests on. If this fails (e.g. the
   agent ignored the signal), the run is diagnosed and the training budget /
   signal strength revisited; we do **not** proceed to a faithfulness claim on an
   agent with no signal-driven mechanism.
2. *Attribution faithfulness:* does each attribution method also rank `signal`
   as the dominant group? Report the fraction of seeds where it does, per method.

**Backbone — rank agreement.** Spearman rank correlation between the per-feature
(43-dim) causal-effect vector and each attribution vector, per seed; report mean
± std across seeds, per attribution method, per ablation mode.

**Interpretation.**
- **H3 confirmed** if attribution diverges from causal — e.g. it under-weights
  the signal relative to its causal effect, or spreads importance onto returns/
  vol features that ablation shows are causally inert (low/negative Spearman,
  signal not ranked top by attribution while it is by causal).
- **H3 refuted** if attribution faithfully recovers the causal ranking (signal
  dominant under both, high Spearman). This is a legitimate, publishable result:
  in a clean, low-dimensional, single-mechanism setting, attribution *can* be
  faithful — a boundary condition on the Atrey/Lu worry.

The verdict is reported as measured, with CIs across seeds; no direction is
assumed.

## 6. Testing (Rule 9 — tests encode why, and validate the method itself)

The probe is a measurement instrument; like the RQ2 skill measure, it must be
validated on a known answer before it is trusted on the RL agent.

- **`test_attribution.py` — known-answer calibration.** On a hand-built **linear
  gate policy** `g = σ(w·o)` with known weights `w`, both saliency and KernelSHAP
  must rank features by `|w|` — the top-weighted feature ranked top, an inert
  feature (`wᵢ = 0`) at the bottom. The stub is a small **torch** module exposing
  *both* interfaces the real code uses: a numpy `predict(obs, deterministic=True)`
  (for KernelSHAP and the ablation `gate_of`) and a differentiable forward giving
  the gate mean (for saliency's autograd). If the attribution code cannot recover
  importance on a linear model, it cannot be trusted on the agent. Also:
  importances are finite; group aggregation sums correctly. To keep the two
  attribution functions decoupled from the concrete SB3 type, they depend only on
  this minimal interface (duck-typed), which the stub and the real PPO model both
  satisfy.
- **`test_interventions.py` — ablation semantics + causal ground truth.**
  freeze removes a group's column variance (variance ≈ 0 after); permute
  preserves the group's column marginal (same sorted values); `causal_effect` is
  ≈ 0 for a **constant/inert** feature and strictly > 0 for the `signal` group on
  a policy that depends on the signal. Environment interventions return a new
  dict without mutating the input, and change only the intended slice.
- **`test_rq3_faithfulness.py` — end-to-end ground-truth gate.** On a policy
  known to use *only* the signal (a stub, or a lightly-trained high-signal
  agent), the pipeline runs, causal ablation ranks `signal` top, and the verdict
  object is emitted with all required keys. This is RQ3's own validity gate,
  mirroring the RQ2 test. Thresholds encode the claim and must not be weakened to
  make a flaky run pass (recalibrate budget/strength instead).

Stochastic-training assertions use smoke/property/ground-truth checks (finite,
sign, ranking), not exact-value equality, consistent with the existing suite.

## 7. Scope, caveats, compute

- **Semantic interventions are the softest part** and are a legibility/sanity
  layer, not the verdict. If vol-shock / signal-flip curves prove fiddly to make
  clean, reduce them to a single illustrative figure and keep the headline
  (ablation-vs-attribution + ground-truth anchor). Do not let them expand scope.
- **Compute is light.** Training ~`N_SEEDS` gate agents at 150k steps on the
  2-asset market is minutes each on CPU; probing is forward passes (seconds).
  Runs locally via `uv run python -m src.rq3_faithfulness`. An optional small LSF
  array over seeds follows the existing `scripts/` pattern if parallelism is
  wanted; GPU is intentionally unused (small MLP, env-bound), consistent with
  MC1.
- **Reuse, don't reinvent.** `train_agent`, `build_env`, `AllocationEnv`,
  `generate_risky_safe_market` are consumed unchanged. Attribution/intervention
  code is new and isolated.
- **Out of scope:** real-data probing (no ground truth), the tilt agent
  (vector output), attribution methods beyond saliency + KernelSHAP (integrated
  gradients / DeepSHAP / occlusion were considered and deferred as scope creep).

## 8. Success criteria

- The attribution instrument passes its known-answer calibration (linear policy).
- The causal instrument passes its ground-truth checks (inert ≈ 0, signal > 0).
- The probed agents pass the causal ground-truth confirmation (signal is the
  dominant causal driver) — establishing the premise.
- A faithfulness verdict (signal-dominance per method + Spearman agreement, with
  across-seed CIs) is produced and saved, in whichever direction the data falls.

Consistent with the project's standard: success is a *rigorous, honest verdict*,
not a particular outcome.
