# Skill vs. Structure in RL Portfolio Allocation

When a reinforcement-learning portfolio allocator *appears* to reduce
out-of-sample left-tail risk, **how much of that is genuine learned skill, and
how much is structure it inherited for free** — the long-only simplex, the
log-reward geometry, and the classical rebalancing premium?

This repository develops and validates a **skill-isolating method** to answer
that, and applies it as a lens on a live contradiction in the literature (some
papers credit Sharpe-reward RL allocators with tail-risk reduction; others find
no such advantage once costs and a strong `1/N` benchmark are in play).

> **Status:** all planned contributions complete. **Plans 1 (synthetic RQ2), 2 (real-data
> RQ1), 3 (expressive-tilt agent), and 4 (causal mechanism probing, RQ3) are done** — the
> skill measure is validated, applied to real ETFs, stress-tested against a far more capable
> agent, and the trained agent's decision mechanism is causally probed against post-hoc
> attribution. See [`docs/`](docs/) and [`context/`](context/) for the full design and
> decision trail.

---

## The idea

Standard RL-allocation studies compare a learned agent to weak baselines and
report risk-adjusted gains. But an agent can look skillful while doing nothing an
transparent rule couldn't: the **rebalancing premium** (Fernholz Stochastic
Portfolio Theory) and log-reward geometry deliver downside protection *mechanically*,
with no forecasting. To separate learned skill from this inherited structure, we:

1. **Fix a structural-null base policy** that already harvests the free structure
   (equal-weight / volatility-scaled).
2. **Train the agent as a residual on that base** — specifically a scalar
   **de-risking gate** `g ∈ [0, 1]`, with executed weights
   `w = (1 − g)·base + g·safe`. `g = 0` reproduces the base exactly, so the
   *zero-skill floor is free* and the agent only moves `g` when timed de-risking
   genuinely pays.
3. **Reward only skill above the base** — the reward is *structure-baselined*:
   the agent's net log-return **minus the base's** net log-return on the same
   market path (both charged their own turnover). The gradient never sees the
   structural growth, only what the agent adds.
4. **Validate the skill measure on synthetic ground truth** — a regime-switching
   market with a *toggleable* leading signal. When the signal is on, timed
   de-risking is possible; when off, the market is unforecastable and genuine
   skill is impossible *by construction*.

> **On novelty (stated plainly):** residual policy learning is a borrowed
> technique (from robotics), and the rebalancing premium is classical finance.
> The contribution is the *purpose and validation*: a structural-null base +
> structure-baselined credit assignment + a synthetic ground-truth test that
> calibrates the skill measure. The relevant prior work is cited, not reinvented.

> **"Did you just cripple the agent?"** A scalar gate is deliberately parsimonious, so
> a fair reader asks whether a *more expressive* agent would have found skill the gate
> couldn't. We answered this directly (Plan 3): a **bounded per-asset residual tilt**,
> `w = project_simplex(base + max_tilt·tanh(a))` with enriched causal features
> (dual-horizon volatility, momentum, base weights). Zero action still reproduces the
> base. This agent *can* express arbitrary long-only skill — and the verdict below holds
> for it too, which is the point.

## Key result (RQ2)

The skill measure vanishes when skill is impossible and grows with how forecastable
the market is. Mean structure-baselined skill on held-out markets (5 seeds, 150k
training steps):

| Signal strength (timeable structure) | Mean skill (held-out) |
| --- | --- |
| 0.00 (unforecastable — the null) | 5.9 × 10⁻⁵ |
| 0.50 | 1.7 × 10⁻⁴ |
| 0.95 (strongly forecastable) | 2.9 × 10⁻⁴ |

Skill rises monotonically with signal strength, and the on-vs-off separation is
clear — the method detects learned skill only when learned skill is possible.

> **Honest caveat** ([`context/open-questions.md`](context/open-questions.md)):
> at higher training budgets the signal-off floor is *noisy and slightly positive*
> — PPO can extract small **spurious** skill from pure noise. That floor is the
> overfitting/luck baseline, so the rigorous statistic is skill **net of the
> matched signal-off null**, with confidence intervals over more seeds. This is
> carried into the real-data phase.

## Key result (RQ1 — real data)

Applied to real ETFs (11 liquid instruments, 2005–2024, anchored walk-forward, 10 bps
costs), the learned de-risking gate adds **no skill above the structural base** — the
verdict the method was built to reach honestly. Skill is reported **net of a
phase-randomization placebo null** (the real-data analog of signal-off), with a CI.
Robustness sweep over the full base ladder (5 agent seeds + 10 placebo draws each):

| Structural base | Agent skill (5 seeds) | Placebo (luck) mean | Skill net of null | 95% CI | Placebo exceedance |
| --- | --- | --- | --- | --- | --- |
| equal-weight | −8.6 × 10⁻⁵ | +9.0 × 10⁻⁵ | −1.76 × 10⁻⁴ | [−2.9, −0.7] × 10⁻⁴ | 1.00 |
| vol-scaled  | −7.1 × 10⁻⁵ | +4.8 × 10⁻⁵ | −1.19 × 10⁻⁴ | [−2.1, −0.3] × 10⁻⁴ | 1.00 |
| risk-parity | −6.0 × 10⁻⁵ | +4.1 × 10⁻⁵ | −1.01 × 10⁻⁴ | [−1.4, −0.6] × 10⁻⁴ | 1.00 |

Every skill-net CI lies **entirely below zero**, and **every** structureless surrogate
beat the real agent (`exceedance = 1.00`) — for all three bases. The agent's optimal
learned behavior is essentially *do nothing* (mean gate ≈ 0.04, staying on the base).
On the same out-of-sample window the agent tracks its base on all tail/risk metrics
(ES, drawdown, Sharpe, Sortino, skew), and both clearly beat naive 1/N — so the tail
benefit over 1/N is **inherited structure, not learned skill**. The placebo mean being
*positive* everywhere is the overfitting floor made visible, and is exactly why netting
against the null is essential.

Together with RQ2 this closes the loop: the method **detects** skill when it exists
(synthetic) and finds **none** on real markets net of luck — robustly across bases and seeds.

## Key result (expressive agent — the fairness check)

Giving the agent a real shot at skill *sharpens* the verdict rather than softening it.
The expressive per-asset tilt passes the **same synthetic validity gate** as the gate —
judged *net of the signal-off null* (the tilt agent cannot reach a costless do-nothing
floor; with no signal it over-churns and loses ~−6 × 10⁻⁵, so we require skill **net of
that churn null** to be clearly positive): `skill_off = −6.4 × 10⁻⁵`,
`skill_on = +1.4 × 10⁻⁴`, **`skill_net = +2.0 × 10⁻⁴`** (4× the floor). The measure still
detects skill only when timeable structure exists — for the capable agent too.

On real ETFs, the same robustness sweep (5 agent seeds + 10 placebo draws per base) shows
the expressive agent adds **no skill above structure — and loses *more* than the gate**:

| Structural base | Gate skill net of null | **Tilt skill net of null** | Placebo exceedance |
| --- | --- | --- | --- |
| equal-weight | −1.76 × 10⁻⁴ | **−2.06 × 10⁻⁴** | 1.00 |
| vol-scaled  | −1.19 × 10⁻⁴ | **−1.75 × 10⁻⁴** | 1.00 |
| risk-parity | −1.01 × 10⁻⁴ | **−1.54 × 10⁻⁴** | 1.00 |

The tilt agent loses ~1.5–1.8× what the gate does, in the same base-ordering (worst for the
weakest base). This is the **over-churn mechanism confirmed on real data**: more
expressiveness means more churn, and where no timeable structure exists that churn is pure
cost drag, not edge. A capable agent doesn't rescue the RL allocator — it demonstrates,
more forcefully, that skill above structure isn't there to be found.

## Key result (RQ3 — is the explanation faithful?)

A separate worry: even when an RL agent *does* have a real mechanism, do standard post-hoc
explanations (SHAP, gradient saliency) correctly identify it? We probe the trained gate agent on
synthetic ground truth — where the leading `signal` feature is, by construction, the only genuine
driver — and compare what *causal interventions* (per-feature ablation) show against what attribution
claims. Across 5 seeds (`signal_strength = 0.95`):

| | Identifies the true (`signal`) driver | Per-feature agreement with causal (Spearman) |
| --- | --- | --- |
| Causal ground truth | 100% of seeds (the premise) | — |
| KernelSHAP | 100% of seeds | +0.61 ± 0.31 |
| Gradient saliency | 80% of seeds | +0.79 ± 0.27 |

So in this clean, single-mechanism setting **attribution is largely faithful** — it recovers the true
driver and correlates positively with causal effect. The strong worry ("attribution misidentifies the
mechanism") is *not* supported here; this is a **boundary condition** on that critique — it says *when*
attribution can be trusted for RL allocators. The failure it does show is coherent, not random: the one
saliency miss is the seed whose agent relies *least* exclusively on the signal (causal signal-share 0.54
vs ~1.0), i.e. reliability frays as the mechanism becomes more distributed. SHAP and saliency also trade
off — SHAP is more reliable at naming the single top driver, saliency tracks the full per-feature profile
more closely.

> **Why this verdict is trustworthy.** The measure was corrected for a subtle confound before use: the
> `signal` feature has ~35–100× the scale of the return/volatility features, and raw gradient saliency is
> scale-invariant while SHAP and ablation are scale-sensitive. Left unfixed, the causal-vs-attribution
> comparison would have been dominated by units, not mechanism — and could have *falsely* read as
> "attribution is unfaithful." Saliency is reported in gradient×std units and all methods share one
> normalized per-feature aggregation, validated on known-answer, feature-swap, and distributed-driver
> calibration tests.

## Repository layout

```
src/
  simplex.py            # Euclidean projection onto the probability simplex
  metrics.py            # ES/CVaR, max drawdown, tail ratio, skew, Sharpe, Sortino, turnover
  synthetic_market.py   # regime-switching markets + toggleable signal (risky+safe & multi-regime worlds)
  base_policies.py      # structural-null bases: equal-weight, vol-scaled, risk-parity (ERC)
  allocation_env.py     # Gymnasium env: de-risking-gate OR bounded per-asset tilt action + structure-baselined reward
  train.py              # PPO training entrypoint (traceable, seeded)
  validate_skill.py     # RQ2 experiment: skill vs. signal strength (synthetic ground truth)
  data.py               # real ETF panel via yfinance, cached (RQ1)
  real_market.py        # returns -> env market dict with a causal (no-lookahead) crisis signal
  baselines.py          # 1/N, minimum-variance, CVaR-min baselines + cost-aware roller
  walk_forward.py       # anchored expanding-window walk-forward, per-fold retrain, OOS stitching
  placebo.py            # phase-randomization null (real-data overfitting/luck baseline)
  rq1_real_data.py      # RQ1 experiment: skill net of null + metrics table + figures
  interventions.py      # RQ3 causal track: replay, feature-group ablation, env interventions
  attribution.py        # RQ3 attribution track: gradient saliency + KernelSHAP (grad x std units)
  rq3_faithfulness.py   # RQ3 experiment: causal vs attribution faithfulness verdict
scripts/                # LSF drivers: parallel RQ1 run + robustness sweep (array job) + RQ3 experiment
tests/                  # unit + ground-truth tests (incl. the RQ2 validity gate)
docs/                   # design spec + implementation plans
context/                # decision trail: scoping, novelty checks, decisions, open questions
```

## Setup & run

Uses [`uv`](https://docs.astral.sh/uv/) (not pip):

```bash
uv sync                                   # install from the lockfile
uv run pytest -q                          # run the test suite

# Run entrypoints as modules (absolute imports):
uv run python -m src.train                # train one gate agent (default run)
uv run python -m src.validate_skill       # full RQ2 experiment -> outputs/<timestamp>_skill-validation/
uv run python -m src.rq1_real_data        # full RQ1 real-data experiment -> outputs/<timestamp>_rq1-real-data/
```

The real-data experiments are compute-heavy (many independent PPO trainings). On an
IBM Spectrum LSF / CCC cluster they run as CPU jobs — `scripts/rq1_lsf.sh` (one parallel
run) and `scripts/rq1_sweep_task.sh` (a `bsub` array over bases × seeds × placebo draws),
aggregated by `scripts/rq1_sweep_aggregate.py`. GPU is intentionally unused: the policy is
a small MLP, so the bottleneck is environment stepping, not matrix math.

Every run writes a timestamped folder under `outputs/` with `config.json`,
`results.json`, and (for the experiment) a `skill_vs_signal.png` figure.

## Roadmap

- **Plan 1 — synthetic ground-truth validation (RQ2): complete.** The skill measure is
  calibrated on regime-switching markets with a toggleable signal.
- **Plan 2 — real-data verdict (RQ1): complete.** Liquid multi-asset ETFs, anchored
  walk-forward, 10 bps costs, the base ladder (equal-weight / vol-scaled / risk-parity ERC),
  and literature baselines (`1/N`, minimum-variance, CVaR-min). Skill reported *net of a
  phase-randomization placebo null* with CIs; verdict is learned skill ≈ 0, robust across
  bases and seeds.
- **Plan 3 — expressive-tilt fairness check: complete.** A bounded per-asset residual
  tilt with enriched causal features, validated on synthetic ground truth (net-of-null)
  and swept on real ETFs. Verdict holds: the capable agent adds no skill above structure
  and over-churns *more* than the gate on real markets.
- **Plan 4 — causal mechanism probing (RQ3): complete.** Interventions (feature-group
  freeze/permute ablation, volatility shocks, signal flips) on the trained gate agent vs.
  post-hoc attribution (KernelSHAP, gradient saliency), adjudicated against the known
  `signal` driver. Verdict: in this clean setting attribution is largely faithful — a
  boundary condition on the attribution-faithfulness critique.
- **Deferred / future.** Re-run the faithfulness probe where attribution is likelier to
  fray (the expressive tilt agent; the real-data agent — less concentrated mechanisms), and
  a write-up pulling MC1 + real RQ1 + RQ3 together.

## Notes

- Source-literature PDFs are intentionally **not** in this repo (copyright); the
  citations live in [`docs/`](docs/) and [`context/`](context/).
- This is exploratory research code optimized for clarity and traceability, not a
  production library.
