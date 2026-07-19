# Skill vs. Structure in RL Portfolio Allocation

When a reinforcement-learning portfolio allocator *appears* to reduce
out-of-sample left-tail risk, **how much of that is genuine learned skill, and
how much is structure it inherited for free** — the long-only simplex, the
log-reward geometry, and the classical rebalancing premium?

This repository develops and validates a **skill-isolating method** to answer
that, and applies it as a lens on a live contradiction in the literature (some
papers credit Sharpe-reward RL allocators with tail-risk reduction; others find
no such advantage once costs and a strong `1/N` benchmark are in play).

> **Status:** research in progress. **Plan 1 — synthetic ground-truth validation
> — is complete** (this repo). Plans 2 (real-data verdict) and 3 (causal
> mechanism probing) are scoped but not yet built. See
> [`docs/`](docs/) and [`context/`](context/) for the full design and decision trail.

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

## Repository layout

```
src/
  simplex.py            # Euclidean projection onto the probability simplex
  metrics.py            # ES/CVaR, max drawdown, tail ratio, skew, Sharpe, Sortino, turnover
  synthetic_market.py   # regime-switching markets + toggleable signal (incl. risky+safe world)
  base_policies.py      # structural-null bases (equal-weight, vol-scaled)
  allocation_env.py     # Gymnasium env: de-risking-gate action + structure-baselined reward
  train.py              # PPO training entrypoint (traceable, seeded)
  validate_skill.py     # RQ2 experiment: skill vs. signal strength
tests/                  # unit + ground-truth tests (incl. the RQ2 validity gate)
docs/                   # design spec + implementation plan
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
```

Every run writes a timestamped folder under `outputs/` with `config.json`,
`results.json`, and (for the experiment) a `skill_vs_signal.png` figure.

## Roadmap

- **Plan 2 — real-data verdict (RQ1):** liquid multi-asset ETFs, anchored
  walk-forward, realistic transaction costs, an SPT/growth-optimal base added to
  the ladder, and literature baselines (`1/N`, minimum-variance, tail-minimizing
  strategies). Reports skill *net of the null* on real markets.
- **Plan 3 — causal mechanism probing (RQ3):** interventions (volatility shocks,
  regime flips, feature freezes) on the trained agent to test whether standard
  attribution (SHAP/saliency) identifies the true mechanism.

## Notes

- Source-literature PDFs are intentionally **not** in this repo (copyright); the
  citations live in [`docs/`](docs/) and [`context/`](context/).
- This is exploratory research code optimized for clarity and traceability, not a
  production library.
