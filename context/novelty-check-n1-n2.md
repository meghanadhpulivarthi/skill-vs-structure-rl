# Novelty Check — N1 (skill-vs-geometry) + N2 (synthetic ground-truth)

**Date:** 2026-07-18. Scoped manual search (2 rounds), not full deep-research.
Builds on [[lit-review-full-reads]], [[deep-research-findings]].

**Idea under test:** The emergent left-tail-risk reduction credited to Sharpe/
log-reward RL allocators is largely a MECHANICAL artifact of reward geometry
(log-return + long-only simplex), not learned skill — established via synthetic
ground-truth markets where timeable structure can be switched on/off.

---

## Verdict: PARTIALLY PRECEDENTED MECHANISM, NOVEL AS A DEBUNKING BRIDGE

- **The "it's geometry not skill" MECHANISM is NOT new.** It is the well-known
  **rebalancing premium / volatility pumping / diversification return**, formalized
  in **Fernholz Stochastic Portfolio Theory (SPT)**. A long-only portfolio
  rebalanced to fixed weights mechanically harvests a log-growth premium from
  volatility — no forecasting/skill required. Must-cite precedents found:
  - "Demystifying the rebalancing premium" (SSRN 2927791)
  - "The Rebalancing Premium" (INSEAD) — explicitly long-only fixed-weight case
  - Rebalancing premium = difference in LOG performance (SPT framing)
  - Oxford-Man "Deep Learning for Portfolio Optimisation" (2020) — already links
    **functionally generated portfolios (SPT) with deep learning**
  - "Optimal dynamic fixed-mix portfolios based on RL" (S0952197624007577) — RL
    for fixed-mix (constant-rebalanced) portfolios w/ SSD over benchmark
  => A reviewer WILL ask "isn't this just the rebalancing premium?" We must own
     that up front. It is not a mechanism we discovered.

- **Using that mechanism as the NULL to debunk RL's emergent-skill claim IS
  novel.** No paper argues that what the RL-allocation literature celebrates as
  emergent tail-risk skill is substantially the rebalancing premium + log-reward
  geometry, reproducible with NO learning. The SPT+DL work goes the opposite way
  (uses SPT to BUILD models), not to falsify RL skill claims.

- **N2 (synthetic ground-truth falsification separating genuine volatility-timing
  skill from the mechanical rebalancing/geometry effect) appears genuinely
  unoccupied** for portfolio RL. No hits on shuffled/synthetic-data skill-vs-luck
  tests for RL allocators targeting the tail benefit specifically.

## Sharpened thesis (better than the original vague "reward geometry")

> The emergent left-tail-risk reduction attributed to Sharpe/log-reward RL
> allocators is largely the classical **rebalancing premium / SPT log-growth
> effect under a long-only simplex** — reproducible by a constant-rebalanced or
> volatility-scaled rule with matched average exposure, WITHOUT learning. This
> explains the live contradiction: RL "beats" mean-variance (which does not
> harvest the premium) yet cannot beat 1/N (which already does — cf. DeMiguel,
> Kruthof & Müller).

- Concrete transparent surrogates (now literature-grounded, not ad hoc):
  constant-rebalanced 1/N, volatility-scaled/target, functionally-generated
  (SPT) portfolios. Plus a RANDOM/untrained long-only policy as the pure-geometry
  floor.
- Falsifiable null: RL OOS tail reduction is statistically indistinguishable from
  the rebalancing-premium/geometry surrogate with matched exposure.
- N2 makes it causal: in synthetic markets with NO timeable vol structure, genuine
  skill => no tail benefit; mechanical effect => benefit persists. Switch structure
  on to confirm the agent CAN exploit it when it exists.

## Remaining scoop risk to verify before finalizing
- Confirm nobody has explicitly framed RL portfolio outperformance AS the
  rebalancing premium / SPT effect (Oxford-Man 2020 and S0952197624007577 are the
  closest — read both fully before writing the paper's positioning).

## Net
N1+N2 is a GO, with the framing shifted from "reward geometry (novel mechanism)"
to "the known rebalancing premium / SPT effect is the null that explains away the
RL emergent-skill narrative (novel claim + novel synthetic-falsification method)."
Cite SPT / rebalancing-premium literature prominently; do not reinvent it.
