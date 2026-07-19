# Scoop Reads — closed before writing Plan 2 (2026-07-19)

Spec §3 required these reads before committing the paper's positioning. All
three are now read; **none scoops the MC1 contribution** (structural-null base +
de-risking gate + structure-baselined reward + synthetic ground-truth calibration,
applied as a skill-vs-structure verdict on real ETFs).

## Pollok & Robik 2026 — "End-to-End Parametric Portfolio Policies for Cross-Asset Futures / When Do AI Models Beat Simple Rules?" (arXiv 2607.00475)
- End-to-end transformer/LSTM mapping state → weights (signed-softmax), on 16 CME
  futures, 2000–2024, expanding-window walk-forward. Reward = differentiable
  Sharpe (cost-aware variant subtracts turnover). Baselines: 1/N, risk parity, TSMOM.
- Anchor-question overlap: explicitly asks "when does an AI policy beat simple
  rules by enough to justify complexity?" and finds — via ex-post α/β regression on
  1/N — that transformer equity performance is "attributed to market exposure
  rather than residual alpha" ("closer to a risk-efficient directional exposure").
- Does NOT: train as a residual on a base; use a structure-baselined (agent−base)
  reward; validate on synthetic ground truth; use SPT / rebalancing-premium base.
- **Verdict: SUPPORTING, not scooping.** Independently corroborates the
  skill-vs-structure suspicion in equities, but with post-hoc attribution, not a
  learned residual with a validated skill measure. Cite as strong motivation and
  contrast (we bake structure-baselining into training + calibrate on ground truth,
  vs. their after-the-fact regression).

## CAFPO 2025 — "Deep Reinforcement Learning in Factor Investment" (arXiv 2509.16206)
- Conditional auto-encoded latent factors (94 firm characteristics) → PPO/DDPG →
  long-short weights on US equities 2000–2020. SHAP for factor attribution.
  Baselines (EW, VW, Markowitz, vanilla-DRL, FF-DRL) used for comparison only.
- End-to-end; no residual/base, no structure-baselined reward, no skill-vs-structure
  isolation, no synthetic ground truth, no SPT/growth-optimal base.
- **Verdict: distinct.** Single-name equity factor RL; orthogonal to our multi-asset
  structure-accounting method.

## IJCAI 2025 (Chen et al.) — "Enhancing Portfolio Optimization via Heuristic-Guided Inverse RL with Multi-Objective Reward and Graph-based Policy Learning"
- INVERSE RL: learns a reward from an interpretable expert-strategy generator
  (sector diversification + correlation constraints), heterogeneous graph policy
  with hierarchical attention, multi-objective reward. Real market data only.
- Cites SPT (Fernholz & Karatzas 2005) only as ONE example heuristic in related
  work — NOT as a base the agent residualizes on.
- Does NOT: train a residual on an SPT/structural-null base; use an agent−base
  structure-baselined reward; validate a skill measure on synthetic ground truth.
- **Verdict: distinct.** Different problem (recover a reward from expert demos),
  different mechanism (graph IRL), no ground-truth skill calibration.

## Bottom line
The "structural-null base + RL residual with structure-baselined credit +
synthetic ground-truth skill calibration" combination remains unoccupied. Proceed
to Plan 2. Add Pollok & Robik as a primary motivating/contrast citation.
