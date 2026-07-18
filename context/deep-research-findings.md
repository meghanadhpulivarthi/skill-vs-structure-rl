# Deep Research Findings — Mechanism/Explainability Gap in RL Allocation

**Date:** 2026-07-18
**Method:** deep-research workflow (run wf_5b990ee6-822). 5 angles, 22 sources
fetched, 103 claims extracted, 25 adversarially verified (2/3-refute kill rule),
21 confirmed. Raw output: /tmp/.../tasks/w3bp7o0ur.output (ephemeral — re-run to
regenerate).

---

## Bottom line

The **attribution-by-replication gap is genuinely OPEN** (high confidence, 3-0):
no 2019-2026 paper tests whether an RL allocator's *downside/left-tail* reduction
can be reproduced by transparent heuristics (vol targeting, risk parity, ERC,
de-risking/cash rules). Papers either (a) omit risk-heuristic baselines (compare
only to 1/N + buy-and-hold), (b) engineer the tail benefit into an asymmetric
reward (mean-CVaR, Sortino) so there's no emergent skill to reduce, or (c)
attribute only MEAN-return performance to beta-vs-alpha, not tail behavior.

## Confirmed findings

1. **Gap #1 open** (high). Xue & Ye 2025, de-la-Rica-Escudero 2025, Jung & Oh
   2025 all lack transparent-risk-heuristic baselines on the tail dimension.
2. **Symmetric-reward / asymmetric-outcome tension is LIVE** (high). Agents
   trained on Sharpe/variance routinely reported on Sortino/5% CVaR/tail
   ratio/max-drawdown. Even AlphaPortfolio (Sharpe reward) credited with
   skewness 1.42-1.91 and low max drawdown.
3. **Mechanism reduction already demonstrated on the RETURN dimension** (high) —
   see SCOOP FLAG below.
4. **XAI is a dead end for mechanism attribution** (high). SHAP/LIME/saliency on
   RL allocators reveal WHICH features drive weights, never WHETHER the policy is
   behaviorally equivalent to a known heuristic. Post-hoc saliency is non-causal,
   "exploratory not explanatory" (Atrey 2020 ICLR; Lu 2024; Springer XRL review
   2023) — BUT that evidence is from Atari/control, NOT portfolio (must re-test
   in-domain, not cite-and-assume).
5. **Cost/turnover robustness gap partially closed** (high). Jung & Oh charge
   10bps, track turnover; RL routinely fails to cleanly beat 1/N; reward-driven
   differences often statistically fragile (no FDR-surviving p-values in crypto).
   => RQ1/RQ2 are confirmatory, not novel.

## SCOOP FLAG (highest priority)

**Pollok & Robik 2026, "End-to-End Parametric Portfolio Policies for Cross-Asset
Futures Timing: When Do AI Models Beat Simple Rules?" (arXiv 2607.00475).**
Closest existing work. Already does: Newey-West HAC factor decomposition
attributing a transformer Sharpe-policy's return to market beta (0.63*, alpha
n.s.); shows it "matches rather than beats 1/N"; benchmarks vs 1/N, risk parity
(inverse-vol), TSMOM under realistic costs + walk-forward; concludes the policy
learns "a steady, sub-unit exposure to risk" = disguised vol-scaling.
**Differentiation required:** they do this on MEAN-RETURN alpha/beta, on
cross-asset FUTURES. The TAIL/DOWNSIDE reduction (CVaR, max drawdown, tail ratio)
is untouched. A defensible project must target the tail dimension + behavioral
policy-matching (not just factor regression) + an equities universe (LKW class).

## Supporting adjacent evidence

- Jang & Seong (DDPG): attributes 27% max-drawdown reduction to explicit
  correlation/diversification mechanism, NOT emergent skill. (Supports thesis.)
- 800-TD3-agent study: only BARRA systematic-risk features gave downside
  protection (max drawdown -0.71%, p=0.02) — small isolated effect, not general.
- Regime-CVaR allocator: 226% turnover, net-negative after costs. (Cost point.)
- FinRL benchmarking survey (2504.02281): field has "no widely accepted
  benchmark"; names weak evaluation, poor interpretability, overengineering as
  open concerns — supports rigor/mechanism (not alpha) positioning.

## Caveats / honesty flags

- Several key sources are 2026-dated (Pollok & Robik; Yu & Chang) or late-2025
  (Jung & Oh) — near/after knowledge cutoff. Existence corroborated across
  indexes, but fine-grained numbers partly abstract-verified. Re-read full PDFs
  before the paper's lit-review section.
- MDPI 403'd direct fetches (snippet/EBSCO-verified only).
- Anchor paper (LKW 2023) was NOT re-examined in this corpus; its status as
  "the falsifiable target" is inferred from the surrounding literature's failure
  to close the gap.

## Refuted (did NOT survive 2/3 verification) — do not rely on

- "XAI reveals decisions driven by specific price levels (Apple close #1)" (1-2).
- "Saliency explanations are unfalsifiable/subjective" strong form (1-2).
- "Only interpretable-by-design agents are faithful" strong form (1-2).
- "Yu & Chang charge only small (unrealistic) costs" (1-2).

See [[lit-scoop-check.md]] for the earlier scoped pass this builds on.
