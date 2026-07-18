# Lit Scoop Check — Is RQ3 novel?

**Date:** 2026-07-18
**Purpose:** Go/no-go on the framing claim that RQ3 (mechanism attribution of RL
tail-risk reduction) is the real, unscooped contribution, and that RQ1/RQ2
(costs, 1/N) are well-trodden scaffolding.
**Method:** Scoped manual web search (Option 2). Not a full deep-research pass.

---

## Verdict

- **RQ1 (costs kill RL edge) — CONFIRMED as received wisdom / freshly scooped.**
- **RQ2 (can't beat 1/N) — CONFIRMED as received wisdom / freshly scooped.**
- **RQ3 (is tail-risk reduction emergent skill or a transparent vol-scaling
  heuristic?) — OPEN, but surrounded by adjacent work that must be cited and
  differentiated.** The specific move — *attribution-by-replication* (show a
  transparent vol-scaling rule reproduces most of the tail benefit) — does not
  appear to have been done.

## Key evidence

1. **"Can deep reinforcement learning beat 1/N"** (ScienceDirect,
   S154461232500131X, 2025). Snippet: *"SAC's high turnover leads to negative
   net returns under modest transaction costs."* Directly does costs + turnover
   + 1/N. => RQ1/RQ2 are not novel as standalone contributions.
   NOTE: full text 403'd — snippet only. Tail-risk coverage unconfirmed.

2. **"Explainable Post hoc Portfolio Management Financial Policy of a DRL
   agent"** (de-la-Rica-Escudero et al., PLoS ONE 2025; arXiv 2407.14486).
   Closest paper to RQ3. Does PPO + SHAP + LIME + feature importance for
   *generic* interpretability ("do the agent's actions follow an investment
   policy"). Does NOT target tail risk specifically. Does NOT test a transparent
   heuristic surrogate. => attribution-by-feature-importance, NOT
   attribution-by-replication. RQ3's angle is distinct.

3. **DRL for volatility targeting** (HAL-03202431, 2021). Builds vol targeting
   INTO the reward. Opposite direction from RQ3, which asks whether vol-scaling
   *emerges* and can be stripped out to reproduce the benefit.

4. **"Multi-dimensional Attribution Analysis of DRL"** (ACM 3801228.3801310,
   2026). Paywalled (403). Likely feature-attribution, not behavioral surrogate.
   FLAG: re-check before finalizing novelty claim.

## Reframing implied by these findings

RQ3's novelty is NOT "explain the RL agent" (SHAP/LIME papers already exist).
It is the **behavioral-surrogate / mechanism-reduction test**: a transparent,
near-parameter-free vol-scaling rule reproduces most of the tail-risk benefit,
i.e. attribution-by-replication rather than attribution-by-feature-importance.
RQ1/RQ2 should be framed as "cost-auditing *this specific anchor paper* (Lavko
et al.), which no one has done," not as general novelty.

## Limits / open flags

- Could not read full text of the "beat 1/N" paper (403) or the ACM attribution
  paper (403). Both should be obtained before the paper's lit-review section.
- Escalation to full deep-research NOT triggered: scoop check came back clear
  enough (RQ3 open) rather than ambiguous.
