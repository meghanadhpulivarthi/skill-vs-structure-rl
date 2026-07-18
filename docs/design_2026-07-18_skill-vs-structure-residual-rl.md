# Design Spec — Skill vs. Inherited Structure in RL Portfolio Allocation

**Status:** Design (approved in brainstorming, pre-plan).
**Date:** 2026-07-18.
**Supersedes:** `research_problem.md` (the original Lavko/Klein/Walther-audit
framing). That framing was narrowed/scooped during lit review; this document is
the current problem. See `context/` for the trail:
`lit-scoop-check.md` → `deep-research-findings.md` → `lit-review-full-reads.md`
→ `novelty-check-n1-n2.md` → `novelty-check-mc1.md`.
**Note:** working dir is not a git repo, so this spec was NOT committed. If git is
initialized later, commit this file as the first design artifact.

---

## 1. One-sentence problem

When a Sharpe/log-reward RL allocator appears to reduce out-of-sample left-tail
risk, how much of that is *genuine learned skill* versus *inherited structure*
(the long-only simplex, log-reward geometry, and the classical rebalancing
premium) — and do standard explanations identify the real mechanism?

## 2. Why this is worthy (and not niche)

- **A live contradiction anchors it.** Lavko/Klein/Walther (2023) credit a
  Sharpe-reward RL allocator with left-tail-risk reduction; Kruthof & Müller
  (2025), testing a Sharpe-reward SAC across 7 datasets / ~300 yrs, find *no*
  consistent tail advantage vs 1/N and total collapse under costs; Che Zheng
  (2026) and Benhamou (2021) *do* see drawdown reduction and informally attribute
  it to de-risking / volatility-regime-timing. Nobody has resolved *when* the
  benefit exists or *tested* its mechanism.
- **It targets the subfield's own credibility anxiety.** RL-in-finance surveys
  (e.g. FinRL 2025) name weak evaluation, missing baselines, and poor
  interpretability as open problems. "How much RL 'skill' is inherited structure?"
  is that anxiety made concrete and testable.
- **The contribution is a reusable method, not a one-paper audit** (see §4).

## 3. Novelty positioning (honest)

- The residual-policy technique (base policy + learned correction) is **borrowed**
  from robotics Residual Policy Learning (Silver 2018; Johannink 2019). We do not
  claim to invent it.
- The "it's geometry not skill" mechanism is the **classical rebalancing premium /
  volatility pumping / Stochastic Portfolio Theory (Fernholz)** — cited, not
  reinvented. Its economic analog in finance is enhanced indexing / smart-beta
  "benchmark + active tilt."
- **The contribution is the skill-accounting methodology:** a theoretically-
  grounded *structural-null base*, *structure-baselined credit assignment*, and
  *synthetic ground-truth validation* of the resulting skill measure — plus (MC2)
  a *causal* mechanism probe. This combination, and its purpose (isolating and
  validating learned skill in RL allocators), is unoccupied.
- **Scoop reads required before writing positioning:** Pollok & Robik 2026
  (arXiv 2607.00475); "DRL in Factor Investment" (arXiv 2509.16206); confirm no
  existing "SPT/growth-optimal base + RL residual" work in finance.

## 4. Research questions & hypotheses

- **RQ1 (skill isolation).** Trained as a residual on a structural-null base that
  already harvests the rebalancing premium, does the RL agent add any OOS
  tail/risk benefit beyond the base, after costs?
  - *H1:* Most of the tail benefit is captured by the base; the learned residual
    adds little net of costs.
- **RQ2 (ground truth).** In synthetic markets with no timeable structure, does
  the residual's benefit vanish (as genuine skill must), and reappear when a
  timeable signal is injected?
  - *H2:* Yes — the residual measures skill only when skill is possible. A
    persistent benefit in structure-free markets would expose it as artifact.
- **RQ3 (mechanism, MC2).** Do causal interventions (vol shocks, regime flips,
  feature freezes) show the agent's de-risking is driven by what post-hoc
  attribution (SHAP/saliency) claims?
  - *H3:* Standard attribution misidentifies the mechanism; the causal probe does
    not.

Every hypothesis is falsifiable in either direction; a "residual adds ~nothing"
result is a strong, publishable outcome, not a failure.

## 5. Method — MC1 (core contribution)

**5.1 Structural-null base policy.** A transparent, non-learned base spanning
inherited structure, built as a small ladder so the skill floor is *strong*:
1. constant-rebalanced 1/N (harvests the baseline rebalancing premium);
2. volatility-scaled 1/N (adds the vol-targeting de-risking channel);
3. an SPT / diversity-weighted (growth-optimal-oriented) portfolio.
Decision to lock in planning: start with (1)+(2); add (3) if (1)+(2) leave an
obvious structural gap. Rationale: a weak base inflates apparent skill.

**5.2 Residual policy (PPO) — DE-RISKING GATE parameterization.** *(Revised
2026-07-18 after an unconstrained N-dim tilt failed to learn — see below.)* The
agent outputs a scalar **de-risking gate** `g_t ∈ [0,1]` and the executed
portfolio is a blend of the base and a safe allocation:
`w_t = (1 - g_t)·base_t + g_t·safe`. `g_t = 0` reproduces the base exactly (zero
skill by construction); the agent only raises `g_t` when timed de-risking
genuinely pays. Reward is the structure-baselined log-return (§5.3). PPO chosen
for on-policy stability, standard use in the literature, and SB3 support.

*Why this, not a free tilt:* the original design was an unconstrained N-dim tilt
`w = project_simplex(base + a)`. Empirically PPO over-tilted (mean|action|≈0.7),
paid turnover every step, and scored **negative** skill even against the weakest
base with a near-perfect signal (150k steps, VecNormalize) — it never learned the
"do nothing" floor. The scalar gate makes parsimony structural: the floor is free,
and the action space matches the scientific object (learned de-risking timing).
Prototype (minimal risky+safe world, 120k steps): signal-on skill **+8.4e-5**
(gate 0.13), signal-off skill **≈0** (gate 0.00) — the RQ2 shape holds. The
transparent surrogate for RQ1 becomes a signal/volatility-threshold gate rule.

**5.3 Structure-baselined credit assignment (the methodological teeth).** The
advantage is netted against the base policy's realized log-growth, so the policy
gradient only "sees" skill above the structural floor. This distinguishes the
method from generic residual RL and defines the operational skill measure.

## 6. Validation — synthetic ground truth (minimal regime-switching)

A calm/crisis regime-switching return generator with a **toggleable predictable
signal** — a state variable that genuinely forecasts the regime switch.
- Train the full pipeline (base + residual PPO) with signal **ON** and **OFF**.
- **Skill measure is validated iff** the residual's OOS benefit → 0 with signal
  OFF and recovers with signal ON. This is the ground-truth calibration that turns
  the method from a demonstration into a validated measurement.
- Kept deliberately minimal (regime-switching + toggle); richer stochastic-vol /
  factor structure is out of scope unless the minimal design proves insufficient.

## 7. MC2 — causal mechanism probe (second contribution, STAGED)

On the trained agent, apply systematic environment interventions — inject a
volatility shock, flip the regime, freeze/permute the volatility feature — and
measure the *causal* change in de-risking behavior. Compare against what
SHAP/saliency attribute over the same decisions.
- Deliverable: a reusable causal-probing protocol for RL allocators and a verdict
  on whether post-hoc attribution is causally faithful in this domain (the
  Atrey/Lu critique, never tested for portfolios).
- **Staging (timeline mitigation):** §5–6 (MC1) is a complete paper on its own.
  MC2 is built only after MC1 lands, so a timeline slip degrades gracefully to
  MC1-only rather than sinking the project.

## 8. Real-data demonstration & evaluation

- **Universe:** multi-asset liquid ETFs — SPDR sector ETFs + US Treasuries (e.g.
  IEF/TLT) + gold (GLD) + a couple of international equity ETFs; ~10–20 tradable
  instruments. Daily, ~2005–2025, free via `yfinance`, survivorship-clean, spans
  2008 / 2020 / 2022 stress.
- **Protocol:** anchored walk-forward (no full-sample tuning); turnover-aware
  transaction costs ON from day one.
- **Tail/risk metrics (matched to the literature):** 99% expected shortfall /
  CVaR, maximum drawdown, tail ratio, skewness; plus Sharpe / Sortino / turnover.
- **Baselines:** the structural ladder (§5.1), 1/N, minimum-variance, and
  RMZ/KP-style tail-minimizing strategies (as in Kruthof & Müller).

## 9. Success criteria

A **validated skill-accounting method** that (a) provably isolates learned skill
on synthetic ground truth (RQ2), and (b) delivers a clean verdict on real ETFs:
how much of the tail benefit is skill vs inherited structure (RQ1). MC2 adds (c) a
causal-faithfulness verdict on standard explanations (RQ3). Success is NOT "beat
the market"; a rigorous "residual adds ~nothing above structure" result clears the
bar.

## 10. Locked scope decisions (from brainstorming)

| Decision | Choice |
|---|---|
| Asset universe | Multi-asset liquid ETFs (yfinance, ~2005–2025) |
| Primary RL algorithm | PPO (Stable-Baselines3) |
| Methodological scope | MC1 + MC2, staged (MC1 first, MC2 as extension) |
| Synthetic markets | Minimal regime-switching + toggleable predictable signal |

## 11. Risks & feasibility (4 months, solo, GPUs available)

- **MC1+MC2 in 4 months solo** → mitigated by staging (§7); MC1 is the shippable core.
- **Residual PPO instability on the simplex** → simplex projection + entropy
  control + the structure-baselined advantage (which reduces gradient variance).
- **Weak structural null inflates apparent skill** → the base ladder (§5.1),
  strengthened before trusting any positive skill result.
- **Scoop risk** → close the §3 reads before writing the paper's positioning.
- **Synthetic-market realism** → intentionally minimal; escalate only if the
  ground-truth test is inconclusive.

## 12. Out of scope (YAGNI)

Multi-agent RL; alpha/return-maximization framing; point-in-time single-name
equity constituents; richer stochastic-vol/factor synthetic markets; algorithm zoo
(SAC/TD3) beyond PPO unless a robustness check is cheap late in the project.
