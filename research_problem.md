# Research Problem Report — Project A

## Does reinforcement-learning portfolio allocation survive an honest protocol, and where does its tail-risk reduction actually come from?

**Type:** Replication + adversarial stress test + explainability
**Anchor paper:** Lavko, Klein & Walther (2023), *Reinforcement Learning and Portfolio
Allocation: Challenging Traditional Allocation Methods?*, QMS Working Paper 2023/01.
**Status:** Problem statement (pre-plan). Written 2026-07-12.

---

### 1. One-sentence problem

Model-free RL allocators reportedly beat mean-variance optimization and reduce
out-of-sample left-tail risk — but the anchor paper never charges them
transaction costs, never cleanly beats naive 1/N, and only *promises*
explainability; this project asks whether the edge is real under an honest
protocol and, if so, what mechanism actually produces it.

### 2. Why this is a worthy problem (not a cookie-cutter project)

The single most common ML-in-finance project is "I trained a deep RL agent that
beats the market." It is also the one experienced quants trust least, because it
is almost always contaminated by lookahead bias, no transaction costs, or
overfitting. This project inverts that failure mode: it points ML rigor at
**honesty and mechanism**, not at alpha.

That inversion is exactly what every source in the reference set points to.
Lavko et al. explicitly frame their own work toward *explainable AI* ("explain
how and why RL-generated portfolios perform better") but deliver the explanation
as a promissory note. Halperin, in interview, gives the operating recipe
directly: *select a popular, implementable model, implement it, then find all
the problems with it that were not mentioned in the paper, and deliver.* This
project is that recipe applied to a paper whose own authors left the hardest
questions open.

It also satisfies Bianco's interview test. The questions he probes — *how did
you know your answer was correct? what went wrong? how would you improve it?* —
have real answers here regardless of outcome, because the project is built
around a falsifiable question rather than a target result.

### 3. What the anchor paper actually claims

Lavko, Klein & Walther test **model-free RL agents** (Q-learning with an
experience replay buffer; policy-optimization methods) against traditional
allocation on S&P 500 and European (Bloomberg 500) constituents, clustered into
factor datasets via a Carhart four-factor model.

- **State / action / reward:** state is the price vector; actions are long-only
  portfolio weights on the simplex (wᵢ ≥ 0, Σwᵢ = 1); the reward is the
  **Sharpe ratio**.
- **Trained on price-derived features only** — no firm-specific or macro
  variables — deliberately, to mirror the DeMiguel et al. (2009) benchmark study.
- **Benchmarks:** equal-weight (1/N), and a mean-variance family: global minimum
  variance (GMV), equal risk contribution (ERC), and factor-constrained
  portfolios.
- **Headline findings:**
  1. RL beats mean-variance on risk-adjusted return and probabilistic Sharpe
     ("robust evidence").
  2. RL portfolios are *more stable* — claimed to capture nonlinearity in the
     stochastic discount factor.
  3. Outperformance is **not universal versus 1/N**: RL wins on some factor
     datasets and loses on a small selection of subsamples.
  4. RL **reduces out-of-sample left-tail risk** — the finding they flag as most
     relevant to practitioners.

### 4. The soft spots — "problems not mentioned in the paper"

These are the cracks the project pries open. Each is a concrete, defensible gap.

1. **Transaction costs are absent from the core setup.** RL allocators rebalance
   frequently. A Sharpe-reward agent has no incentive to limit turnover. A
   realistic cost model could erase the mean-variance advantage entirely — this
   is the first thing a practitioner would check and the paper does not report it.
2. **1/N is never cleanly beaten.** This is the DeMiguel (2009) ghost: naive
   diversification is famously hard to beat out of sample. "Beats mean-variance
   but not 1/N" is a weaker claim than it first sounds, and deserves a direct,
   regime-by-regime accounting.
3. **The tail-risk finding contradicts the reward.** The agent optimizes Sharpe
   (a symmetric, variance-based objective) yet is credited with *asymmetric*
   left-tail reduction. Is that a genuine emergent property, or an artifact of
   de-risking / cash-holding in volatile regimes? The paper asserts the benefit
   but does not attribute it.
4. **Explainability is promised, not delivered.** They cite the explainable-AI
   literature and state an aim to "explain how and why," but the mechanism behind
   rebalancing decisions is never opened up.

### 5. Research questions and hypotheses

- **RQ1 (survival):** Under realistic transaction costs and turnover accounting,
  does the RL allocator's risk-adjusted advantage over mean-variance survive?
  - *H1:* A non-trivial share of the reported edge is turnover-driven and shrinks
    materially once costs are charged.
- **RQ2 (the real benchmark):** Across market regimes, does RL beat **1/N** after
  costs — and where does it fail?
  - *H2:* RL does not systematically beat 1/N after costs; any edge is
    regime-dependent (concentrated in high-dispersion / high-volatility periods).
- **RQ3 (mechanism):** Is the left-tail-risk reduction an emergent asymmetric
  skill, or explained by a simpler volatility-timing / de-risking heuristic?
  - *H3:* Much of the tail benefit is reproducible by a transparent
    volatility-scaling rule, implying the RL agent has *learned to de-risk*, not
    to allocate better.

Note that every hypothesis is stated as something that could be **falsified in
my favor or against me** — that is the point. A confirmed edge is a positive
result; a turnover-eaten edge is an equally publishable, equally interview-ready
negative result in the DeMiguel tradition.

### 6. Data

- **Primary:** liquid, survivorship-bias-aware equity/ETF universes at daily
  frequency (e.g. sector/factor ETFs, or point-in-time index constituents).
  Freely obtainable (`yfinance` / public APIs).
- **Rationale:** the anchor paper uses index constituents clustered by factor;
  daily liquid instruments keep transaction-cost modeling honest and keep the
  4-month scope feasible.
- **Regime coverage:** the test window must span at least one crisis
  (e.g. 2020 COVID drawdown, 2022 drawdown) so RQ2/RQ3 have stress to measure.

### 7. Methodology (sketch — full plan is a later artifact)

1. Reproduce a model-free RL allocator (start with one algorithm — e.g. PPO or a
   DQN-style allocator — not a zoo) with a Sharpe reward, matching the paper's
   long-only simplex action space.
2. Implement the honest layer: a turnover-aware transaction-cost model, and a
   **walk-forward / rolling** out-of-sample protocol (no full-sample tuning).
3. Implement the benchmark gauntlet: 1/N, GMV, ERC.
4. Implement the mechanism probe for RQ3: a transparent volatility-scaling
   baseline, plus attribution of the RL agent's rebalancing decisions
   (e.g. weight-change vs. realized-volatility regressions, feature attribution).
5. Report everything regime-by-regime, after costs, with turnover.

### 8. Success criteria (Rule 4 — define success, then iterate)

The project succeeds if it produces a **defensible, honest answer** to RQ1–RQ3,
with:
- a cost/turnover accounting the anchor paper omits;
- a clean regime-by-regime RL-vs-1/N verdict;
- a mechanism attribution for the tail-risk claim (confirmed emergent skill *or*
  reduced to a transparent heuristic).

Success is **not** "beat the market." A rigorous negative result clears the bar.

### 9. Risks and feasibility (4 months, self-built, part-time)

- **Main risk:** RL infrastructure and reward/observation design is the real
  time sink. *Mitigation:* one algorithm, small liquid universe, off-the-shelf RL
  library; treat reward design as the intended "struggle," not scope creep.
- **Overfitting risk:** the very trap this project is meant to expose.
  *Mitigation:* walk-forward protocol fixed **before** any tuning; costs on from
  day one.
- **Derivativeness risk:** could read as "just a replication." *Mitigation:* the
  novelty is the mechanism probe (RQ3) and the honest cost/1-N accounting — the
  parts the authors left undone.

### 10. What it signals for an MFE application

Demonstrates: RL competence *plus* the maturity to distrust it; out-of-sample
discipline; transaction-cost realism; and the explainability instinct Halperin
and Bianco both name as where the field's edge is moving. It is the anti-thesis
of the overfit "trading bot" project — which is precisely why it stands out.

### 11. Open questions to resolve before the implementation plan

- Which single RL algorithm to commit to first?
- Exact universe and date range (drives the cost model and regime coverage).
- Point-in-time constituents vs. ETF proxies (survivorship handling).
