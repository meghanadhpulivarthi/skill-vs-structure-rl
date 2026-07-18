# Full-Text Reads — 5 Papers in lit_review/

**Date:** 2026-07-18. All read cover-to-cover. Builds on
[[deep-research-findings]] and [[lit-scoop-check]].

---

## 1. de-la-Rica-Escudero et al. 2024/25 — "Explainable Post hoc DRL" (2407.14486)
- PPO, 5 tech stocks (AAPL/V/BABA/ADBE/SNE), OHCL features, 2015-17 train / 17-18 trade.
- Post-hoc SHAP + LIME + feature-importance at TRADING time (their novelty vs 4 prior
  papers that explain at training time).
- Delivers WHICH features drive weights (AAPL close #1). That's the ceiling.
- NO costs, NO turnover, NO walk-forward, NO 1/N or heuristic benchmark, NO tail analysis.
- Explicitly: network params "not interpretable"; heuristic hybridization = future work.
- **Takeaway:** XAI answers "which feature," never "is the policy ≡ a known heuristic."
  RQ3's behavioral-replication angle is unoccupied.

## 2. Kruthof & Müller 2025 — "Can DRL beat 1/N" (ssrn-5130133) — MOST CONSEQUENTIAL
- SAC, **Sharpe reward**, 7 Ken French datasets (DeMiguel tradition), ~300 yrs OOS,
  anchored walk-forward w/ transfer learning. State: close + 100d covariance + 26 TА.
- Benchmarks: 1/N, Min-Var, RMZ, KP (RMZ/KP = second-order-stochastic-dominance,
  tail-risk-MINIMIZING strategies — i.e. transparent tail baselines).
- Frictionless: SAC Sharpe 0.45 vs 1/N 0.43 (NOT significant). Loses badly on anomaly (D7).
- **@ 0.1% cost: SAC −4.7% excess / −0.45 Sharpe vs 1/N +4.9% / 0.42. Turnover 40%/day
  (SAC) vs 0.40% (1/N). SAC SIGNIFICANTLY UNDERPERFORMS 1/N after costs.**
- **TAIL: SAC 99% ES −0.033 ≈ 1/N −0.034; skew SAC −0.60 vs 1/N −0.54 (worse).
  "Distribution-based and tail-risk measures do not reveal a consistent advantage for SAC."**
- Uses a timing-vs-asset-preference decomposition (constant-weights, Jacobs 2014).
- **Takeaway:** DIRECTLY CONTRADICTS Lavko/Klein/Walther's tail-reduction headline for a
  Sharpe-reward RL allocator. RQ1/RQ2 are now definitively scooped. Also: RQ3's *premise*
  ("RL reduces tail risk") is contested — must be established, not assumed.

## 3. Jung & Oh 2025 — "Factor-based DRL (FDRL)" (pone.0332779)
- PPO/SAC/TD3 × 5 rewards (Sharpe, Sortino, Static-β, Dynamic-β, Momentum-β).
  β = rolling-regression factor sensitivities embedded in BOTH state and reward.
- Equity/crypto/macro/multi, 2015-25, 70/30 split, **10 bps costs**, turnover measured.
- Strong stats: HAC, Wilcoxon, jackknife Sharpe, moving-block bootstrap, FDR.
- **Reward-driven gains are statistically FRAGILE: crypto MBB never <0.05; only macro
  Dynamic-β robust; equal-weight rivals learners in calm regimes.**
- Tail-aware benefit (Sortino/Static-β) is ENGINEERED into reward, not emergent.
- **Takeaway:** confirms confirmatory nature of RQ1/RQ2; statistical fragility is real;
  tail-aware-reward papers are NOT counterevidence to a mechanism-reduction hypothesis
  (they build the tail benefit in).

## 4. Benhamou et al. 2021 — "Distinguish the indistinguishable" (DTU vol-targeting) — KEY FOR RQ3
- DRL allocates AMONG 9 volatility-targeting models (bond future). Model-free selects
  model-based. Anchored walk-forward + T-test vs benchmark + feature-sensitivity XAI —
  ALL the "honest protocol" pieces in one paper.
- Feature sensitivity: HAR returns/vol + TYVIX vol dominate → agent keys off volatility.
- **U-shape (Fig 13): agent picks LOWEST or HIGHEST vol-estimate model → learned
  regime-timing of volatility exposure. "simultaneously increases net return and
  decreases max drawdown → capacity to detect regime changes."**
- **Takeaway:** closest existing demonstration that DRL drawdown reduction = learned
  vol-regime-timing. BUT it BUILDS ON vol-targeting as substrate; it does not REDUCE a
  raw asset-allocator's tail benefit TO a transparent vol rule. Method source:
  feature-sensitivity, walk-forward, statistical-difference test.

## 5. Che Zheng 2026 — "Multi-dimensional Attribution Analysis of DRL" (3801228.3801310)
- PPO/SAC × MLP/LSTM × feature/reward/freq = 32 configs; attribution = component ablation.
- US tech + safe-haven (AAPL/GOOG/MSFT/Gold/SP500/Cash/US_debt), real costs, rolling val.
- Composite reward (log-ret + left-side trade reward + holding reward − downside penalty
  − cost). Best: PPO-LSTM, Sharpe 1.41, MDD −21.9% vs equal-weight −27.1%.
- **Qualitative mechanism: "drawdown resilience stems from precise risk-asset
  management — during 2022 downturn, PPO-LSTM reduced exposure to high-beta tech,
  shifting to gold and cash."** = de-risking, exactly RQ3's posited mechanism...
- ...but attribution is to COMPONENTS (which of network/feature/reward/freq), NOT to a
  transparent heuristic, and the de-risking is OBSERVED, never replication-tested.
- **Takeaway:** independent qualitative sighting of the de-risking mechanism; still nobody
  runs the replication test. Also a partial scoop on "component attribution" — differentiate
  by doing HEURISTIC-EQUIVALENCE attribution on the TAIL, not component ablation on Sharpe.

---

## Cross-cutting conclusions

1. **RQ1/RQ2 dead as contributions** (Kruthof & Müller + Jung & Oh close them).
2. **A real, live CONTRADICTION now anchors the project:** LKW 2023 (RL reduces left tail)
   vs Kruthof & Müller 2025 (no tail advantage for Sharpe-SAC), while Che Zheng 2026 and
   Benhamou 2021 DO see drawdown reduction and attribute it (informally) to de-risking /
   vol-timing. Nobody has resolved WHEN the tail benefit appears or TESTED its mechanism.
3. **RQ3's specific method — attribution-by-replication of the TAIL benefit to a transparent
   vol-scaling / de-risking / RMZ-KP surrogate — is genuinely unoccupied.** Multiple papers
   circle it (Benhamou builds on vol-targeting; Che Zheng observes de-risking; K&M give the
   tail baselines) but none run the test.
4. **Method scaffolding already exists to borrow:** K&M's tail metrics (ES/EVaR/RLVaR) +
   RMZ/KP transparent tail strategies + timing decomposition; Benhamou's feature-sensitivity
   + walk-forward + statistical-difference test.

## Honesty flags
- Token cost of these full image-PDF reads far exceeded the CLAUDE.md per-task budget;
  done because the user explicitly asked to fully read them.
- Several papers are 2025/2026-dated (near/after knowledge cutoff); treat fine-grained
  numbers as read-from-PDF (reliable here since PDFs are in hand) but cite carefully.
