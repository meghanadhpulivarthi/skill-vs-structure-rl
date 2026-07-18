# Novelty Check — MC1 (structure-controlled / residual RL for skill isolation)

**Date:** 2026-07-18. Scoped manual search. Builds on [[novelty-check-n1-n2]],
[[lit-review-full-reads]].

**Idea under test:** Train the allocator to output a RESIDUAL/tilt on top of an
explicit structure-harvesting base policy (rebalancing-premium / vol-scaled 1/N /
SPT master portfolio), so the learned component isolates skill above the
structural floor; validate on synthetic ground truth.

---

## Verdict: GO, but novelty is "borrowed technique + new purpose," not a new technique

- **Residual Policy Learning (RPL) is a KNOWN, off-the-shelf RL technique** —
  base controller + learned correction. Established in robotics/control: Silver
  et al. 2018 ("Residual RL for robot control"), Johannink et al. 2019, and a
  large 2024-2026 follow-on line (TRANSIC, R-VLA, vessel motion comp, etc.).
  => We are NOT inventing residual RL. A reviewer will know this immediately.

- **In portfolio/trading, the residual-on-structural-base-for-skill-isolation
  framing appears UNOCCUPIED.** No hit for RL trained as a tilt on a
  rebalancing-premium/SPT base to measure marginal skill. Closest neighbors:
  - Pollok & Robik 2026 (end-to-end parametric policies) — NOT residual.
  - "DRL in Factor Investment" (arXiv 2509.16206) — links DRL to factor models,
    not residual-from-structural-base.
  - Classical finance analog EXISTS and must be cited: enhanced indexing / smart
    beta / "benchmark + active tilt" (CFA smart-beta, AQR "New Paradigm in Active
    Equity" w/ ML). That is the economic ancestor of the residual framing.
  - No hit for "reward = excess over structural baseline to isolate skill" as a
    skill-accounting device.

- **The specific combination is novel:** residual RL + base chosen to SPAN
  inherited structure (rebalancing premium / SPT / constraint geometry) + purpose
  = isolate learned skill + synthetic ground-truth validation of the skill split.

## The honest risk (matches the user's own worry)

Positioned as "we applied residual RL to portfolios," MC1 reads as an
**application paper** (known technique, new domain, measurement purpose) — NOT the
strong methodological contribution the user wants.

To be a genuine METHOD contribution, the novel core must be the **skill-accounting
framework**, with residual RL as an ingredient, specifically:
1. **Structural base = theoretically-grounded null**, not ad hoc — e.g. the
   growth-optimal rebalanced / SPT master portfolio that provably harvests the
   rebalancing premium. (Training the residual against a *derived* structural
   optimum is more than generic RPL.)
2. **Structure-baselined advantage:** reward/advantage netted against the base's
   realized log-growth so the policy gradient only "sees" skill. Framed carefully
   this is a contribution ("structure-baselined credit assignment for allocation").
3. **Ground-truth validation:** define skill operationally as the residual's OOS
   contribution that (a) vanishes on synthetic markets with NO timeable structure
   and (b) recovers injected skill when structure is present. This validation is
   the methodological teeth that separates it from an application.

## Remaining scoop risk to close before finalizing
- Read Pollok & Robik 2026 and "DRL in Factor Investment" (2509.16206) fully for
  positioning; confirm neither frames a structural-null residual.
- Confirm no existing "growth-optimal / SPT base + RL residual" work in finance.

## Net
MC1 is a GO if positioned as a **skill-accounting methodology** (structural-null
base + structure-baselined advantage + synthetic ground-truth validation), NOT as
"residual RL applied to portfolios." Cite RPL (robotics) and enhanced-indexing
(finance) as the two ancestors and differentiate on purpose + validation.
