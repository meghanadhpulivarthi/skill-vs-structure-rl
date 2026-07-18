# Open Questions

Unresolved items to address in later work. Newest first.

## 2026-07-18 — The signal-off "skill" floor is noisy and slightly positive

Full RQ2 experiment (gate, 5 seeds, 150k steps, risky+safe world):
- signal=0.0: mean skill **5.9e-5** (std 1.1e-4)
- signal=0.5: mean skill 1.7e-4 (std 9.8e-5)
- signal=0.95: mean skill 2.9e-4 (std 5.8e-5)

Monotonic in signal strength (good — the RQ2 signature), and on≫off separation
is clear. BUT the signal-OFF floor is not a crisp zero: at higher training budget
some seeds let PPO extract small **spurious** skill from pure noise.

Implications / TODO:
1. This spurious floor IS the overfitting/luck baseline. A rigorous skill claim
   should report skill **net of the matched signal-off null** (skill_on − skill_off),
   not raw skill_on. Consider making this the headline statistic.
2. Report confidence intervals (more seeds, e.g. 10–20) — current std ~1e-4 is
   large relative to the off-mean. The shorter n=3/100k test happened to give a
   cleaner off (~6e-6); the estimate is seed- and budget-sensitive.
3. Investigate whether the spurious floor grows with training budget (overfitting)
   — if so, early-stopping / a fixed budget matched across conditions matters for
   fair comparison.

Not a blocker for Plan 1 (method demonstrated; separation clear), but central to
how RQ1 (real-data) skill is reported in Plan 2 and the paper.
