# ICAIF '26 short paper — build & submission notes

**Target venue:** ACM International Conference on AI in Finance (ICAIF '26), Milan,
14–17 Nov 2026.
**Submission deadline:** **2 August 2026, AOE** (no rebuttal period).
**Submission site:** CMT — `cmt3.research.microsoft.com/ICAIF2026/`

## Format constraints (from the CFP)
- ACM `acmart`, **`sigconf` two-column**, **double-blind** (`anonymous`).
- **Hard 8-page limit including figures AND references.** Over-length = desk reject.
- **No supplementary material / appendix.** Everything must fit in the 8 pages.
- In-person presentation required.

The current draft is ~5–6 pages when compiled, so there is headroom to expand
(a figure, fuller related work, a per-metric risk table) if desired — 8 is the max,
not a target.

## Files
- `paper.tex` — the manuscript (`sigconf,anonymous,review`).
- `references.bib` — 17 verified references (only solidly-verified entries kept).

## Compiling
No LaTeX toolchain is installed on this machine, so the paper was **not** compiled
here (static checks only: environment/brace balance and citation integrity all pass).

**Easiest — Overleaf:** create a project, upload `paper.tex` + `references.bib`,
set the compiler to pdfLaTeX. `acmart` is built into Overleaf; it builds as-is.

**Local (if you install TeX Live / MacTeX):**
```
pdflatex paper
bibtex   paper
pdflatex paper
pdflatex paper
```

## Anonymization (double-blind)
- The `anonymous` class option hides the author block automatically and prints
  "Anonymous Author(s)". The real author block (Meghanadh, Independent Researcher,
  meghanadh27@gmail.com) is in `paper.tex` and appears once `anonymous` is removed
  for the **camera-ready**.
- No identifying links are in the text (the repo URL is deliberately omitted;
  "Code and configurations will be released publicly" stands in). Add the real
  repository link only in the camera-ready.

## Before you submit — check these
1. **Verify the `delarica2025` author-name spelling** against the published PLoS ONE
   byline (best-effort transliteration in the `.bib`).
2. Two candidate references from the literature trail were **dropped**, not
   fabricated, because their author bylines could not be verified
   (arXiv:2509.16206; arXiv:2504.02281 FinRL Contests). Re-add with correct authors
   if you want them.
3. Set the real `\setcopyright{...}` / `\acmConference` / DOI block for camera-ready
   (currently neutralized for review).
4. Re-read the numbers against `context/decisions.md` — every figure in the tables
   is transcribed from the recorded experiment verdicts (RQ2/RQ1/RQ3).

## Provenance of the numbers
All results are transcribed from the project's recorded verdicts in
`context/decisions.md`, the README, and the run outputs:
- RQ2 synthetic skill-vs-signal + tilt net-of-null.
- RQ1 real skill net of the placebo null (gate + tilt, three bases).
- RQ3 faithfulness (synthetic gate/tilt + real), per-feature Spearman.
