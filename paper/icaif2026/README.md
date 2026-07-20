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

## Structure: main body + extended appendix
`paper.tex` contains BOTH:
- the **main body** (Intro → Conclusion + references), written to stand alone; and
- an **extended-version appendix** (`\appendix` onward) with the full RQ1 risk-metric
  discussion, the scale/cardinality confound fix, the activity diagnostics, the
  observation layout, and reproducibility detail.

**⚠️ ICAIF forbids appendices/supplementary and caps at 8 pages INCLUDING references.**
So the appendix is for an **extended (arXiv) version only**. For the ICAIF submission:
delete everything from the `\appendix` line to `\end{document}`, then confirm the main
matter is ≤ 8 pages. The main body does not depend on the appendix.

## Files
- **`paper_icaif.tex` — the ICAIF submission cut.** Main body only, no appendix,
  no cross-references into an appendix. This is the file to submit. Estimated ~5
  pages compiled (I could not compile here; see below), i.e. safely under the
  8-page cap with headroom to expand if you want a fuller submission.
- **`paper.tex` — the extended (arXiv) version.** Identical body plus an appendix
  with the full risk-metric discussion, the scale/cardinality confound fix, the
  activity diagnostics, the observation layout, and reproducibility. Use for arXiv,
  NOT for ICAIF (ICAIF forbids appendices).
- `references.bib` — 17 verified references (shared by both), only solidly-verified
  entries kept.

Both `.tex` files pass the same static checks (env/brace balance; 17 citations, 0
undefined, 0 orphan).

## Compiling
There is **no LaTeX toolchain on this machine** (no `pdflatex`, no `tectonic`, no TeX
module), and the sandbox blocked me from downloading a TeX engine — so I could **not**
compile it here. Static checks only: environment/brace balance and citation integrity
(17 keys, 0 undefined, 0 orphan) all pass. Three ways to build:

**1. Overleaf (easiest, no install):** new project → upload `paper.tex` +
`references.bib` → compiler pdfLaTeX. `acmart` is built in; it builds as-is.

**2. Locally, self-contained via `tectonic`** — run this yourself (in-session, prefix
with `!`, or in a terminal); it downloads only what it needs and needs no root:
```
cd /tmp && curl -fsSL \
  https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-x86_64-unknown-linux-musl.tar.gz \
  | tar xz
cd /dccstor/meghanadhp/projects/Helix/rl-allocation-audit/paper/icaif2026
/tmp/tectonic paper.tex        # emits paper.pdf (runs bibtex internally)
```
If you'd rather I run the compile, add a Bash permission rule allowing the download,
and I'll build it and report the page count.

**3. TeX Live / MacTeX (if installed):**
```
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
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
