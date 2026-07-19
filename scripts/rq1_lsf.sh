#!/bin/bash
set -euo pipefail

# LSF batch job for the real RQ1 experiment (CPU, parallel across 6 walk-forward
# runs). Runnable from any working directory: resolves the repo root from this
# script's location. Submit with bsub (see the header comment for the command).

# Config — edit these directly
THREADS_PER_WORKER=4

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Batch environments are minimal — ensure uv (installed in ~/.local/bin) is found.
export PATH="$HOME/.local/bin:$PATH"

# Cap BLAS/torch threads per worker so the 6 concurrent runs share the node's
# cores cleanly; propagates to spawned worker processes via the environment.
export OMP_NUM_THREADS="$THREADS_PER_WORKER"
export MKL_NUM_THREADS="$THREADS_PER_WORKER"
# Repo root on the path so `from src...` and the spawned `scripts.rq1_parallel`
# workers both import correctly.
export PYTHONPATH="$REPO_ROOT"

echo "rq1_lsf: host=$(hostname) repo=$REPO_ROOT start=$(date '+%Y-%m-%d %H:%M:%S')"
uv run python -u -m scripts.rq1_parallel
echo "rq1_lsf: done=$(date '+%Y-%m-%d %H:%M:%S')"
