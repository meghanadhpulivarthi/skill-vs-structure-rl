#!/bin/bash
set -euo pipefail

# LSF CPU job: parallel tilt-model RQ2 calibration. Runnable from any working
# directory (resolves the repo root from this script's location).

# Config — edit these directly
THREADS_PER_WORKER=3

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Batch environments are minimal — ensure uv (in ~/.local/bin) is found.
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS="$THREADS_PER_WORKER"
export MKL_NUM_THREADS="$THREADS_PER_WORKER"
export PYTHONPATH="$REPO_ROOT"

echo "rq2_tilt: host=$(hostname) repo=$REPO_ROOT start=$(date '+%Y-%m-%d %H:%M:%S')"
uv run python -u -m scripts.rq2_tilt_parallel
echo "rq2_tilt: done=$(date '+%Y-%m-%d %H:%M:%S')"
