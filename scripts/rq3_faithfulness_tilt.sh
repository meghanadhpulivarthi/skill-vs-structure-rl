#!/bin/bash
set -euo pipefail

# LSF CPU job: the RQ3 faithfulness experiment for the EXPRESSIVE TILT agent. Trains
# tilt agents on the multi-regime market (signal on), probes the safe-block weight
# (directional de-risking) with the causal + attribution tracks, and writes the
# faithfulness verdict to outputs/<ts>_rq3-faithfulness-tilt/. Heavier than the gate
# probe (121-dim obs, 5 assets, KernelSHAP over 121 features). Runnable from anywhere.

# Config — edit these directly
THREADS_PER_WORKER=4
N_SEEDS=5

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Batch environments are minimal — ensure uv (in ~/.local/bin) is found.
export PATH="$HOME/.local/bin:$PATH"
# Cap threads so torch does not oversubscribe the LSF slot allocation.
export OMP_NUM_THREADS="$THREADS_PER_WORKER"
export MKL_NUM_THREADS="$THREADS_PER_WORKER"
export PYTHONPATH="$REPO_ROOT"

echo "rq3_faithfulness_tilt: host=$(hostname) repo=$REPO_ROOT n_seeds=$N_SEEDS start=$(date '+%Y-%m-%d %H:%M:%S')"
uv run python -u -c "from src.rq3_faithfulness import run_tilt_experiment, TILT_CONFIG; run_tilt_experiment(TILT_CONFIG, $N_SEEDS)"
echo "rq3_faithfulness_tilt: done=$(date '+%Y-%m-%d %H:%M:%S')"
