#!/bin/bash
set -euo pipefail

# LSF CPU job: the RQ3 / MC2 causal-faithfulness experiment. Trains N_SEEDS gate
# agents on the synthetic risky+safe market (signal on), probes each with the
# causal track (feature-group ablation) and the attribution track (saliency +
# KernelSHAP), and writes the faithfulness verdict. Heavy PPO training must run
# on LSF, not the login node. Runnable from any working directory.

# Config — edit these directly
THREADS_PER_WORKER=4

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Batch environments are minimal — ensure uv (in ~/.local/bin) is found.
export PATH="$HOME/.local/bin:$PATH"
# Cap threads so torch does not oversubscribe the LSF slot allocation.
export OMP_NUM_THREADS="$THREADS_PER_WORKER"
export MKL_NUM_THREADS="$THREADS_PER_WORKER"
export PYTHONPATH="$REPO_ROOT"

echo "rq3_faithfulness: host=$(hostname) repo=$REPO_ROOT start=$(date '+%Y-%m-%d %H:%M:%S')"
uv run python -u -m src.rq3_faithfulness
echo "rq3_faithfulness: done=$(date '+%Y-%m-%d %H:%M:%S')"
