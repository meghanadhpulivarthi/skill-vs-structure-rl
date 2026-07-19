#!/bin/bash
set -euo pipefail

# LSF array-task wrapper for the RQ1 robustness sweep. Each array task runs ONE
# walk-forward (agent seed or placebo draw) selected by $LSB_JOBINDEX. Runnable
# from any working directory; also runnable locally as: rq1_sweep_task.sh <index>.

# Config — edit these directly
THREADS_PER_WORKER=4

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Batch environments are minimal — ensure uv (in ~/.local/bin) is found.
export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS="$THREADS_PER_WORKER"
export MKL_NUM_THREADS="$THREADS_PER_WORKER"
export PYTHONPATH="$REPO_ROOT"

TASK_INDEX="${LSB_JOBINDEX:-${1:-}}"
if [ -z "$TASK_INDEX" ]; then
  echo "error: no task index (set LSB_JOBINDEX via array, or pass as arg 1)" >&2
  exit 1
fi

echo "sweep task idx=$TASK_INDEX host=$(hostname) start=$(date '+%Y-%m-%d %H:%M:%S')"
uv run python -u -m scripts.rq1_sweep_task "$TASK_INDEX"
echo "sweep task idx=$TASK_INDEX done=$(date '+%Y-%m-%d %H:%M:%S')"
