#!/bin/bash
set -euo pipefail

# LSF CPU job: the RQ3 faithfulness probe for the REAL-DATA gate agent. Trains a gate agent
# on the full real ETF panel (RQ1's risk_parity headline config), probes its de-risking gate
# with the causal + attribution tracks, and writes the METHOD-AGREEMENT verdict plus activity
# diagnostics to outputs/<ts>_rq3-faithfulness-real/. No ground truth on real data, so the
# verdict is method agreement, not faithfulness; a near-inactive-agent null is expected and
# reported honestly (see docs/design_2026-07-20_rq3-real-faithfulness.md). Heaviest of the
# three probes (232-dim obs, 11 assets, KernelSHAP over 232 features). Runnable from anywhere.
# The ETF panel cache (data/etf_panel.parquet) must already exist — compute nodes have no
# internet to re-download from yfinance.

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

echo "rq3_faithfulness_real: host=$(hostname) repo=$REPO_ROOT n_seeds=$N_SEEDS start=$(date '+%Y-%m-%d %H:%M:%S')"
uv run python -u -c "from src.rq3_faithfulness import run_real_experiment, REAL_CONFIG; run_real_experiment(REAL_CONFIG, $N_SEEDS)"
echo "rq3_faithfulness_real: done=$(date '+%Y-%m-%d %H:%M:%S')"
