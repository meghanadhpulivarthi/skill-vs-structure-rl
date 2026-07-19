"""One task of the RQ1 robustness sweep.

Runs a single walk-forward (either an agent run for a (base, seed), or a placebo
surrogate run for a (base, draw)) and caches its per-fold results. Designed as an
LSF array task: the 1-based array index selects the task. Reuses the tested
`walk_forward_gate` / `phase_randomize` unchanged — this is orchestration only, so
results are identical to a sequential run. Restartable: cached folds are skipped.

Task index layout (1-based), 3 bases x (N_SEEDS agent + N_PLACEBO placebo):
  within each base block of size PER_BASE = N_SEEDS + N_PLACEBO,
  first N_SEEDS indices are agent seeds 0..N_SEEDS-1,
  the rest are placebo draws 0..N_PLACEBO-1.
"""
import sys
from pathlib import Path

# Config — MUST stay consistent with scripts/rq1_sweep_aggregate.py (it imports these).
BASES = ["equal_weight", "vol_scaled", "risk_parity"]
N_SEEDS = 5
N_PLACEBO = 10
WINDOW = 20
COST_BPS = 10.0
TOTAL_TIMESTEPS = 150_000
INITIAL_TRAIN = 1260
TEST_BLOCK = 252
THREADS = 4
SAFE_TICKER_PREFERENCE = ("IEF", "TLT")   # de-risking sleeve, resolved by NAME
PLACEBO_PHASE_BASE = 1000                 # surrogate phase-randomization seed = base + draw
PLACEBO_TRAIN_BASE = 500                  # placebo PPO training seed = base + draw
RUN_DIR = Path(__file__).resolve().parent.parent / "outputs" / "rq1_sweep"

PER_BASE = N_SEEDS + N_PLACEBO
TOTAL_TASKS = len(BASES) * PER_BASE


def resolve_task(index0: int):
    """Map a 0-based task index to (base_name, kind, idx)."""
    base = BASES[index0 // PER_BASE]
    within = index0 % PER_BASE
    if within < N_SEEDS:
        return base, "agent", within
    return base, "placebo", within - N_SEEDS


def main():
    import torch
    torch.set_num_threads(THREADS)
    from src.data import load_etf_panel
    from src.walk_forward import walk_forward_gate
    from src.placebo import phase_randomize

    if len(sys.argv) < 2:
        raise ValueError("usage: python -m scripts.rq1_sweep_task <1-based task index>")
    index1 = int(sys.argv[1])
    if not 1 <= index1 <= TOTAL_TASKS:
        raise ValueError(f"task index {index1} out of range 1..{TOTAL_TASKS}")
    base, kind, idx = resolve_task(index1 - 1)

    panel = load_etf_panel()
    returns = panel["returns"]
    tickers = panel["tickers"]
    safe_ticker = next((t for t in SAFE_TICKER_PREFERENCE if t in tickers), None)
    if safe_ticker is None:
        raise ValueError(f"no safe sleeve {SAFE_TICKER_PREFERENCE} in panel tickers {tickers}")
    safe_index = tickers.index(safe_ticker)

    base_config = {"base_name": base, "window": WINDOW, "cost_bps": COST_BPS,
                   "safe_asset_index": safe_index, "total_timesteps": TOTAL_TIMESTEPS,
                   "initial_train": INITIAL_TRAIN, "test_block": TEST_BLOCK}

    if kind == "agent":
        run_dir = RUN_DIR / base / f"agent_seed{idx}"
        config = {**base_config, "seed": idx}
        print(f"[task {index1}/{TOTAL_TASKS}] base={base} AGENT seed={idx} -> {run_dir}", flush=True)
        result = walk_forward_gate(returns, config, run_dir=run_dir)
    else:
        run_dir = RUN_DIR / base / f"placebo_{idx:02d}"
        config = {**base_config, "seed": PLACEBO_TRAIN_BASE + idx}
        surrogate = phase_randomize(returns, seed=PLACEBO_PHASE_BASE + idx)
        print(f"[task {index1}/{TOTAL_TASKS}] base={base} PLACEBO draw={idx} -> {run_dir}", flush=True)
        result = walk_forward_gate(surrogate, config, run_dir=run_dir)

    print(f"[task {index1}/{TOTAL_TASKS}] DONE base={base} {kind} idx={idx} "
          f"mean_skill={result['mean_skill']:.6e}", flush=True)


if __name__ == "__main__":
    main()
