"""Parallel driver for the real RQ1 experiment (CPU, LSF-friendly).

The RQ1 pipeline runs 6 INDEPENDENT walk-forward computations — 1 agent run and
`n_placebo` placebo-null runs — each internally a sequence of per-fold PPO
trainings. This driver runs those 6 runs concurrently across worker processes
(each capped to a few threads), pre-filling the exact per-fold npz cache that
`run_rq1` -> `walk_forward_gate` / `placebo_null` expect. It then calls the
unchanged `run_rq1` with the same run_dir, which finds every fold cached and only
stitches + writes results/figures.

Nothing in the scientific computation changes: the driver calls the same
functions with the same seeds and the same cache layout as `src.rq1_real_data`,
so results are bit-identical to the sequential run — it is orchestration only.
GPU is intentionally NOT used: the policy is a tiny MLP and the bottleneck is
single-env Python rollout stepping, so this is a CPU-parallel job.
"""
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Config — edit these directly. MUST match src/rq1_real_data.py __main__ so the
# pre-filled cache matches what run_rq1 recomputes/reads.
BASE_NAME = "risk_parity"
WINDOW = 20
COST_BPS = 10.0
TOTAL_TIMESTEPS = 150_000
SEED = 0
INITIAL_TRAIN = 1260
TEST_BLOCK = 252
N_PLACEBO = 5
THREADS_PER_WORKER = 4                      # 6 workers x 4 threads = 24 cores (match bsub -n)
SAFE_TICKER_PREFERENCE = ("IEF", "TLT")     # de-risking sleeve, resolved by NAME (cols are alphabetical)
RUN_DIR = Path(__file__).resolve().parent.parent / "outputs" / "rq1_parallel"


def _limit_threads():
    # Each worker is one of 6 concurrent processes; cap its BLAS/torch threads so
    # the 6 runs share cores cleanly instead of each grabbing the whole node.
    import torch
    torch.set_num_threads(THREADS_PER_WORKER)


def agent_run(returns, config, run_dir):
    _limit_threads()
    from src.walk_forward import walk_forward_gate
    walk_forward_gate(returns, config, run_dir=Path(run_dir) / "agent")
    return "agent"


def placebo_run(returns, config, run_dir, draw):
    _limit_threads()
    from src.walk_forward import walk_forward_gate
    from src.placebo import phase_randomize
    # Mirror src.placebo.placebo_null exactly (as called by run_rq1 with
    # seed=config["seed"]+100): surrogate seed = SEED+100+draw, training seed =
    # SEED+draw, cache dir = <run_dir>/placebo/placebo_<draw>. This is the ONLY
    # replicated logic; a mismatch would merely cause run_rq1 to retrain (never a
    # wrong result), and the cache-hit log below verifies it.
    surrogate = phase_randomize(returns, seed=config["seed"] + 100 + draw)
    draw_dir = Path(run_dir) / "placebo" / f"placebo_{draw:02d}"
    walk_forward_gate(surrogate, {**config, "seed": config["seed"] + draw}, run_dir=draw_dir)
    return f"placebo_{draw}"


def main():
    os.environ.setdefault("OMP_NUM_THREADS", str(THREADS_PER_WORKER))
    from src.data import load_etf_panel
    from src.rq1_real_data import run_rq1

    panel = load_etf_panel()
    returns = panel["returns"]
    tickers = panel["tickers"]

    safe_ticker = next((t for t in SAFE_TICKER_PREFERENCE if t in tickers), None)
    if safe_ticker is None:
        raise ValueError(f"no safe sleeve {SAFE_TICKER_PREFERENCE} in panel tickers {tickers}")
    safe_index = tickers.index(safe_ticker)
    print(f"rq1_parallel: safe sleeve = {safe_ticker} at column {safe_index} of {tickers}", flush=True)

    config = {"base_name": BASE_NAME, "window": WINDOW, "cost_bps": COST_BPS,
              "safe_asset_index": safe_index, "total_timesteps": TOTAL_TIMESTEPS,
              "seed": SEED, "initial_train": INITIAL_TRAIN, "test_block": TEST_BLOCK,
              "n_placebo": N_PLACEBO}
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"rq1_parallel: pre-filling cache in {RUN_DIR} with {1 + N_PLACEBO} concurrent runs "
          f"({THREADS_PER_WORKER} threads each)", flush=True)

    # spawn (not fork) so torch/SB3 state is not inherited across the pool.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1 + N_PLACEBO, mp_context=context) as executor:
        futures = [executor.submit(agent_run, returns, config, str(RUN_DIR))]
        for draw in range(N_PLACEBO):
            futures.append(executor.submit(placebo_run, returns, config, str(RUN_DIR), draw))
        # .result() re-raises any worker exception loudly instead of hanging.
        for future in futures:
            print(f"rq1_parallel: completed {future.result()}", flush=True)

    print("rq1_parallel: cache pre-filled; aggregating via run_rq1 (should hit cache for every fold)",
          flush=True)
    result = run_rq1(config, returns=returns, run_dir=RUN_DIR)
    print(f"rq1_parallel: DONE. skill_net={result['skill_net']:.6e} "
          f"CI={result['skill_net_ci']} placebo_mean={result['placebo_mean']:.6e} "
          f"placebo_exceedance={result['placebo_exceedance']:.3f}", flush=True)


if __name__ == "__main__":
    main()
