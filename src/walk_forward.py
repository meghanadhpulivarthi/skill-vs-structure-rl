"""Anchored (expanding-window) walk-forward for the de-risking-gate agent.

Retrains the gate on each fold's train slice and evaluates on the immediately
following out-of-sample block, then stitches OOS results. No full-sample tuning
(spec §8). The stitched mean structure-baselined reward is the OOS skill measure;
the placebo null (Task 6) turns it into a skill-net-of-luck statistic. Restartable:
completed folds are cached and skipped, and mean_skill is always recomputed over
all folds (never a partial subset).
"""
from pathlib import Path

import numpy as np

from src.real_market import build_real_market
from src.train import build_env, train_agent


def make_folds(n_steps: int, initial_train: int, test_block: int) -> list:
    folds = []
    train_stop = initial_train
    while train_stop < n_steps:
        test_stop = min(train_stop + test_block, n_steps)
        folds.append((range(0, train_stop), range(train_stop, test_stop)))
        train_stop = test_stop
    return folds


def roll_policy(model, market: dict, config: dict) -> dict:
    env = build_env(market, config)
    obs, _ = env.reset(seed=0)
    baselined_reward, port_return, base_return, gate = [], [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        baselined_reward.append(reward)
        port_return.append(info["port_return"])
        base_return.append(info["base_return"])
        gate.append(info["gate"])
        done = term or trunc
    return {
        "baselined_reward": np.asarray(baselined_reward, dtype=float),
        "port_return": np.asarray(port_return, dtype=float),
        "base_return": np.asarray(base_return, dtype=float),
        "gate": np.asarray(gate, dtype=float),
    }


def walk_forward_gate(returns: np.ndarray, config: dict, run_dir=None) -> dict:
    returns = np.asarray(returns, dtype=float)
    folds = make_folds(returns.shape[0], config["initial_train"], config["test_block"])
    window = config["window"]
    safe_index = config["safe_asset_index"]
    run_dir = Path(run_dir) if run_dir is not None else None
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)

    if config["initial_train"] < window:
        raise ValueError(f"initial_train ({config['initial_train']}) must be >= window ({window}) "
                         "so each fold has a real preceding warm-up window")

    per_fold = []
    for fold_index, (train_slice, test_slice) in enumerate(folds):
        cache = run_dir / f"fold_{fold_index:02d}.npz" if run_dir is not None else None
        if cache is not None and cache.exists():
            print(f"walk_forward_gate: fold {fold_index} cached; loading {cache}")
            per_fold.append(dict(np.load(cache)))
            continue

        train_returns = returns[train_slice.start:train_slice.stop]
        # Prepend the required warm-up window to the test slice so the env's warm-up
        # is drawn from genuine prior data. Gate mode starts at _t=window; tilt mode
        # starts at _t=2*window for richer observation. roll_policy yields exactly
        # the test-region steps: no per-fold OOS-day drop and no cold-started signal.
        required_window = (2 * window if config.get("action_mode", "gate") == "tilt"
                           else window)
        eval_start = test_slice.start - required_window
        test_returns = returns[eval_start:test_slice.stop]
        train_market = build_real_market(train_returns, safe_index, window)
        model = train_agent(train_market, config)
        test_market = build_real_market(test_returns, safe_index, window)
        rolled = roll_policy(model, test_market, config)
        print(f"walk_forward_gate: fold {fold_index} OOS mean skill = "
              f"{rolled['baselined_reward'].mean():.6e} (gate mean {rolled['gate'].mean():.3f})")
        if cache is not None:
            np.savez(cache, **rolled)
        per_fold.append(rolled)

    stitched = {
        f"oos_{key}": np.concatenate([fold[key] for fold in per_fold])
        for key in ["baselined_reward", "port_return", "base_return", "gate"]
    }
    fold_mean_skill = [float(fold["baselined_reward"].mean()) for fold in per_fold]
    # Recompute the aggregate over ALL stitched OOS steps, not a partial subset.
    stitched["fold_mean_skill"] = fold_mean_skill
    stitched["mean_skill"] = float(stitched["oos_baselined_reward"].mean())
    return stitched
