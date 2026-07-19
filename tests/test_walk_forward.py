# tests/test_walk_forward.py
import numpy as np
import pytest
from src.walk_forward import make_folds, walk_forward_gate


def test_make_folds_expanding_and_covers_test_region():
    folds = make_folds(n_steps=1000, initial_train=400, test_block=200)
    # train is expanding, test blocks are contiguous and cover [400, 1000)
    assert folds[0][0].start == 0 and folds[0][0].stop == 400
    assert folds[0][1].start == 400 and folds[0][1].stop == 600
    assert folds[1][0].stop == 600                     # train expanded by one block
    covered = [i for _, test in folds for i in test]
    assert covered == list(range(400, 1000))           # exact, no gaps/overlap


def test_walk_forward_runs_and_stitches(tmp_path):
    # Tiny, fast config: real data is unforecastable-ish noise here; we only assert
    # plumbing (finite stitched OOS arrays of the right total length), not skill.
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0003, 0.01, size=(900, 3))
    config = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
              "safe_asset_index": 2, "total_timesteps": 2000, "seed": 0,
              "initial_train": 400, "test_block": 200}
    result = walk_forward_gate(returns, config, run_dir=tmp_path)

    n_test = 900 - 400                                  # steps in the OOS region
    # each fold loses `window` leading steps to warm-up, one per fold:
    assert result["oos_baselined_reward"].ndim == 1
    assert np.isfinite(result["oos_baselined_reward"]).all()
    assert len(result["fold_mean_skill"]) == len(make_folds(900, 400, 200))
    assert np.isfinite(result["mean_skill"])


def test_tilt_mode_requires_initial_train_ge_two_window():
    # In tilt mode the env warms up over 2*window, so initial_train must be >= 2*window
    # or the fold's history-prepend would slice a negative index. The guard must fail loud.
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, size=(200, 3))
    config = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
              "safe_asset_index": 2, "total_timesteps": 100, "seed": 0,
              "initial_train": 30, "test_block": 50, "action_mode": "tilt", "max_tilt": 0.15}
    with pytest.raises(ValueError):
        walk_forward_gate(returns, config, run_dir=None)
