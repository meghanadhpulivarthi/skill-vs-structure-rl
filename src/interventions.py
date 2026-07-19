"""Causal-probing interventions for the RQ3 faithfulness experiment.

The CAUSAL track: replay the trained gate agent's deterministic decisions on a
held-out market, then measure how those decisions change under interventions —
feature-group ablation (freeze / permute) and environment interventions (vol
shock, signal flip). Pure functions operating on injected callables and arrays,
so they are testable on a known policy and reusable on the real PPO agent.
"""
import numpy as np

from src.train import build_env


def feature_groups(window: int, n_assets: int) -> dict:
    """Index sets partitioning the gate observation into semantic groups.

    Gate obs layout (src/allocation_env.py): [window*n_assets returns | n_assets
    short_vol | 1 signal]. The `signal` group is the known ground-truth driver.
    """
    n_returns = window * n_assets
    return {
        "returns": list(range(0, n_returns)),
        "short_vol": list(range(n_returns, n_returns + n_assets)),
        "signal": [n_returns + n_assets],
    }


def rollout_observations(model, market: dict, config: dict) -> np.ndarray:
    """Replay the deterministic policy; return the observations that drove each
    decision (shape [T, obs_dim]). The terminal all-zeros obs is excluded because
    it never drives a decision."""
    env = build_env(market, config)
    obs, _ = env.reset(seed=0)
    observations = []
    done = False
    while not done:
        observations.append(np.asarray(obs, dtype=np.float32))
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return np.asarray(observations, dtype=np.float32)


def make_gate_fn(model):
    """Adapter: numpy observation stack -> gate values in [0,1] (clipped as the
    env does). Used by both feature ablation and KernelSHAP."""
    def gate_fn(observations: np.ndarray) -> np.ndarray:
        obs = np.atleast_2d(np.asarray(observations, dtype=np.float32))
        actions, _ = model.predict(obs, deterministic=True)
        return np.clip(np.asarray(actions, dtype=float).reshape(-1), 0.0, 1.0)
    return gate_fn
