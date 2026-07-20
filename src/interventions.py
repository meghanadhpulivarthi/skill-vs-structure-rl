"""Causal-probing interventions for the RQ3 faithfulness experiment.

The CAUSAL track: replay the trained gate agent's deterministic decisions on a
held-out market, then measure how those decisions change under interventions —
feature-group ablation (freeze / permute) and environment interventions (vol
shock, signal flip). Pure functions operating on injected callables and arrays,
so they are testable on a known policy and reusable on the real PPO agent.
"""
import numpy as np

from src.simplex import project_to_simplex
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


def feature_groups_tilt(window: int, n_assets: int) -> dict:
    """Semantic index groups for the TILT observation (src/allocation_env.py tilt mode):
    [window*n returns | n short_vol | n long_vol | n momentum | n base_weights | 1 signal].
    The `signal` group is the known ground-truth driver."""
    n_returns = window * n_assets
    offset = n_returns
    return {
        "returns": list(range(0, n_returns)),
        "short_vol": list(range(offset, offset + n_assets)),
        "long_vol": list(range(offset + n_assets, offset + 2 * n_assets)),
        "momentum": list(range(offset + 2 * n_assets, offset + 3 * n_assets)),
        "base_weights": list(range(offset + 3 * n_assets, offset + 4 * n_assets)),
        "signal": [offset + 4 * n_assets],
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


def freeze_group(observations: np.ndarray, indices: list) -> np.ndarray:
    """Replace the group's columns with their per-column mean over the stack —
    removes that group's variation while leaving its scale roughly intact."""
    out = np.array(observations, dtype=np.float32)   # copy
    out[:, indices] = observations[:, indices].mean(axis=0, keepdims=True)
    return out


def permute_group(observations: np.ndarray, indices: list, rng: np.random.Generator) -> np.ndarray:
    """Row-permute the group's columns (shared permutation), breaking their
    temporal alignment with the target while preserving each column's marginal."""
    out = np.array(observations, dtype=np.float32)   # copy
    order = rng.permutation(observations.shape[0])
    out[:, indices] = observations[order][:, indices]
    return out


def causal_effect(gate_fn, observations, indices, mode: str, seed: int = 0) -> float:
    """Mean absolute change in the gate decision when `indices` are ablated.
    Larger => the agent's de-risking depends more on those features."""
    observations = np.asarray(observations, dtype=np.float32)
    baseline = gate_fn(observations)
    if mode == "freeze":
        ablated_obs = freeze_group(observations, indices)
    elif mode == "permute":
        ablated_obs = permute_group(observations, indices, np.random.default_rng(seed))
    else:
        raise ValueError(f"unknown mode {mode!r}; expected 'freeze' or 'permute'")
    ablated = gate_fn(ablated_obs)
    return float(np.mean(np.abs(np.asarray(baseline) - np.asarray(ablated))))


def inject_vol_shock(market: dict, t0: int, width: int, multiplier: float,
                     risky_index: int = 0) -> dict:
    """Transiently amplify the risky asset's return magnitude over [t0, t0+width)
    (a volatility spike). Returns a NEW market dict; the input is not mutated."""
    returns = np.array(market["returns"], dtype=float)   # copy
    returns[t0:t0 + width, risky_index] *= multiplier
    return {**market, "returns": returns}


def flip_signal(market: dict, t0: int, value: float) -> dict:
    """Force the leading signal to `value` at step t0, holding the rest fixed, to
    read the gate's response to the signal alone. Returns a NEW market dict."""
    signal = np.array(market["signal"], dtype=float)     # copy
    signal[t0] = value
    return {**market, "signal": signal}


def make_safe_weight_fn(model, base_obs_idx, safe_asset_idx, max_tilt):
    """Adapter: numpy observation stack -> safe-block weight of the executed tilt portfolio.
    Reads the base weights from the obs (base_obs_idx), applies the bounded tilt
    (max_tilt*tanh(action)) and the simplex projection, and sums the safe-asset weights.
    This is the tilt analog of make_gate_fn (the scalar de-risking behavioral object)."""
    base_obs_idx = list(base_obs_idx)
    safe_asset_idx = list(safe_asset_idx)

    def safe_weight_fn(observations):
        obs = np.atleast_2d(np.asarray(observations, dtype=np.float32))
        actions, _ = model.predict(obs, deterministic=True)
        actions = np.atleast_2d(np.asarray(actions, dtype=float))
        base = obs[:, base_obs_idx].astype(float)
        tilt = max_tilt * np.tanh(actions)
        out = np.empty(len(obs), dtype=float)
        for i in range(len(obs)):
            weights = project_to_simplex(base[i] + tilt[i])
            out[i] = float(weights[safe_asset_idx].sum())
        return out
    return safe_weight_fn
