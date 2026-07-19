"""Gymnasium env: PPO learns either a scalar DE-RISKING GATE or a bounded
PER-ASSET TILT on a structural-null base.

Structure-baselined reward is the methodological core (spec §5.3): the agent is
rewarded ONLY for the log-growth it adds over the base on the same market path,
net of its extra turnover cost.

Action modes (spec §5.2, revised 2026-07-18):
  gate (default): scalar gate g in [0, 1]; w = (1-g)*base + g*safe.
    g=0 reproduces the base exactly (zero skill by construction).
  tilt: per-asset tilt a in [-ACTION_BOUND, ACTION_BOUND]^n;
    w = project_to_simplex(base + max_tilt*tanh(a)).
    Zero action reproduces the base exactly (same zero-skill floor).
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.base_policies import BASE_POLICIES
from src.simplex import project_to_simplex

# Tilt-mode action bound. tanh saturates well inside +/-4, so the agent can reach
# +/-max_tilt while SB3 gets the finite Box bound it requires.
ACTION_BOUND = 4.0


class AllocationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, market: dict, base_name: str, window: int = 20,
                 cost_bps: float = 10.0, safe_asset_index: int = None,
                 action_mode: str = "gate", max_tilt: float = 0.15):
        super().__init__()
        if base_name not in BASE_POLICIES:
            raise ValueError(f"unknown base_name {base_name!r}; expected one of {list(BASE_POLICIES)}")
        if action_mode not in ("gate", "tilt"):
            raise ValueError(f"unknown action_mode {action_mode!r}; expected 'gate' or 'tilt'")
        self.returns = np.asarray(market["returns"], dtype=float)
        self.signal = np.asarray(market["signal"], dtype=float)
        self.n_steps, self.n_assets = self.returns.shape
        self.base_policy = BASE_POLICIES[base_name]
        self.window = window
        self.cost_rate = cost_bps * 1e-4
        self.action_mode = action_mode
        self.max_tilt = max_tilt
        # tilt mode adds a longer-horizon volatility feature, so it needs 2*window
        # of history before the first decision.
        self.long_window = 2 * window
        self.start_t = window if action_mode == "gate" else self.long_window

        # Safe allocation the GATE de-risks toward (unused in tilt mode).
        if safe_asset_index is None:
            safe_asset_index = self.n_assets - 1
        if not 0 <= safe_asset_index < self.n_assets:
            raise ValueError(f"safe_asset_index {safe_asset_index} out of range for {self.n_assets} assets")
        self.safe_asset_index = safe_asset_index
        self.safe_weights = np.zeros(self.n_assets)
        self.safe_weights[safe_asset_index] = 1.0

        if action_mode == "gate":
            obs_dim = window * self.n_assets + self.n_assets + 1
            self.action_space = spaces.Box(0.0, 1.0, (1,), dtype=np.float32)
        else:
            obs_dim = window * self.n_assets + 4 * self.n_assets + 1
            self.action_space = spaces.Box(-ACTION_BOUND, ACTION_BOUND, (self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)

        self._t = None
        self._prev_weights = None
        self._prev_base = None
        self.last_info = None

    def _observation(self) -> np.ndarray:
        win = self.returns[self._t - self.window:self._t]
        short_vol = win.std(axis=0)
        if self.action_mode == "gate":
            obs = np.concatenate([win.flatten(), short_vol, [self.signal[self._t]]])
            return obs.astype(np.float32)
        long_win = self.returns[self._t - self.long_window:self._t]
        long_vol = long_win.std(axis=0)
        momentum = win.mean(axis=0)
        base_weights = self.base_policy(win)
        obs = np.concatenate([win.flatten(), short_vol, long_vol, momentum,
                              base_weights, [self.signal[self._t]]])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = self.start_t
        self._prev_weights = np.full(self.n_assets, 1.0 / self.n_assets)
        self._prev_base = np.full(self.n_assets, 1.0 / self.n_assets)
        self.last_info = None
        return self._observation(), {}

    def _net_log_return(self, weights, prev_weights, asset_returns):
        turnover = 0.5 * np.abs(weights - prev_weights).sum()
        gross = float(weights @ asset_returns)
        cost = self.cost_rate * turnover
        return np.log(1.0 + gross) - cost

    def step(self, action):
        win = self.returns[self._t - self.window:self._t]
        base_weights = self.base_policy(win)
        if self.action_mode == "gate":
            gate = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], 0.0, 1.0))
            # Blend of two simplex points is itself on the simplex — no projection needed.
            weights = (1.0 - gate) * base_weights + gate * self.safe_weights
            activity = gate
        else:
            tilt = self.max_tilt * np.tanh(np.asarray(action, dtype=float).reshape(-1))
            weights = project_to_simplex(base_weights + tilt)
            activity = 0.5 * float(np.abs(weights - base_weights).sum())

        asset_returns = self.returns[self._t]
        agent_log = self._net_log_return(weights, self._prev_weights, asset_returns)
        base_log = self._net_log_return(base_weights, self._prev_base, asset_returns)
        reward = agent_log - base_log  # structure-baselined credit

        self.last_info = {
            "weights": weights,
            "base_weights": base_weights,
            "gate": float(activity),
            "port_return": float(weights @ asset_returns),
            "base_return": float(base_weights @ asset_returns),
        }
        self._prev_weights = weights
        self._prev_base = base_weights
        self._t += 1

        terminated = self._t >= self.n_steps
        obs = self._observation() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), terminated, False, self.last_info
