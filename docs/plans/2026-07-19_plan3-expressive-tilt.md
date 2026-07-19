# Expressive Residual-Tilt Agent Implementation Plan (Plan 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the RL allocator a genuinely expressive action (a bounded per-asset tilt on the structural-null base) plus richer features, re-validate the skill measure on synthetic ground truth, and re-run the real-data RQ1 verdict — so the "skill vs. structure" conclusion is about RL capability, not action-space poverty.

**Architecture:** Parameterize the existing `AllocationEnv` with `action_mode ∈ {"gate","tilt"}`. Tilt mode outputs a per-asset tilt `a`, executes `w = project_to_simplex(base + max_tilt·tanh(a))` (zero action → base), and observes an enriched, causal feature set. The structure-baselined reward, long-only simplex, base ladder, walk-forward, placebo null, and LSF drivers are all reused unchanged; only `action_mode`/`max_tilt` flow through the config. RQ2 (synthetic) re-validation is the gate that must pass before any real-data claim.

**Tech Stack:** Python 3.11, `uv`, NumPy, SciPy, Gymnasium, Stable-Baselines3 (PPO, PyTorch), pytest, tqdm, matplotlib.

## Global Constraints

- Package manager is `uv` only (`uv add`, `uv run python -u`); never pip. (CLAUDE.md)
- Config constants at the TOP of each file, clearly labelled; `snake_case`; full words; flat over nested; 4-space indent. (code-style.md)
- No silent failures: no bare `except`; every skipped/guarded branch logged; `dict[key]` not `.get()` unless the absence is an explicitly handled default; no `None`/`NaN` flowing downstream unguarded. (code-style.md)
- No absolute paths; scripts runnable from any working directory (resolve paths relative to `__file__`). (code-style.md)
- Determinism: every stochastic function takes an explicit integer `seed`. Traceability: experiment scripts print a run header and write to a timestamped `outputs/` dir with `config.json`/`results.json`. (traceability.md)
- **No lookahead:** every observation feature at decision step `t` uses only data strictly before the return it earns (`returns[:t]`). Unit-tested.
- **Measurement invariants (do NOT change):** structure-baselined reward `agent_net_log − base_net_log`; long-only simplex weights; zero action reproduces the base exactly (free do-nothing floor); skill on real data reported **net of the phase-randomization placebo null**; **RQ2 re-validation precedes any real-data claim**. (spec §2)
- **Gate mode must stay behavior-identical** (default `action_mode="gate"`), so existing Plan-1/Plan-2 results and tests remain reproducible.
- Action bound: tilt-mode action space is `Box(-ACTION_BOUND, ACTION_BOUND, (n_assets,))` with `ACTION_BOUND = 4.0` (finite bound required by SB3; `tanh(4)≈0.999` so the agent can reach ±`max_tilt`). Default `max_tilt = 0.15`.

## Interfaces reused unchanged (Plan 1/2)

- `src.simplex.project_to_simplex(v) -> np.ndarray`
- `src.base_policies.BASE_POLICIES` (keys `equal_weight`, `vol_scaled`, `risk_parity`)
- `src.train.train_agent(market, config) -> PPO` (reads `base_name, window, cost_bps, safe_asset_index, total_timesteps, seed`; will additionally read `action_mode, max_tilt` after Task 3)
- `src.walk_forward.walk_forward_gate`, `src.placebo.placebo_null`, `src.rq1_real_data.run_rq1` — all consume `config` and reach the env through `src.train.build_env`, so they run the tilt model once `action_mode`/`max_tilt` are in the config (Task 3). No changes needed in those modules.
- `src.validate_skill.evaluate_skill(model, market, config)` (rolls the policy; reads `info["gate"]`, `info["port_return"]`, `info["base_return"]`, `info["weights"]`).

## File structure

```
src/allocation_env.py     # MODIFY: add action_mode "gate"|"tilt", max_tilt, enriched tilt obs (Task 1)
src/synthetic_market.py   # MODIFY: add generate_multi_regime_market (fair multi-asset RQ2 bed) (Task 2)
src/train.py              # MODIFY: build_env passes action_mode/max_tilt from config (Task 3)
src/validate_skill.py     # MODIFY: market factory (config["market"]) so RQ2 can use the tilt bed (Task 4)
scripts/rq1_sweep_task.py # MODIFY: ACTION_MODE/MAX_TILT constants so the LSF sweep can run the tilt model (Task 5)
tests/test_*.py           # one test module per task
```

---

### Task 1: `AllocationEnv` tilt mode (bounded per-asset tilt + enriched causal observation)

**Files:**
- Modify: `src/allocation_env.py`
- Test: `tests/test_allocation_env_tilt.py` (new; keep `tests/test_allocation_env.py` — it is the gate-mode regression suite)

**Interfaces:**
- Consumes: `project_to_simplex` (add import), `BASE_POLICIES`.
- Produces: `AllocationEnv(market, base_name, window=20, cost_bps=10.0, safe_asset_index=None, action_mode="gate", max_tilt=0.15)`.
  - `action_mode="gate"` (default): unchanged — action `Box(0,1,(1,))`, `w=(1-g)·base+g·safe`, obs = `[window·n | per-asset vol | signal]`, episode starts at `_t=window`.
  - `action_mode="tilt"`: action `Box(-4,4,(n,))`; `w=project_to_simplex(base + max_tilt·tanh(a))`; obs = `[window·n | short vol(n) | long vol(n) | momentum(n) | base weights(n) | signal(1)]`; episode starts at `_t=long_window` where `long_window=2·window`.
  - `last_info["gate"]` in tilt mode holds the **activity scalar** `0.5·Σ|w−base|` (deviation from base), so `evaluate_skill`/`roll_policy` keep working; other `last_info` keys unchanged.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_allocation_env_tilt.py
import numpy as np
from src.synthetic_market import generate_market
from src.allocation_env import AllocationEnv


def _tilt_env(n_assets=5, n_steps=400, max_tilt=0.15):
    market = generate_market(n_assets=n_assets, n_steps=n_steps, seed=5, signal_strength=0.8)
    return AllocationEnv(market, base_name="equal_weight", window=20, cost_bps=10.0,
                         action_mode="tilt", max_tilt=max_tilt)


def test_tilt_reset_obs_matches_declared_shape():
    env = _tilt_env()
    obs, _ = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    # enriched obs: window*n + 4*n + 1
    assert env.observation_space.shape[0] == 20 * 5 + 4 * 5 + 1


def test_tilt_action_space_is_per_asset_and_bounded():
    env = _tilt_env()
    assert env.action_space.shape == (5,)
    assert np.isfinite(env.action_space.low).all() and np.isfinite(env.action_space.high).all()


def test_tilt_zero_action_reproduces_base_and_zero_reward():
    env = _tilt_env()
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.zeros(5))
    np.testing.assert_allclose(info["weights"], info["base_weights"], atol=1e-8)
    assert abs(reward) < 1e-8


def test_tilt_weights_stay_on_simplex_for_arbitrary_actions():
    env = _tilt_env()
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    for _ in range(50):
        _, _, term, trunc, info = env.step(rng.normal(size=5))
        w = info["weights"]
        assert np.all(w >= -1e-9)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-8)
        if term or trunc:
            break


def test_tilt_bounded_by_max_tilt():
    # each executed weight is within max_tilt of the base before projection renormalizes;
    # deviation per asset cannot exceed max_tilt by more than projection slack.
    env = _tilt_env(max_tilt=0.15)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.full(5, 100.0))  # saturates tanh -> +max_tilt each
    dev = np.abs(info["weights"] - info["base_weights"])
    assert dev.max() <= 0.15 + 1e-6


def test_tilt_episode_length_uses_long_window():
    env = _tilt_env(n_steps=400)
    env.reset(seed=0)
    steps = 0
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(np.zeros(5))
        done = term or trunc
        steps += 1
    assert steps == 400 - 40  # n_steps - long_window (2*window)


def test_tilt_info_gate_is_activity_scalar():
    env = _tilt_env()
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.full(5, 100.0))
    expected = 0.5 * np.abs(info["weights"] - info["base_weights"]).sum()
    assert abs(info["gate"] - expected) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_allocation_env_tilt.py -v`
Expected: FAIL — `AllocationEnv.__init__() got an unexpected keyword argument 'action_mode'`.

- [ ] **Step 3: Modify `src/allocation_env.py`**

Add the projection import near the top (after the `BASE_POLICIES` import):

```python
from src.simplex import project_to_simplex
```

Add a module constant after the imports:

```python
# Tilt-mode action bound. tanh saturates well inside +/-4, so the agent can reach
# +/-max_tilt while SB3 gets the finite Box bound it requires.
ACTION_BOUND = 4.0
```

Replace the constructor and the reset/observation/step methods with the mode-aware versions. The constructor:

```python
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
```

The observation (mode-aware; all features causal — they slice `returns` strictly before `self._t`):

```python
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
```

Reset (starts at the mode's `start_t`):

```python
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = self.start_t
        self._prev_weights = np.full(self.n_assets, 1.0 / self.n_assets)
        self._prev_base = np.full(self.n_assets, 1.0 / self.n_assets)
        self.last_info = None
        return self._observation(), {}
```

Step (mode-aware action → weights; shared reward core; `info["gate"]` = gate value or tilt activity):

```python
    def step(self, action):
        win = self.returns[self._t - self.window:self._t]
        base_weights = self.base_policy(win)
        if self.action_mode == "gate":
            gate = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], 0.0, 1.0))
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
```

Leave `_net_log_return` unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_allocation_env_tilt.py tests/test_allocation_env.py -v`
Expected: new tilt tests pass AND the existing gate tests still pass (regression — gate mode unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/allocation_env.py tests/test_allocation_env_tilt.py
git commit -m "feat: AllocationEnv tilt action mode (bounded per-asset tilt + enriched causal obs)"
```

---

### Task 2: Multi-asset regime market (fair RQ2 bed for the tilt model)

**Files:**
- Modify: `src/synthetic_market.py`
- Test: `tests/test_synthetic_market.py` (append)

**Interfaces:**
- Consumes: nothing new (reuses the module's regime/signal machinery and the `RISKY_*`/`SAFE_*` constants).
- Produces: `generate_multi_regime_market(n_risky: int, n_safe: int, n_steps: int, seed: int, signal_strength: float, crisis_persistence: float = 0.9, calm_persistence: float = 0.98) -> dict` with `returns` `(n_steps, n_risky+n_safe)` (risky assets first, then safe), `signal` `(n_steps,)`, `regime` `(n_steps,)`. Risky assets crash in crisis (`RISKY_*` params, cross-correlated via `CRISIS_CORR`); safe assets stay calm (`SAFE_*` params). The tilt agent can tilt toward the safe block when `signal` fires — the cross-asset skill RQ2 tests.

- [ ] **Step 1: Write the failing test (append to `tests/test_synthetic_market.py`)**

```python
# append to tests/test_synthetic_market.py
from src.synthetic_market import generate_multi_regime_market


def test_multi_regime_shapes_and_safe_block_is_low_vol():
    m = generate_multi_regime_market(n_risky=3, n_safe=2, n_steps=20_000, seed=1, signal_strength=0.9)
    assert m["returns"].shape == (20_000, 5)
    risky_vol = m["returns"][:, :3].std()
    safe_vol = m["returns"][:, 3:].std()
    assert safe_vol < 0.3 * risky_vol


def test_multi_regime_crisis_hurts_risky_not_safe():
    m = generate_multi_regime_market(n_risky=3, n_safe=2, n_steps=20_000, seed=2, signal_strength=0.9)
    crisis = m["regime"] == 1
    risky_mean = m["returns"][:, :3].mean(axis=1)
    safe_mean = m["returns"][:, 3:].mean(axis=1)
    assert risky_mean[crisis].mean() < risky_mean[~crisis].mean()
    assert abs(safe_mean[crisis].mean() - safe_mean[~crisis].mean()) < 0.001


def test_multi_regime_signal_toggle():
    on = generate_multi_regime_market(n_risky=3, n_safe=2, n_steps=20_000, seed=3, signal_strength=0.9)
    corr_on = np.corrcoef(on["signal"][:-1], on["regime"][1:])[0, 1]
    assert corr_on > 0.2
    off = generate_multi_regime_market(n_risky=3, n_safe=2, n_steps=20_000, seed=3, signal_strength=0.0)
    corr_off = abs(np.corrcoef(off["signal"][:-1], off["regime"][1:])[0, 1])
    assert corr_off < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_synthetic_market.py -k multi_regime -v`
Expected: FAIL — `cannot import name 'generate_multi_regime_market'`.

- [ ] **Step 3: Add the implementation to `src/synthetic_market.py`** (after `generate_risky_safe_market`)

```python
def generate_multi_regime_market(
    n_risky: int,
    n_safe: int,
    n_steps: int,
    seed: int,
    signal_strength: float,
    crisis_persistence: float = 0.9,
    calm_persistence: float = 0.98,
) -> dict:
    """Multi-asset risky+safe world (risky assets first, then safe).

    Generalizes generate_risky_safe_market to n_risky crisis-hurt assets (cross-
    correlated, so diversification within the risky block fails in a crisis) and
    n_safe calm safe-haven assets. The signal predicts the next crisis, so an
    expressive tilt agent can time a tilt toward the safe block — the cross-asset
    skill the tilt RQ2 tests. signal_strength=0 removes the information (the null).
    """
    rng = np.random.default_rng(seed)

    regime = np.zeros(n_steps, dtype=int)
    for t in range(1, n_steps):
        if regime[t - 1] == 0:
            regime[t] = 0 if rng.random() < calm_persistence else 1
        else:
            regime[t] = 1 if rng.random() < crisis_persistence else 0

    noise = rng.random(n_steps)
    next_is_crisis = np.zeros(n_steps)
    next_is_crisis[:-1] = (regime[1:] == 1).astype(float)
    signal = signal_strength * next_is_crisis + (1.0 - signal_strength) * noise

    calm_cov = _regime_cov(n_risky, RISKY_CALM_VOL, CALM_CORR)
    crisis_cov = _regime_cov(n_risky, RISKY_CRISIS_VOL, CRISIS_CORR)
    returns = np.zeros((n_steps, n_risky + n_safe))
    for t in range(n_steps):
        if regime[t] == 0:
            returns[t, :n_risky] = rng.multivariate_normal(np.full(n_risky, RISKY_CALM_DRIFT), calm_cov)
        else:
            returns[t, :n_risky] = rng.multivariate_normal(np.full(n_risky, RISKY_CRISIS_DRIFT), crisis_cov)
        returns[t, n_risky:] = rng.normal(SAFE_DRIFT, SAFE_VOL, size=n_safe)

    return {"returns": returns, "signal": signal, "regime": regime}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_synthetic_market.py -k multi_regime -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/synthetic_market.py tests/test_synthetic_market.py
git commit -m "feat: multi-asset regime market (cross-asset RQ2 bed for the tilt agent)"
```

---

### Task 3: Wire `action_mode`/`max_tilt` through `build_env` + tilt smoke

**Files:**
- Modify: `src/train.py`
- Test: `tests/test_train_tilt.py` (new)

**Interfaces:**
- Consumes: `AllocationEnv` tilt mode (Task 1), `generate_multi_regime_market` (Task 2).
- Produces: `build_env` reads `config["action_mode"]` (default `"gate"`) and `config["max_tilt"]` (default `0.15`) and passes them to `AllocationEnv`. `train_agent` is unchanged (it already builds via `build_env`). This makes every downstream consumer (`walk_forward`, `placebo`, `rq1_real_data`, sweep) run the tilt model when the config carries `action_mode="tilt"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_tilt.py
import numpy as np
from src.synthetic_market import generate_multi_regime_market
from src.train import build_env, train_agent


def test_build_env_defaults_to_gate():
    market = generate_multi_regime_market(n_risky=3, n_safe=2, n_steps=300, seed=1, signal_strength=0.9)
    env = build_env(market, {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0})
    assert env.action_mode == "gate"                      # default preserved


def test_tilt_agent_beats_base_when_signal_exists():
    # With a strong leading signal, the tilt agent should earn positive mean
    # structure-baselined reward OOS (times a tilt toward the safe block). End-to-end
    # check that the expressive action can detect skill when skill is possible.
    config = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
              "action_mode": "tilt", "max_tilt": 0.15, "total_timesteps": 120_000, "seed": 0}
    market = generate_multi_regime_market(3, 2, 6000, seed=11, signal_strength=0.95)
    model = train_agent(market, config)

    eval_market = generate_multi_regime_market(3, 2, 6000, seed=12, signal_strength=0.95)
    env = build_env(eval_market, config)
    obs, _ = env.reset(seed=0)
    rewards = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        rewards.append(reward)
        done = term or trunc
    assert np.isfinite(rewards).all()
    assert np.mean(rewards) > 0.0                          # detects skill when signal exists
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train_tilt.py::test_build_env_defaults_to_gate -v`
Expected: FAIL — `AttributeError: 'AllocationEnv' object has no attribute 'action_mode'` is already fixed by Task 1, so this fails instead because `build_env` does not yet pass `action_mode` (env defaults to gate, so this specific assert may pass) — the binding failure is the tilt test, which errors because `build_env` drops `action_mode`/`max_tilt` and builds a gate env whose action space is `(1,)`, so `model.predict` returns a scalar action the tilt eval mishandles. Confirm at least one test in the file fails before implementing.

- [ ] **Step 3: Modify `build_env` in `src/train.py`**

```python
def build_env(market: dict, config: dict) -> AllocationEnv:
    return AllocationEnv(
        market,
        base_name=config["base_name"],
        window=config["window"],
        cost_bps=config["cost_bps"],
        safe_asset_index=config.get("safe_asset_index"),
        action_mode=config.get("action_mode", "gate"),
        max_tilt=config.get("max_tilt", 0.15),
    )
```

(`.get` here is the explicitly-handled-default case allowed by the style rule — absence means gate/0.15.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_train_tilt.py -v -s`
Expected: 2 passed. (The tilt smoke trains ~120k PPO steps; a couple minutes on CPU. If `mean(rewards) > 0` is flaky, raise `total_timesteps` to 200_000 — do NOT weaken the assertion; a tilt agent that cannot beat the base with a near-perfect signal and a safe block to tilt into is under-trained or `max_tilt` is too small.)

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train_tilt.py
git commit -m "feat: pass action_mode/max_tilt through build_env; tilt smoke on multi-regime market"
```

---

### Task 4: Tilt RQ2 re-validation (the ground-truth gate) + `max_tilt` calibration

**Files:**
- Modify: `src/validate_skill.py`
- Test: `tests/test_validate_skill_tilt.py` (new)

**Interfaces:**
- Consumes: `run_skill_validation` (Plan 1), `generate_multi_regime_market` (Task 2), tilt env via config (Tasks 1/3).
- Produces: `run_skill_validation` selects the synthetic market from `config["market"]` — `"risky_safe"` (default, unchanged for the gate RQ2) or `"multi_regime"` (reads `config["n_risky"]`, `config["n_safe"]`). `action_mode`/`max_tilt` flow through `config` to `train_agent`/`evaluate_skill`. Same return shape (`by_strength`) and artifacts as before.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validate_skill_tilt.py
from src.validate_skill import run_skill_validation


def test_tilt_skill_vanishes_without_signal_and_appears_with_signal():
    # THE RQ2 GATE FOR THE TILT MODEL. The skill measure must be ~0 with no timeable
    # signal (the tilt agent must learn tilt~0) and clearly positive when the signal
    # exists (it times a tilt toward the safe block). Same validity claim as the gate,
    # now for the expressive action.
    config = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
              "action_mode": "tilt", "max_tilt": 0.15, "market": "multi_regime",
              "n_risky": 3, "n_safe": 2, "total_timesteps": 120_000, "n_steps": 6000}
    result = run_skill_validation(config, signal_strengths=(0.0, 0.95), n_seeds=3)

    skill_off = result["by_strength"]["0.0"]["mean_baselined_reward"]
    skill_on = result["by_strength"]["0.95"]["mean_baselined_reward"]

    # Floor 5e-5: comfortably above the ~1e-5 noise floor, below observed on-skill.
    # If skill_on falls below this, do NOT weaken the assertion — recalibrate max_tilt
    # (spec §3/§6: RQ2 is where max_tilt is set) and re-run. See context/decisions.md.
    assert skill_on > 5e-5
    assert abs(skill_off) < 5e-5
    assert skill_on > 3 * abs(skill_off)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_validate_skill_tilt.py -v`
Expected: FAIL — `run_skill_validation` ignores `config["market"]` and always builds `generate_risky_safe_market` (a 2-asset market), so `n_risky`/`n_safe` are unused and the tilt agent runs on the wrong bed.

- [ ] **Step 3: Modify `src/validate_skill.py`**

Add the multi-regime import to the existing import line:

```python
from src.synthetic_market import generate_risky_safe_market, generate_multi_regime_market
```

Add a market factory above `run_skill_validation`:

```python
def _build_synthetic_market(config: dict, seed: int, signal_strength: float) -> dict:
    market_kind = config.get("market", "risky_safe")   # default preserves the gate RQ2
    if market_kind == "risky_safe":
        return generate_risky_safe_market(config["n_steps"], seed=seed, signal_strength=signal_strength)
    if market_kind == "multi_regime":
        return generate_multi_regime_market(config["n_risky"], config["n_safe"], config["n_steps"],
                                            seed=seed, signal_strength=signal_strength)
    raise ValueError(f"unknown market {market_kind!r}; expected 'risky_safe' or 'multi_regime'")
```

Replace the two `generate_risky_safe_market(...)` calls inside `run_skill_validation` with the factory:

```python
            train_market = _build_synthetic_market(config, seed=1000 + seed, signal_strength=strength)
            model = train_agent(train_market, {**config, "seed": seed})
            eval_market = _build_synthetic_market(config, seed=2000 + seed, signal_strength=strength)
```

Leave everything else (aggregation, artifacts, plot) unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_validate_skill_tilt.py -v -s`
Expected: 1 passed. (Trains several tilt agents; minutes on CPU.) **If `skill_on` is not clearly above the floor:** this is the RQ2-gated `max_tilt` calibration the spec describes — try `max_tilt ∈ {0.2, 0.25}` (more expressive) and re-run; if `skill_off` rises above the floor instead, lower `max_tilt`. Record the chosen `max_tilt` and the RQ2 numbers in `context/decisions.md`. Do NOT weaken the thresholds — they encode the paper's validity claim (Rule 9). Also confirm the gate RQ2 is untouched: `uv run pytest tests/test_validate_skill.py -v`.

- [ ] **Step 5: Commit**

```bash
git add src/validate_skill.py tests/test_validate_skill_tilt.py context/decisions.md
git commit -m "feat: tilt RQ2 re-validation on multi-regime market (skill measure gate for the tilt model)"
```

---

### Task 5: Tilt real-data RQ1 integration + LSF sweep support

**Files:**
- Modify: `scripts/rq1_sweep_task.py` (add `ACTION_MODE`/`MAX_TILT` constants, threaded into every run's config)
- Test: `tests/test_rq1_real_data_tilt.py` (new)

**Interfaces:**
- Consumes: `run_rq1` (Plan 2), tilt config path (Tasks 1/3). No changes to `walk_forward`/`placebo`/`rq1_real_data` — they already thread `config` to `build_env`, so `action_mode`/`max_tilt` in the config make them run the tilt model.
- Produces: an integration test that the full RQ1 pipeline runs end-to-end with `action_mode="tilt"` and yields a finite `skill_net` with the expected keys; and a sweep driver that can run the tilt model on the cluster by flipping `ACTION_MODE = "tilt"` (writes to a separate `outputs/rq1_sweep_tilt/` run dir to avoid colliding with the gate sweep's cache).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rq1_real_data_tilt.py
import numpy as np
from src.rq1_real_data import run_rq1


def test_rq1_runs_end_to_end_with_tilt_action(tmp_path):
    # Plumbing check: the RQ1 pipeline runs with the expressive tilt action and
    # produces the same result shape (skill_net + CI + metrics table). On i.i.d.
    # noise there is no timeable structure, so skill_net must not be positively
    # distinguishable from the placebo null (CI straddles ~0) — the tilt action
    # must not manufacture skill from noise any more than the gate did.
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0003, 0.01, size=(900, 4))
    config = {"base_name": "risk_parity", "window": 20, "cost_bps": 10.0,
              "action_mode": "tilt", "max_tilt": 0.15,
              "total_timesteps": 1500, "seed": 0,
              "initial_train": 400, "test_block": 250, "n_placebo": 2}
    result = run_rq1(config, returns=returns, run_dir=tmp_path)

    assert "skill_net" in result and np.isfinite(result["skill_net"])
    assert len(result["skill_net_ci"]) == 2
    assert {"agent", "base", "one_over_n", "min_variance", "cvar_min"}.issubset(result["metrics_table"].keys())
    low, high = result["skill_net_ci"]
    assert low <= 5e-5 and high >= -5e-5                   # no manufactured skill from noise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rq1_real_data_tilt.py -v`
Expected: The pipeline should already accept `action_mode` via config (Task 3). If it errors, the failure identifies the gap; if it passes immediately, that is acceptable (Task 3 wired it) — proceed to add sweep support in Step 3 and keep this as the regression guard.

- [ ] **Step 3: Add tilt support to `scripts/rq1_sweep_task.py`**

Add two constants in the config block (top of the file):

```python
ACTION_MODE = "gate"    # set to "tilt" to sweep the expressive residual-tilt agent
MAX_TILT = 0.15         # tilt-mode cap (ignored in gate mode); calibrate via the tilt RQ2
```

Change `RUN_DIR` to separate tilt runs from gate runs:

```python
RUN_DIR = (Path(__file__).resolve().parent.parent / "outputs"
           / ("rq1_sweep_tilt" if ACTION_MODE == "tilt" else "rq1_sweep"))
```

Add `action_mode`/`max_tilt` to `base_config` in `main()`:

```python
    base_config = {"base_name": base, "window": WINDOW, "cost_bps": COST_BPS,
                   "safe_asset_index": safe_index, "total_timesteps": TOTAL_TIMESTEPS,
                   "initial_train": INITIAL_TRAIN, "test_block": TEST_BLOCK,
                   "action_mode": ACTION_MODE, "max_tilt": MAX_TILT}
```

(Do the same in `scripts/rq1_sweep_aggregate.py`'s `base_cfg` and its imports of the constants, so the aggregator reads the matching cache. Import `ACTION_MODE`, `MAX_TILT`, and the tilt-aware `RUN_DIR` from `scripts.rq1_sweep_task` exactly as it already imports the other constants.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rq1_real_data_tilt.py -v -s`
Expected: 1 passed. (Trains tiny tilt agents across folds + placebo; a few minutes.) Also confirm the gate path is untouched: `uv run python -c "import scripts.rq1_sweep_task as t; print(t.ACTION_MODE, t.RUN_DIR.name)"` prints `gate rq1_sweep`.

- [ ] **Step 5: Commit**

```bash
git add scripts/rq1_sweep_task.py scripts/rq1_sweep_aggregate.py tests/test_rq1_real_data_tilt.py
git commit -m "feat: tilt-model support in the RQ1 pipeline + LSF sweep (separate tilt run dir)"
```

- [ ] **Step 6: (Operational, after RQ2 passes) run the real tilt sweep**

Once the tilt RQ2 gate (Task 4) passes with the chosen `max_tilt`, set `ACTION_MODE = "tilt"` (and `MAX_TILT` to the calibrated value) in `scripts/rq1_sweep_task.py`, then submit the array exactly as the gate sweep (see `scripts/rq1_sweep_task.sh`):

```bash
bsub -J "rq1_sweep_tilt[1-45]" -q normal -n 4 -R "span[hosts=1] rusage[mem=1G]" -M 16G -W 2:00 \
  -o logs/sweep_tilt_%J_%I.out -e logs/sweep_tilt_%J_%I.err bash scripts/rq1_sweep_task.sh
```

Then aggregate: `uv run python -u -m scripts.rq1_sweep_aggregate` → `outputs/rq1_sweep_tilt/summary.json`. Compare the tilt agent's skill_net-by-base against the gate agent's (Plan 2 result). This step produces the paper's headline comparison; it trains RL models but adds no new tested code, so it is operational, not a TDD task.

---

## Plan 3 Self-Review

**Spec coverage:**
- Bounded residual-tilt action, `w=project_simplex(base+max_tilt·tanh(a))`, zero→base (spec §3) → Task 1.
- Enriched, causal tilt-mode observation (momentum + long vol + base weights) (spec §4) → Task 1 (`_observation`), no-lookahead by construction + shape/bound tests.
- Structure-baselined reward unchanged; long-only simplex; do-nothing floor (spec §2) → Task 1 (shared reward core; zero-action test).
- `action_mode` env parameterization; downstream reuse (spec §5) → Task 1 + Task 3 (build_env passthrough) + Task 5 (pipeline/sweep).
- Fair multi-asset RQ2 bed (spec §6 needs a market the tilt can express skill in) → Task 2.
- RQ2 re-validation gate FIRST, `max_tilt` tuned on RQ2 (spec §6, §3) → Task 4 (+ calibration step, no threshold-weakening).
- Real-data RQ1 + sweep net of placebo null, gate reported alongside (spec §6, §7) → Task 5 (+ operational Step 6).
- Testing list (spec §8): zero-action=base, simplex, no-lookahead/enriched-feature shape, RQ2 gate, tilt-beats-base, gate-mode regression → Tasks 1, 3, 4.
- Scope/YAGNI: PPO only, long-only, same universe/costs/walk-forward; MC2 deferred (spec §9) → respected; no new algorithms introduced.

**Placeholder scan:** none — every code step has complete code and exact commands. RL/real-data steps use property/ground-truth/plumbing assertions (correct for stochastic RL); the one hyperparameter (`max_tilt`) has an explicit RQ2-driven calibration procedure rather than a hidden TODO.

**Type consistency:** `AllocationEnv(..., action_mode, max_tilt)` defined in Task 1, consumed by `build_env` in Task 3, threaded via `config` in Tasks 4–5. `info["gate"]` remains present in both modes (gate value / tilt activity), so `evaluate_skill`/`roll_policy` are unchanged. `generate_multi_regime_market(n_risky, n_safe, n_steps, seed, signal_strength)` defined in Task 2, consumed in Tasks 3–4. `config["market"]`/`n_risky`/`n_safe` defined/consumed in Task 4. `ACTION_MODE`/`MAX_TILT`/`RUN_DIR` in Task 5 shared between `rq1_sweep_task.py` and `rq1_sweep_aggregate.py`. Consistent.

---

## Next (not in this plan)
- **MC2 causal probing (RQ3):** intervention harness (vol shocks, regime flips, feature freeze/permute) on the trained agent + SHAP/saliency faithfulness verdict. Deferred; write as its own plan after the expressive-agent verdict lands.
