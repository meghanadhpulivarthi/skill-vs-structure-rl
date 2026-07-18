# MC1 Skill-Isolating Residual RL — Synthetic Ground-Truth Validation (Plan 1 of 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a skill-isolating residual-RL method on synthetic markets where the ground truth is known — proving the skill measure vanishes when no timeable structure exists and reappears when it does (spec RQ2).

**Architecture:** A regime-switching synthetic market emits returns plus a *toggleable* leading signal. A non-learned **structural-null base policy** (constant-rebalanced 1/N and volatility-scaled 1/N) defines the "free structure" floor. A PPO agent outputs a **residual tilt** on the base over the long-only simplex; its reward is the agent's net log-return **minus the base's net log-return on the same path** (structure-baselined credit assignment), so the gradient only rewards skill above structure. The RQ2 experiment trains with the signal ON vs OFF and checks the residual's out-of-sample benefit is ~0 when OFF and positive when ON.

**Tech Stack:** Python 3.11, `uv` (never pip), NumPy, pandas, SciPy, Gymnasium, Stable-Baselines3 (PyTorch), pytest, tqdm, matplotlib.

## Global Constraints

- Package manager is `uv` only: `uv add <pkg>`, `uv run python -u <script>`. Never `pip`. (CLAUDE.md)
- Code style: config constants at the TOP of each script, clearly labelled; `snake_case`; full words (`n_assets` not `na`); flat over nested; 4-space indent. (code-style.md)
- No silent failures: no bare `except`; every `continue`/`break` logged; `dict[key]` not `.get()` unless absence is explicitly handled; no `None` flowing downstream unguarded. (code-style.md)
- No absolute paths in any script; scripts runnable from any working directory (resolve paths relative to the file). (code-style.md)
- Traceability: every training/experiment script prints a run header (timestamp, script, config), writes to `outputs/YYYY-MM-DD_HH-MM-SS_<desc>/`, saves `config.json` and `results.json`, and never overwrites an old output dir. Wrap major loops in `tqdm`; run with `python -u`. (traceability.md)
- Determinism: every stochastic function takes an explicit integer `seed`; no reliance on global RNG state.
- All portfolio weights are long-only on the simplex: `w >= 0`, `sum(w) == 1`.

---

### Task 1: Project scaffolding (git, uv, dependencies, layout)

**Files:**
- Create: `pyproject.toml` (via `uv init`)
- Create: `src/__init__.py`, `tests/__init__.py`
- Create: `.gitignore`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: a runnable `uv` project with `pytest` available; `src/` importable as a package; git repo so later `git commit` steps work.

- [ ] **Step 1: Initialize git and uv project**

This directory is not yet a git repo (frequent-commit discipline needs one). Run from the project root:

```bash
cd /dccstor/meghanadhp/projects/Helix/rl-allocation-audit
git init
uv init --python 3.11 --no-workspace
```

- [ ] **Step 2: Add dependencies**

```bash
uv add numpy pandas scipy gymnasium "stable-baselines3[extra]" torch matplotlib tqdm
uv add --dev pytest
```

- [ ] **Step 3: Create package layout and .gitignore**

Create `src/__init__.py` (empty) and `tests/__init__.py` (empty). Create `.gitignore`:

```gitignore
__pycache__/
*.pyc
.venv/
outputs/
.pytest_cache/
```

`outputs/` is git-ignored because runs are large and reproducible from config; the `outputs/` dir itself already exists in the repo.

- [ ] **Step 4: Verify the toolchain**

Run: `uv run python -u -c "import numpy, pandas, scipy, gymnasium, stable_baselines3, torch, matplotlib, tqdm; print('deps ok')"`
Expected: prints `deps ok` with no import error.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/__init__.py tests/__init__.py .gitignore
git commit -m "chore: scaffold uv project and dependencies"
```

---

### Task 2: Simplex projection utility

**Files:**
- Create: `src/simplex.py`
- Test: `tests/test_simplex.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `project_to_simplex(v: np.ndarray) -> np.ndarray` — Euclidean projection of a length-`n` vector onto the probability simplex (`w >= 0`, `sum(w) == 1`). Used by the base policies (Task 5) and the RL env (Task 6) to keep weights long-only.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_simplex.py
import numpy as np
from src.simplex import project_to_simplex


def test_already_on_simplex_is_unchanged():
    w = np.array([0.2, 0.3, 0.5])
    out = project_to_simplex(w)
    np.testing.assert_allclose(out, w, atol=1e-9)


def test_projection_sums_to_one_and_nonnegative():
    v = np.array([3.0, -1.0, 0.5, 2.0])
    out = project_to_simplex(v)
    assert np.all(out >= -1e-12)
    np.testing.assert_allclose(out.sum(), 1.0, atol=1e-9)


def test_negative_inputs_are_clipped_to_zero_mass():
    # a very negative entry should receive zero weight
    v = np.array([1.0, 1.0, -50.0])
    out = project_to_simplex(v)
    assert out[2] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_simplex.py -v`
Expected: FAIL with `ModuleNotFoundError` / `cannot import name 'project_to_simplex'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/simplex.py
"""Euclidean projection onto the probability simplex (Duchi et al. 2008)."""
import numpy as np


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    # Why this algorithm: exact O(n log n) Euclidean projection; keeps the
    # residual action's geometry meaningful (nearest long-only portfolio to
    # base + tilt) rather than an arbitrary softmax squashing.
    v = np.asarray(v, dtype=float)
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not cond.any():
        # Degenerate input (e.g. all -inf); fall back to uniform and log it.
        print("project_to_simplex: no positive support found; returning uniform")
        return np.full(n, 1.0 / n)
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(v - theta, 0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_simplex.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/simplex.py tests/test_simplex.py
git commit -m "feat: simplex projection utility"
```

---

### Task 3: Risk & performance metrics

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: nothing.
- Produces (all take a 1-D array of per-period *simple* returns unless noted):
  - `expected_shortfall(returns: np.ndarray, alpha: float = 0.99) -> float` — mean of the worst `(1-alpha)` fraction of returns (a negative number for a loss).
  - `max_drawdown(returns: np.ndarray) -> float` — most negative peak-to-trough of the cumulative-return curve (negative number).
  - `tail_ratio(returns: np.ndarray) -> float` — `|95th pctile| / |5th pctile|`.
  - `skewness(returns: np.ndarray) -> float`.
  - `sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float`.
  - `sortino(returns: np.ndarray, periods_per_year: int = 252) -> float`.
  - `turnover(weights: np.ndarray) -> float` — mean per-step `0.5 * sum|w_t - w_{t-1}|` over a `(T, n_assets)` weight path.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import numpy as np
from src import metrics


def test_expected_shortfall_picks_worst_tail():
    r = np.array([-0.10, -0.05, 0.0, 0.01, 0.02] * 20)  # 100 obs
    es = metrics.expected_shortfall(r, alpha=0.99)  # worst 1% => the single -0.10
    assert es <= -0.09


def test_max_drawdown_is_negative_for_a_dip():
    r = np.array([0.1, -0.5, 0.1])
    assert metrics.max_drawdown(r) < -0.4


def test_sharpe_zero_mean_is_zero():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.01, size=10_000)
    assert abs(metrics.sharpe(r)) < 0.15


def test_turnover_zero_for_constant_weights():
    w = np.tile(np.array([0.5, 0.5]), (10, 1))
    assert metrics.turnover(w) == 0.0


def test_turnover_full_switch():
    w = np.array([[1.0, 0.0], [0.0, 1.0]])
    # 0.5 * (|0-1| + |1-0|) = 1.0 on the single transition
    assert abs(metrics.turnover(w) - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL with import error on `src.metrics`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/metrics.py
"""Risk/performance metrics on per-period simple-return arrays."""
import numpy as np
from scipy import stats


def expected_shortfall(returns: np.ndarray, alpha: float = 0.99) -> float:
    returns = np.asarray(returns, dtype=float)
    tail_prob = 1.0 - alpha
    cutoff = max(1, int(np.floor(tail_prob * returns.size)))
    worst = np.sort(returns)[:cutoff]
    return float(worst.mean())


def max_drawdown(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    wealth = np.cumprod(1.0 + returns)
    running_peak = np.maximum.accumulate(wealth)
    drawdown = wealth / running_peak - 1.0
    return float(drawdown.min())


def tail_ratio(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    right = abs(np.percentile(returns, 95))
    left = abs(np.percentile(returns, 5))
    if left == 0.0:
        print("tail_ratio: left tail is zero; returning nan")
        return float("nan")
    return float(right / left)


def skewness(returns: np.ndarray) -> float:
    return float(stats.skew(np.asarray(returns, dtype=float)))


def sharpe(returns: np.ndarray, periods_per_year: int = 252) -> float:
    returns = np.asarray(returns, dtype=float)
    std = returns.std(ddof=1)
    if std == 0.0:
        print("sharpe: zero volatility; returning 0.0")
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def sortino(returns: np.ndarray, periods_per_year: int = 252) -> float:
    returns = np.asarray(returns, dtype=float)
    downside = returns[returns < 0.0]
    if downside.size == 0:
        print("sortino: no downside returns; returning inf")
        return float("inf")
    downside_std = np.sqrt(np.mean(downside ** 2))
    return float(returns.mean() / downside_std * np.sqrt(periods_per_year))


def turnover(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    if weights.shape[0] < 2:
        return 0.0
    step_turnover = 0.5 * np.abs(np.diff(weights, axis=0)).sum(axis=1)
    return float(step_turnover.mean())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: risk and performance metrics"
```

---

### Task 4: Synthetic regime-switching market with toggleable signal

**Files:**
- Create: `src/synthetic_market.py`
- Test: `tests/test_synthetic_market.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `generate_market(n_assets: int, n_steps: int, seed: int, signal_strength: float, crisis_persistence: float = 0.9, calm_persistence: float = 0.98) -> dict` returning:
  - `"returns"`: `(n_steps, n_assets)` simple returns,
  - `"signal"`: `(n_steps,)` leading indicator in `[0, 1]` (predictive of the *next* step's crisis probability only when `signal_strength > 0`; pure noise when `signal_strength == 0`),
  - `"regime"`: `(n_steps,)` int array (0 = calm, 1 = crisis).
  The ground-truth knob: `signal_strength == 0.0` ⇒ no timeable structure ⇒ skill is impossible.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synthetic_market.py
import numpy as np
from src.synthetic_market import generate_market


def test_shapes_and_keys():
    m = generate_market(n_assets=5, n_steps=1000, seed=1, signal_strength=0.8)
    assert m["returns"].shape == (1000, 5)
    assert m["signal"].shape == (1000,)
    assert m["regime"].shape == (1000,)
    assert set(np.unique(m["regime"])).issubset({0, 1})


def test_crisis_has_higher_volatility_than_calm():
    m = generate_market(n_assets=5, n_steps=20_000, seed=2, signal_strength=0.8)
    calm_vol = m["returns"][m["regime"] == 0].std()
    crisis_vol = m["returns"][m["regime"] == 1].std()
    assert crisis_vol > 1.5 * calm_vol


def test_signal_predicts_next_regime_only_when_strength_positive():
    # With signal, current signal should correlate with NEXT-step crisis.
    m_on = generate_market(n_assets=3, n_steps=20_000, seed=3, signal_strength=0.9)
    next_crisis = m_on["regime"][1:]
    corr_on = np.corrcoef(m_on["signal"][:-1], next_crisis)[0, 1]
    assert corr_on > 0.2

    m_off = generate_market(n_assets=3, n_steps=20_000, seed=3, signal_strength=0.0)
    corr_off = abs(np.corrcoef(m_off["signal"][:-1], m_off["regime"][1:])[0, 1])
    assert corr_off < 0.05


def test_determinism():
    a = generate_market(n_assets=4, n_steps=500, seed=7, signal_strength=0.5)
    b = generate_market(n_assets=4, n_steps=500, seed=7, signal_strength=0.5)
    np.testing.assert_array_equal(a["returns"], b["returns"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_synthetic_market.py -v`
Expected: FAIL with import error on `src.synthetic_market`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/synthetic_market.py
"""Two-regime (calm/crisis) synthetic market with a toggleable leading signal.

Ground-truth design: the ONLY way an agent can reduce tail risk beyond the
structural base is by using `signal` to anticipate crises. Setting
`signal_strength = 0` removes that information, so any residual "skill" a method
reports in that setting is an artifact, not skill. This is the RQ2 test bed.
"""
import numpy as np

# Config — regime return/vol parameters (daily-like scale)
CALM_DRIFT = 0.0004
CALM_VOL = 0.008
CRISIS_DRIFT = -0.0015
CRISIS_VOL = 0.030
CALM_CORR = 0.2      # cross-asset correlation in calm regime
CRISIS_CORR = 0.7    # correlations spike in crises (diversification fails)


def _regime_cov(n_assets: int, vol: float, corr: float) -> np.ndarray:
    cov = np.full((n_assets, n_assets), corr)
    np.fill_diagonal(cov, 1.0)
    return cov * (vol ** 2)


def generate_market(
    n_assets: int,
    n_steps: int,
    seed: int,
    signal_strength: float,
    crisis_persistence: float = 0.9,
    calm_persistence: float = 0.98,
) -> dict:
    rng = np.random.default_rng(seed)

    # 1) Simulate the hidden regime path as a 2-state Markov chain.
    regime = np.zeros(n_steps, dtype=int)
    for t in range(1, n_steps):
        if regime[t - 1] == 0:
            regime[t] = 0 if rng.random() < calm_persistence else 1
        else:
            regime[t] = 1 if rng.random() < crisis_persistence else 0

    # 2) Leading signal: informative about NEXT regime iff signal_strength>0.
    #    signal_t = strength * 1[regime_{t+1}=crisis] + (1-strength)*noise.
    noise = rng.random(n_steps)
    next_is_crisis = np.zeros(n_steps)
    next_is_crisis[:-1] = (regime[1:] == 1).astype(float)
    signal = signal_strength * next_is_crisis + (1.0 - signal_strength) * noise

    # 3) Asset returns, regime-dependent mean & covariance.
    calm_cov = _regime_cov(n_assets, CALM_VOL, CALM_CORR)
    crisis_cov = _regime_cov(n_assets, CRISIS_VOL, CRISIS_CORR)
    returns = np.zeros((n_steps, n_assets))
    for t in range(n_steps):
        if regime[t] == 0:
            returns[t] = rng.multivariate_normal(np.full(n_assets, CALM_DRIFT), calm_cov)
        else:
            returns[t] = rng.multivariate_normal(np.full(n_assets, CRISIS_DRIFT), crisis_cov)

    return {"returns": returns, "signal": signal, "regime": regime}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_synthetic_market.py -v`
Expected: 4 passed. (The correlation thresholds are loose on purpose; if `test_signal_predicts...` is flaky, raise `n_steps`, do not loosen below 0.2/0.05.)

- [ ] **Step 5: Commit**

```bash
git add src/synthetic_market.py tests/test_synthetic_market.py
git commit -m "feat: regime-switching synthetic market with toggleable signal"
```

---

### Task 5: Structural-null base policies

**Files:**
- Create: `src/base_policies.py`
- Test: `tests/test_base_policies.py`

**Interfaces:**
- Consumes: `project_to_simplex` (Task 2).
- Produces two callables, each mapping a trailing return window to next-step target weights:
  - `equal_weight_base(return_window: np.ndarray) -> np.ndarray` — constant `1/n` (harvests the baseline rebalancing premium).
  - `vol_scaled_base(return_window: np.ndarray, target_vol: float = 0.01) -> np.ndarray` — inverse-volatility weights across assets, projected to the simplex (adds the vol-targeting de-risking channel). `return_window` is `(window, n_assets)`.
  - `BASE_POLICIES: dict[str, callable]` mapping `"equal_weight"` and `"vol_scaled"` to the two functions, so the env (Task 6) selects a base by name.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_base_policies.py
import numpy as np
from src.base_policies import equal_weight_base, vol_scaled_base, BASE_POLICIES


def test_equal_weight_is_uniform_and_on_simplex():
    win = np.zeros((20, 4))
    w = equal_weight_base(win)
    np.testing.assert_allclose(w, np.full(4, 0.25))
    np.testing.assert_allclose(w.sum(), 1.0)


def test_vol_scaled_downweights_high_vol_asset():
    rng = np.random.default_rng(0)
    win = np.column_stack([
        rng.normal(0, 0.005, 60),   # low-vol asset
        rng.normal(0, 0.05, 60),    # high-vol asset
    ])
    w = vol_scaled_base(win)
    assert w[0] > w[1]                      # low-vol gets more weight
    np.testing.assert_allclose(w.sum(), 1.0)
    assert np.all(w >= 0)


def test_registry_contains_both():
    assert set(BASE_POLICIES.keys()) == {"equal_weight", "vol_scaled"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_base_policies.py -v`
Expected: FAIL with import error on `src.base_policies`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/base_policies.py
"""Non-learned structural-null base policies (the 'free structure' floor).

These already capture the inherited-structure channels the residual agent must
beat: equal_weight harvests the rebalancing premium; vol_scaled adds volatility
targeting (mechanical de-risking). A strong base is essential — a weak base
inflates the apparent skill of the residual (spec §11).
"""
import numpy as np
from src.simplex import project_to_simplex


def equal_weight_base(return_window: np.ndarray) -> np.ndarray:
    n_assets = return_window.shape[1]
    return np.full(n_assets, 1.0 / n_assets)


def vol_scaled_base(return_window: np.ndarray, target_vol: float = 0.01) -> np.ndarray:
    # Inverse-volatility weights over the trailing window. target_vol is kept in
    # the signature for interface stability with later exposure-scaling work;
    # cross-asset weights are scale-free so it does not change relative weights.
    vol = return_window.std(axis=0)
    vol = np.where(vol <= 1e-8, 1e-8, vol)  # guard divide-by-zero, no silent nan
    inv_vol = 1.0 / vol
    return project_to_simplex(inv_vol / inv_vol.sum())


BASE_POLICIES = {
    "equal_weight": equal_weight_base,
    "vol_scaled": vol_scaled_base,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_base_policies.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/base_policies.py tests/test_base_policies.py
git commit -m "feat: structural-null base policies (equal-weight, vol-scaled)"
```

---

### Task 6: Allocation environment (residual action + structure-baselined reward)

**Files:**
- Create: `src/allocation_env.py`
- Test: `tests/test_allocation_env.py`

**Interfaces:**
- Consumes: `project_to_simplex` (Task 2), `BASE_POLICIES` (Task 5).
- Produces: `AllocationEnv(gymnasium.Env)` with constructor
  `AllocationEnv(market: dict, base_name: str, window: int = 20, cost_bps: float = 10.0)`.
  - Observation: concatenation of the flattened trailing return window `(window * n_assets,)`, per-asset trailing vol `(n_assets,)`, and the current `signal` scalar → `Box` of shape `(window*n_assets + n_assets + 1,)`.
  - Action: `Box(-inf, inf, (n_assets,))` — the residual tilt; final weights `w = project_to_simplex(base_weights + action)`.
  - Reward: **structure-baselined** — `r_t = (log(1 + w·ret_t) - cost) - (log(1 + base·ret_t) - base_cost)`. The agent is credited only for beating the base *after* paying for its extra turnover.
  - Exposes `self.last_info` dict per step with keys `"weights"`, `"base_weights"`, `"port_return"`, `"base_return"` for evaluation (Task 8).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_allocation_env.py
import numpy as np
from src.synthetic_market import generate_market
from src.allocation_env import AllocationEnv


def _make_env(base_name="vol_scaled"):
    market = generate_market(n_assets=4, n_steps=300, seed=5, signal_strength=0.8)
    return AllocationEnv(market, base_name=base_name, window=20, cost_bps=10.0)


def test_reset_returns_obs_of_declared_shape():
    env = _make_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape


def test_zero_action_reproduces_base_and_gives_zero_reward():
    # residual = 0 => weights == base => structure-baselined reward == 0.
    env = _make_env()
    env.reset(seed=0)
    action = np.zeros(env.action_space.shape[0])
    _, reward, _, _, info = env.step(action)
    np.testing.assert_allclose(info["weights"], info["base_weights"], atol=1e-8)
    assert abs(reward) < 1e-8


def test_weights_stay_on_simplex_for_arbitrary_action():
    env = _make_env()
    env.reset(seed=0)
    rng = np.random.default_rng(1)
    for _ in range(50):
        _, _, term, trunc, info = env.step(rng.normal(size=env.action_space.shape[0]))
        w = info["weights"]
        assert np.all(w >= -1e-9)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-8)
        if term or trunc:
            break


def test_episode_terminates_at_end_of_series():
    env = _make_env()
    env.reset(seed=0)
    steps = 0
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(np.zeros(env.action_space.shape[0]))
        done = term or trunc
        steps += 1
    assert steps == 300 - 20  # n_steps - window
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_allocation_env.py -v`
Expected: FAIL with import error on `src.allocation_env`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/allocation_env.py
"""Gymnasium env: PPO learns a residual tilt on a structural-null base policy.

Structure-baselined reward is the methodological core (spec §5.3): the agent is
rewarded ONLY for the log-growth it adds over the base on the same market path,
net of its extra turnover cost. With no timeable signal, positive expected
reward is unattainable, so the learned tilt should collapse toward the base.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.simplex import project_to_simplex
from src.base_policies import BASE_POLICIES


class AllocationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, market: dict, base_name: str, window: int = 20, cost_bps: float = 10.0):
        super().__init__()
        if base_name not in BASE_POLICIES:
            raise ValueError(f"unknown base_name {base_name!r}; expected one of {list(BASE_POLICIES)}")
        self.returns = np.asarray(market["returns"], dtype=float)
        self.signal = np.asarray(market["signal"], dtype=float)
        self.n_steps, self.n_assets = self.returns.shape
        self.base_policy = BASE_POLICIES[base_name]
        self.window = window
        self.cost_rate = cost_bps * 1e-4

        obs_dim = window * self.n_assets + self.n_assets + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-np.inf, np.inf, (self.n_assets,), dtype=np.float32)

        self._t = None
        self._prev_weights = None
        self._prev_base = None
        self.last_info = None

    def _observation(self) -> np.ndarray:
        win = self.returns[self._t - self.window:self._t]
        vol = win.std(axis=0)
        obs = np.concatenate([win.flatten(), vol, [self.signal[self._t]]])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t = self.window
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
        weights = project_to_simplex(base_weights + np.asarray(action, dtype=float))

        asset_returns = self.returns[self._t]
        agent_log = self._net_log_return(weights, self._prev_weights, asset_returns)
        base_log = self._net_log_return(base_weights, self._prev_base, asset_returns)
        reward = agent_log - base_log  # structure-baselined credit

        self.last_info = {
            "weights": weights,
            "base_weights": base_weights,
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

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_allocation_env.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/allocation_env.py tests/test_allocation_env.py
git commit -m "feat: allocation env with residual action and structure-baselined reward"
```

---

### Task 7: Training entrypoint (PPO) with traceable outputs

**Files:**
- Create: `src/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `AllocationEnv` (Task 6).
- Produces:
  - `build_env(market: dict, config: dict) -> AllocationEnv` — thin factory reading `base_name`, `window`, `cost_bps` from `config`.
  - `train_agent(market: dict, config: dict) -> stable_baselines3.PPO` — trains PPO on the env for `config["total_timesteps"]`, seeded by `config["seed"]`; prints the run header; returns the trained model.
  - Running `uv run python -u -m src.train` executes a small default run and writes `outputs/<ts>_train-smoke/{config.json,results.json}`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py
import numpy as np
from src.synthetic_market import generate_market
from src.train import build_env, train_agent


def test_smoke_training_runs_and_beats_base_with_signal():
    # With a strong signal, a briefly-trained agent should earn positive mean
    # structure-baselined reward on a fresh episode (i.e. beat the base).
    market = generate_market(n_assets=4, n_steps=4000, seed=11, signal_strength=0.95)
    config = {"base_name": "vol_scaled", "window": 20, "cost_bps": 10.0,
              "total_timesteps": 20_000, "seed": 0}
    model = train_agent(market, config)

    eval_market = generate_market(n_assets=4, n_steps=4000, seed=12, signal_strength=0.95)
    env = build_env(eval_market, config)
    obs, _ = env.reset(seed=0)
    rewards = []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, _ = env.step(action)
        rewards.append(reward)
        done = term or trunc
    assert np.isfinite(rewards).all()          # no NaNs (stability)
    assert np.mean(rewards) > 0.0               # adds skill when signal exists
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_train.py -v`
Expected: FAIL with import error on `src.train`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/train.py
"""Train residual-PPO on the allocation env. Traceable, seeded, config-driven."""
import os
import json
import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.synthetic_market import generate_market
from src.allocation_env import AllocationEnv

# Config — edit these directly for the default smoke run
DEFAULT_CONFIG = {
    "base_name": "vol_scaled",
    "window": 20,
    "cost_bps": 10.0,
    "total_timesteps": 20_000,
    "seed": 0,
    "n_assets": 4,
    "n_steps": 4000,
    "signal_strength": 0.95,
}


def build_env(market: dict, config: dict) -> AllocationEnv:
    return AllocationEnv(
        market,
        base_name=config["base_name"],
        window=config["window"],
        cost_bps=config["cost_bps"],
    )


def train_agent(market: dict, config: dict) -> PPO:
    env = Monitor(build_env(market, config))
    model = PPO("MlpPolicy", env, seed=config["seed"], verbose=0)
    model.learn(total_timesteps=config["total_timesteps"], progress_bar=True)
    return model


def _run_default():
    config = dict(DEFAULT_CONFIG)
    now = datetime.datetime.now()
    print("=" * 60)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Config      : {config}")
    print("=" * 60)

    market = generate_market(config["n_assets"], config["n_steps"],
                             seed=config["seed"], signal_strength=config["signal_strength"])
    model = train_agent(market, config)

    out_dir = Path(__file__).resolve().parent.parent / "outputs" / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_train-smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    model.save(out_dir / "ppo_model")
    with open(out_dir / "results.json", "w") as f:
        json.dump({"status": "trained", "model_path": str(out_dir / "ppo_model.zip")}, f, indent=2)
    print(f"Saved run to: {out_dir}")


if __name__ == "__main__":
    _run_default()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_train.py -v -s`
Expected: 1 passed. (Runs ~20k PPO steps; up to a couple minutes on CPU, seconds on GPU. If `mean(rewards) > 0` is flaky, raise `total_timesteps` to 50_000 — do NOT weaken the assertion; a method that can't beat the base *with* a near-perfect signal is broken.)

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat: PPO training entrypoint with traceable outputs"
```

---

### Task 8: RQ2 skill-validation experiment (signal ON vs OFF)

**Files:**
- Create: `src/validate_skill.py`
- Test: `tests/test_validate_skill.py`

**Interfaces:**
- Consumes: `generate_market` (Task 4), `build_env` + `train_agent` (Task 7), `metrics` (Task 3).
- Produces:
  - `evaluate_skill(model, market: dict, config: dict) -> dict` — rolls the deterministic policy over `market`, returns `{"mean_baselined_reward", "residual_turnover", "es_agent", "es_base", "es_reduction"}` where `es_*` use `expected_shortfall` on realized `port_return`/`base_return` and `es_reduction = es_agent - es_base` (positive = agent has a *less negative*, i.e. better, tail).
  - `run_skill_validation(config: dict, signal_strengths=(0.0, 0.95), n_seeds=5) -> dict` — for each signal strength, trains on a train-seed market and evaluates on held-out seed markets; aggregates mean baselined reward across seeds. Writes a traceable output dir with `config.json` + `results.json` + a bar plot `skill_vs_signal.png`.
  - Running `uv run python -u -m src.validate_skill` runs the full RQ2 experiment.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validate_skill.py
from src.validate_skill import run_skill_validation


def test_skill_vanishes_without_signal_and_appears_with_signal():
    # THE RQ2 GROUND-TRUTH TEST. Skill measure (mean structure-baselined reward
    # on held-out data) must be ~0 with no timeable signal, and clearly positive
    # when the signal exists. This is the paper's core validity claim.
    config = {"base_name": "vol_scaled", "window": 20, "cost_bps": 10.0,
              "total_timesteps": 30_000, "seed": 0, "n_assets": 4, "n_steps": 4000}
    result = run_skill_validation(config, signal_strengths=(0.0, 0.95), n_seeds=3)

    skill_off = result["by_strength"]["0.0"]["mean_baselined_reward"]
    skill_on = result["by_strength"]["0.95"]["mean_baselined_reward"]

    assert skill_off < 5e-5      # no timeable structure => ~no skill (allow tiny noise)
    assert skill_on > 2e-4        # timeable structure => measurable skill
    assert skill_on > 3 * abs(skill_off)  # signal-driven skill dominates noise floor
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_validate_skill.py -v`
Expected: FAIL with import error on `src.validate_skill`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/validate_skill.py
"""RQ2: validate the skill measure against synthetic ground truth.

The structure-baselined mean reward on held-out markets IS the operational skill
measure. It must vanish when signal_strength=0 (no timeable structure => skill
impossible) and be positive when signal_strength>0. Passing this calibrates the
measure so real-data results (Plan 2) are interpretable.
"""
import json
import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.synthetic_market import generate_market
from src.train import build_env, train_agent
from src.metrics import expected_shortfall


def evaluate_skill(model, market: dict, config: dict) -> dict:
    env = build_env(market, config)
    obs, _ = env.reset(seed=0)
    baselined_rewards, port_returns, base_returns, weights_path = [], [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        baselined_rewards.append(reward)
        port_returns.append(info["port_return"])
        base_returns.append(info["base_return"])
        weights_path.append(info["weights"])
        done = term or trunc

    es_agent = expected_shortfall(np.array(port_returns))
    es_base = expected_shortfall(np.array(base_returns))
    turnover_path = 0.5 * np.abs(np.diff(np.array(weights_path), axis=0)).sum(axis=1)
    return {
        "mean_baselined_reward": float(np.mean(baselined_rewards)),
        "residual_turnover": float(turnover_path.mean()),
        "es_agent": es_agent,
        "es_base": es_base,
        "es_reduction": float(es_agent - es_base),
    }


def run_skill_validation(config: dict, signal_strengths=(0.0, 0.95), n_seeds: int = 5) -> dict:
    now = datetime.datetime.now()
    print("=" * 60)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Config      : {config} | strengths={signal_strengths} | n_seeds={n_seeds}")
    print("=" * 60)

    by_strength = {}
    for strength in signal_strengths:
        seed_rewards = []
        for seed in tqdm(range(n_seeds), desc=f"signal={strength}"):
            train_market = generate_market(config["n_assets"], config["n_steps"],
                                           seed=1000 + seed, signal_strength=strength)
            model = train_agent(train_market, {**config, "seed": seed})
            eval_market = generate_market(config["n_assets"], config["n_steps"],
                                          seed=2000 + seed, signal_strength=strength)
            seed_rewards.append(evaluate_skill(model, eval_market, config)["mean_baselined_reward"])
        by_strength[str(strength)] = {
            "mean_baselined_reward": float(np.mean(seed_rewards)),
            "std_baselined_reward": float(np.std(seed_rewards)),
            "per_seed": [float(x) for x in seed_rewards],
        }
        print(f"signal={strength}: mean skill = {by_strength[str(strength)]['mean_baselined_reward']:.6f}")

    out_dir = Path(__file__).resolve().parent.parent / "outputs" / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_skill-validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump({"config": config, "signal_strengths": list(signal_strengths), "n_seeds": n_seeds}, f, indent=2)
    result = {"by_strength": by_strength}
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    labels = [str(s) for s in signal_strengths]
    means = [by_strength[s_label]["mean_baselined_reward"] for s_label in labels]
    plt.figure()
    plt.bar(labels, means)
    plt.xlabel("signal_strength (timeable structure)")
    plt.ylabel("mean structure-baselined reward (skill)")
    plt.title("RQ2: skill vanishes without timeable structure")
    plt.savefig(out_dir / "skill_vs_signal.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved run to: {out_dir}")
    return result


if __name__ == "__main__":
    _config = {"base_name": "vol_scaled", "window": 20, "cost_bps": 10.0,
               "total_timesteps": 50_000, "seed": 0, "n_assets": 4, "n_steps": 6000}
    run_skill_validation(_config, signal_strengths=(0.0, 0.5, 0.95), n_seeds=5)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_validate_skill.py -v -s`
Expected: 1 passed. (Trains several small agents; minutes on CPU, faster on GPU. If flaky, raise `total_timesteps`/`n_seeds`; do NOT weaken the ground-truth thresholds — they encode the paper's central validity claim per Rule 9.)

- [ ] **Step 5: Run the full experiment and inspect the artifact**

Run: `uv run python -u -m src.validate_skill`
Expected: prints per-strength skill; writes `outputs/<ts>_skill-validation/` containing `results.json` and `skill_vs_signal.png` showing skill rising with `signal_strength` and ~0 at 0.0.

- [ ] **Step 6: Commit**

```bash
git add src/validate_skill.py tests/test_validate_skill.py
git commit -m "feat: RQ2 skill-validation experiment on synthetic ground truth"
```

---

## Plan 1 Self-Review

**Spec coverage (Plan-1 scope):**
- Structural-null base (spec §5.1) → Task 5 (equal-weight + vol-scaled ladder). SPT base deferred to Plan 2 per spec §5.1 decision.
- Residual policy on simplex (§5.2) → Tasks 2 (projection) + 6 (env action).
- Structure-baselined credit assignment (§5.3) → Task 6 reward (agent net log-return − base net log-return).
- Synthetic ground truth + toggleable signal (§6) → Task 4.
- RQ2 validation (§4, §9) → Task 8.
- Metrics matched to literature (§8) → Task 3 (ES, max drawdown, tail ratio, skew, Sharpe, Sortino, turnover).
- Traceability / uv / no-silent-failure constraints → embedded in Tasks 1, 7, 8 and Global Constraints.
- **Deferred to later plans (correctly out of Plan-1 scope):** real ETF data + walk-forward + literature baselines (Plan 2, RQ1); causal probing (Plan 3, RQ3); SPT/growth-optimal base (Plan 2).

**Placeholder scan:** none — every step has runnable code, exact commands, and expected output. Training/experiment steps use smoke/property assertions (finiteness, sign, ground-truth thresholds) rather than exact-value asserts, which is the correct verification for stochastic RL.

**Type consistency:** `project_to_simplex` (Task 2) consumed identically in Tasks 5–6. `BASE_POLICIES` keys `"equal_weight"`/`"vol_scaled"` defined in Task 5, referenced in Tasks 6–7. `build_env`/`train_agent` signatures defined in Task 7, reused in Task 8. `AllocationEnv.last_info` keys (`weights`, `base_weights`, `port_return`, `base_return`) defined in Task 6, consumed in Task 8. Metric names in Task 3 match calls in Task 8. Consistent.

---

## Next plans (not yet written)
- **Plan 2 — Real-data demonstration (RQ1):** yfinance multi-asset ETF pipeline, anchored walk-forward, turnover costs, SPT/growth-optimal base added to the ladder, literature baselines (1/N, min-var, RMZ/KP), and the skill-vs-structure verdict on real markets. Write after Plan 1 executes.
- **Plan 3 — MC2 causal probing (RQ3):** intervention harness (vol shocks, regime flips, feature freeze/permute) on the trained agent + SHAP/saliency comparison and causal-faithfulness verdict. Write last; staged so a slip degrades to MC1-only.
