# MC2 Causal-Probing (RQ3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a causal-probing harness that tests whether post-hoc attribution (SHAP / gradient saliency) faithfully identifies the mechanism driving a trained gate agent's de-risking, adjudicated against synthetic ground truth (the known `signal` driver).

**Architecture:** Three new `src/` modules — `interventions.py` (causal track: deterministic replay, feature-group ablation, environment interventions), `attribution.py` (post-hoc track: gradient saliency + KernelSHAP), and `rq3_faithfulness.py` (the experiment: train gate agents, run both tracks over the same decisions, compute the faithfulness verdict). Attribution/intervention functions operate on injected callables (`gate_fn`, `gate_mean_fn`) so they can be tested on a known linear policy and reused on the real PPO agent via thin adapters.

**Tech Stack:** Python 3.11, `uv`, NumPy, SciPy (`spearmanr`), PyTorch (via Stable-Baselines3), `shap` (new dependency), Gymnasium, pytest.

## Global Constraints

- Package manager is `uv` only — add deps with `uv add`, run with `uv run`. Never `pip`.
- Run entrypoints as modules: `uv run python -m src.rq3_faithfulness` (absolute `from src...` imports fail as scripts).
- No absolute paths in code; derive output dirs from `Path(__file__).resolve().parent.parent`.
- Traceability: the experiment entrypoint prints a run header (timestamp, script, config) and writes a timestamped `outputs/<ts>_rq3-faithfulness/` with `config.json`, `results.json`, and figures.
- Fail loud: no bare `except`, no silent `continue`/`break`, no `.get()` whose `None` flows downstream. Raise `ValueError` on unknown modes/args.
- Tests encode intent (Rule 9). The attribution instrument must pass a known-answer calibration (linear policy → importance ∝ |w|) before it is trusted; do NOT weaken ground-truth thresholds to make a stochastic run pass — recalibrate budget/strength instead.
- The gate agent is on the fixed 2-asset risky+safe market: `window=20`, `n_assets=2`, so obs is 43-dim `[40 returns | 2 short_vol | 1 signal]`; the `signal` feature (index 42) is the known ground-truth driver.
- Reuse existing code unchanged: `src/train.py` (`build_env`, `train_agent`), `src/allocation_env.py`, `src/synthetic_market.py` (`generate_risky_safe_market`).

---

### Task 1: Feature groups, deterministic replay, and the numpy gate adapter

**Files:**
- Create: `src/interventions.py`
- Test: `tests/test_interventions.py`

**Interfaces:**
- Consumes: `src.train.build_env(market, config) -> AllocationEnv`; `model.predict(obs, deterministic=True) -> (action, state)` (SB3 or any duck-typed stub).
- Produces:
  - `feature_groups(window: int, n_assets: int) -> dict[str, list[int]]` — keys `"returns"`, `"short_vol"`, `"signal"`.
  - `rollout_observations(model, market: dict, config: dict) -> np.ndarray` — shape `[T, obs_dim]`, the observations that drove each deterministic decision.
  - `make_gate_fn(model) -> Callable[[np.ndarray], np.ndarray]` — maps a `[T, obs_dim]` (or 1-D) observation stack to gate values in `[0,1]`, clipped exactly as the env clips.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_interventions.py
import numpy as np
from src.interventions import feature_groups, rollout_observations, make_gate_fn


def test_feature_groups_partition_the_obs_vector():
    groups = feature_groups(window=20, n_assets=2)
    assert groups["returns"] == list(range(0, 40))
    assert groups["short_vol"] == [40, 41]
    assert groups["signal"] == [42]
    # groups must exactly partition the 43-dim gate obs (no overlap, full cover)
    all_idx = groups["returns"] + groups["short_vol"] + groups["signal"]
    assert sorted(all_idx) == list(range(43))


class _ConstGate:
    # minimal duck-typed stand-in for an SB3 model: gate = clip(mean of obs, 0, 1)
    def predict(self, obs, deterministic=True):
        obs = np.atleast_2d(np.asarray(obs, dtype=np.float32))
        return obs.mean(axis=1, keepdims=True), None


def test_rollout_and_gate_fn_shapes_and_clipping():
    from src.synthetic_market import generate_risky_safe_market
    market = generate_risky_safe_market(300, seed=1, signal_strength=0.95)
    config = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
              "safe_asset_index": 1, "action_mode": "gate"}
    model = _ConstGate()
    obs_stack = rollout_observations(model, market, config)
    # one observation per decision: T = n_steps - window
    assert obs_stack.shape == (300 - 20, 43)
    assert np.isfinite(obs_stack).all()
    gate_fn = make_gate_fn(model)
    gates = gate_fn(obs_stack)
    assert gates.shape == (280,)
    assert (gates >= 0.0).all() and (gates <= 1.0).all()  # clipped to the env's [0,1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_interventions.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.interventions'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/interventions.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_interventions.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/interventions.py tests/test_interventions.py
git commit -m "feat: RQ3 feature groups, deterministic replay, gate adapter"
```

---

### Task 2: Feature-group ablation and causal effect

**Files:**
- Modify: `src/interventions.py`
- Test: `tests/test_interventions.py`

**Interfaces:**
- Consumes: `make_gate_fn` output type `Callable[[np.ndarray], np.ndarray]`.
- Produces:
  - `freeze_group(observations: np.ndarray, indices: list[int]) -> np.ndarray` — copy with those columns set to their column mean.
  - `permute_group(observations: np.ndarray, indices: list[int], rng: np.random.Generator) -> np.ndarray` — copy with those columns row-permuted (shared permutation across the group's columns).
  - `causal_effect(gate_fn, observations, indices, mode: str, seed: int = 0) -> float` — mean absolute change in gate when the group is ablated; `mode` in `{"freeze", "permute"}`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_interventions.py
from src.interventions import freeze_group, permute_group, causal_effect


def test_freeze_removes_variance_permute_preserves_marginal():
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(200, 43)).astype(np.float32)
    frozen = freeze_group(obs, [42])
    assert np.allclose(frozen[:, 42].var(), 0.0)          # variation removed
    assert np.allclose(frozen[:, :42], obs[:, :42])       # other columns untouched
    permuted = permute_group(obs, [42], np.random.default_rng(1))
    assert np.allclose(np.sort(permuted[:, 42]), np.sort(obs[:, 42]))  # marginal preserved
    assert np.allclose(permuted[:, :42], obs[:, :42])


def test_causal_effect_zero_for_inert_feature_positive_for_used_feature():
    # A gate that depends ONLY on the signal feature (index 42): ablating the
    # signal must move the gate; ablating an inert feature (index 0) must not.
    def signal_only_gate(observations):
        obs = np.atleast_2d(np.asarray(observations, dtype=np.float32))
        return 1.0 / (1.0 + np.exp(-4.0 * obs[:, 42]))
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(300, 43)).astype(np.float32)
    signal_effect = causal_effect(signal_only_gate, obs, [42], "freeze")
    inert_effect = causal_effect(signal_only_gate, obs, [0], "freeze")
    assert signal_effect > 1e-3
    assert inert_effect < 1e-9
    # permute mode is also well-defined and non-negative
    assert causal_effect(signal_only_gate, obs, [42], "permute", seed=3) > 1e-3


def test_causal_effect_rejects_unknown_mode():
    import pytest
    with pytest.raises(ValueError):
        causal_effect(lambda o: np.zeros(len(o)), np.zeros((5, 43)), [0], "scramble")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_interventions.py -v`
Expected: FAIL — `ImportError: cannot import name 'freeze_group'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/interventions.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_interventions.py -v`
Expected: PASS (5 tests total).

- [ ] **Step 5: Commit**

```bash
git add src/interventions.py tests/test_interventions.py
git commit -m "feat: RQ3 feature-group ablation and causal effect"
```

---

### Task 3: Environment interventions (vol shock, signal flip)

**Files:**
- Modify: `src/interventions.py`
- Test: `tests/test_interventions.py`

**Interfaces:**
- Produces:
  - `inject_vol_shock(market: dict, t0: int, width: int, multiplier: float, risky_index: int = 0) -> dict` — new market with the risky asset's returns amplified over `[t0, t0+width)`; input not mutated.
  - `flip_signal(market: dict, t0: int, value: float) -> dict` — new market with `signal[t0]` set to `value`; input not mutated.

These produce gate-response evidence that the agent is causally responsive to the crisis mechanism (a legibility/sanity layer, not the verdict — see spec §7).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_interventions.py
from src.interventions import inject_vol_shock, flip_signal


def test_inject_vol_shock_is_local_and_nonmutating():
    from src.synthetic_market import generate_risky_safe_market
    market = generate_risky_safe_market(300, seed=2, signal_strength=0.95)
    original_returns = np.array(market["returns"])   # snapshot
    shocked = inject_vol_shock(market, t0=100, width=5, multiplier=5.0, risky_index=0)
    assert np.allclose(market["returns"], original_returns)          # input untouched
    assert np.allclose(shocked["returns"][100:105, 0],
                       original_returns[100:105, 0] * 5.0)            # intended slice scaled
    assert np.allclose(shocked["returns"][:100], original_returns[:100])  # elsewhere unchanged
    assert np.allclose(shocked["returns"][:, 1], original_returns[:, 1])  # safe asset unchanged
    assert np.allclose(shocked["signal"], market["signal"])          # signal preserved


def test_flip_signal_sets_one_step_without_mutating():
    from src.synthetic_market import generate_risky_safe_market
    market = generate_risky_safe_market(300, seed=3, signal_strength=0.95)
    original_signal = np.array(market["signal"])
    flipped = flip_signal(market, t0=150, value=1.0)
    assert np.allclose(market["signal"], original_signal)   # input untouched
    assert flipped["signal"][150] == 1.0
    assert np.allclose(np.delete(flipped["signal"], 150), np.delete(original_signal, 150))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_interventions.py -v`
Expected: FAIL — `ImportError: cannot import name 'inject_vol_shock'`.

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/interventions.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_interventions.py -v`
Expected: PASS (7 tests total).

- [ ] **Step 5: Commit**

```bash
git add src/interventions.py tests/test_interventions.py
git commit -m "feat: RQ3 environment interventions (vol shock, signal flip)"
```

---

### Task 4: Gradient saliency attribution

**Files:**
- Create: `src/attribution.py`
- Test: `tests/test_attribution.py`

**Interfaces:**
- Consumes: a differentiable `gate_mean_fn: Callable[[torch.Tensor], torch.Tensor]` mapping a `[T, obs_dim]` tensor to a `[T]` gate tensor.
- Produces:
  - `make_gate_mean_fn(model) -> Callable[[torch.Tensor], torch.Tensor]` — extracts the SB3 policy's deterministic gate (Gaussian mean, pre-clip) as a differentiable function of the observation tensor.
  - `saliency_importance(gate_mean_fn, observations: np.ndarray) -> np.ndarray` — per-feature importance `mean_t |∂g/∂oᵢ|`, shape `[obs_dim]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_attribution.py
import numpy as np
import torch
from src.attribution import saliency_importance


def test_saliency_ranks_features_by_weight_on_linear_policy():
    # Known-answer calibration: a linear gate g = sigmoid(w . o). The saliency of
    # feature i is |w_i| * sigmoid'(.) > proportional ordering by |w_i|. Feature 5
    # has the largest weight; feature 10 has zero weight (inert).
    weights = np.zeros(43, dtype=np.float32)
    weights[5] = 3.0
    weights[10] = 0.0
    weights[20] = 1.0
    w = torch.tensor(weights)

    def linear_gate_mean(obs_tensor):
        return torch.sigmoid(obs_tensor @ w)

    rng = np.random.default_rng(0)
    obs = rng.normal(scale=0.1, size=(200, 43)).astype(np.float32)
    importance = saliency_importance(linear_gate_mean, obs)

    assert importance.shape == (43,)
    assert np.isfinite(importance).all()
    assert importance.argmax() == 5                 # largest-weight feature ranked top
    assert importance[10] < importance[20] < importance[5]   # inert < small < large
    assert importance[10] < 1e-6                    # inert feature ~ zero saliency
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_attribution.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.attribution'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/attribution.py
"""Post-hoc attribution of the trained gate agent's de-risking decisions.

The ATTRIBUTION track for the RQ3 faithfulness experiment: gradient saliency and
KernelSHAP, both returning a per-feature importance vector over the 43-dim gate
observation. Compared against the CAUSAL track (src/interventions.py) to test
whether attribution identifies the true (signal) driver. Functions take injected
callables so they can be calibrated on a known linear policy before use on PPO.
"""
import numpy as np
import torch


def make_gate_mean_fn(model):
    """Adapter: the SB3 policy's deterministic gate (Gaussian mean, PRE-clip) as a
    differentiable function of the observation tensor. Pre-clip is intentional —
    the env's [0,1] clip has zero gradient in saturation and would zero out
    saliency exactly where the agent is decisive."""
    def gate_mean_fn(obs_tensor: torch.Tensor) -> torch.Tensor:
        distribution = model.policy.get_distribution(obs_tensor)
        return distribution.distribution.mean[:, 0]
    return gate_mean_fn


def saliency_importance(gate_mean_fn, observations: np.ndarray) -> np.ndarray:
    """Per-feature gradient saliency: mean_t |d gate / d obs_i| over the stack."""
    obs_tensor = torch.as_tensor(np.asarray(observations, dtype=np.float32))
    obs_tensor.requires_grad_(True)
    gate = gate_mean_fn(obs_tensor)
    # Rows are independent, so grad of the sum w.r.t. row i equals grad of gate_i.
    gate.sum().backward()
    grads = obs_tensor.grad.detach().numpy()
    return np.mean(np.abs(grads), axis=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_attribution.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add src/attribution.py tests/test_attribution.py
git commit -m "feat: RQ3 gradient-saliency attribution (calibrated on linear policy)"
```

---

### Task 5: KernelSHAP attribution and group aggregation

**Files:**
- Modify: `src/attribution.py`, `pyproject.toml` (via `uv add shap`)
- Test: `tests/test_attribution.py`

**Interfaces:**
- Consumes: a numpy `gate_fn: Callable[[np.ndarray], np.ndarray]` (from `src.interventions.make_gate_fn`).
- Produces:
  - `shap_importance(gate_fn, observations, n_background: int = 40, n_explain: int = 60, seed: int = 0) -> np.ndarray` — per-feature `mean |SHAP value|`, shape `[obs_dim]`.
  - `aggregate_to_groups(importance: np.ndarray, groups: dict[str, list[int]]) -> dict[str, float]` — summed `|importance|` per group.

**Setup note:** this task adds the `shap` dependency. State it before the code: run `uv add shap`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_attribution.py
from src.attribution import shap_importance, aggregate_to_groups


def test_shap_ranks_features_by_weight_on_linear_policy():
    # Same known-answer calibration as saliency, for the model-agnostic path.
    weights = np.zeros(43, dtype=np.float32)
    weights[5] = 3.0
    weights[20] = 1.0  # feature 10 stays inert (weight 0)

    def linear_gate(observations):
        obs = np.atleast_2d(np.asarray(observations, dtype=np.float32))
        return 1.0 / (1.0 + np.exp(-(obs @ weights)))

    rng = np.random.default_rng(0)
    obs = rng.normal(scale=0.1, size=(120, 43)).astype(np.float32)
    importance = shap_importance(linear_gate, obs, n_background=30, n_explain=40, seed=0)

    assert importance.shape == (43,)
    assert np.isfinite(importance).all()
    assert importance.argmax() == 5              # largest-weight feature ranked top
    assert importance[10] < importance[20]       # inert below the small-weight feature


def test_aggregate_to_groups_sums_absolute_importance():
    importance = np.zeros(43)
    importance[0:40] = 0.01     # returns block sums to 0.4
    importance[40:42] = 0.5     # short_vol sums to 1.0
    importance[42] = 2.0        # signal
    groups = {"returns": list(range(40)), "short_vol": [40, 41], "signal": [42]}
    agg = aggregate_to_groups(importance, groups)
    assert np.isclose(agg["returns"], 0.4)
    assert np.isclose(agg["short_vol"], 1.0)
    assert np.isclose(agg["signal"], 2.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_attribution.py -v`
Expected: FAIL — `ImportError: cannot import name 'shap_importance'` (and `shap` not yet installed).

- [ ] **Step 3: Write minimal implementation**

First add the dependency (state explicitly — new package): `uv add shap`

```python
# append to src/attribution.py
def shap_importance(gate_fn, observations, n_background: int = 40,
                    n_explain: int = 60, seed: int = 0) -> np.ndarray:
    """Per-feature KernelSHAP importance: mean |SHAP value| over an explained
    sample, using a seeded background sub-sample. `gate_fn` is the numpy gate
    adapter (src.interventions.make_gate_fn)."""
    import shap
    observations = np.asarray(observations, dtype=np.float32)
    rng = np.random.default_rng(seed)
    n = len(observations)
    background = observations[rng.choice(n, size=min(n_background, n), replace=False)]
    explain = observations[rng.choice(n, size=min(n_explain, n), replace=False)]
    # KernelExplainer uses the legacy global RNG internally; seed it for determinism.
    np.random.seed(seed)
    explainer = shap.KernelExplainer(gate_fn, background)
    values = np.asarray(explainer.shap_values(explain, silent=True))
    return np.mean(np.abs(values), axis=0)


def aggregate_to_groups(importance: np.ndarray, groups: dict) -> dict:
    """Sum |importance| within each semantic feature group."""
    importance = np.asarray(importance, dtype=float)
    return {name: float(np.abs(importance[indices]).sum()) for name, indices in groups.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_attribution.py -v`
Expected: PASS (3 tests total). (KernelSHAP on 43 features is a few seconds.)

- [ ] **Step 5: Commit**

```bash
git add src/attribution.py tests/test_attribution.py pyproject.toml uv.lock
git commit -m "feat: RQ3 KernelSHAP attribution + group aggregation; add shap dep"
```

---

### Task 6: Faithfulness verdict and the RQ3 experiment

**Files:**
- Create: `src/rq3_faithfulness.py`
- Test: `tests/test_rq3_faithfulness.py`

**Interfaces:**
- Consumes: `feature_groups`, `rollout_observations`, `make_gate_fn`, `causal_effect` (from `src.interventions`); `make_gate_mean_fn`, `saliency_importance`, `shap_importance`, `aggregate_to_groups` (from `src.attribution`); `train_agent` (from `src.train`); `generate_risky_safe_market` (from `src.synthetic_market`); `scipy.stats.spearmanr`.
- Produces:
  - `run_probe(gate_fn, gate_mean_fn, observations, groups, seed: int = 0) -> dict` — pure verdict computation (no training / IO): per-group causal effect (freeze + permute), per-group attribution (saliency + shap), per-feature Spearman agreement, and the top group per method.
  - `run_experiment(config: dict, n_seeds: int) -> dict` — trains `n_seeds` gate agents, runs `run_probe` per seed on a held-out market, aggregates across seeds, and saves a timestamped run.

**The verdict `run_probe` returns (exact keys — later readers depend on these):**
```python
{
  "causal":       {"freeze": {group: float}, "permute": {group: float}},
  "attribution":  {"saliency": {group: float}, "shap": {group: float}},
  "spearman":     {"saliency_freeze": float, "saliency_permute": float,
                   "shap_freeze": float, "shap_permute": float},
  "top_group":    {"causal_freeze": str, "causal_permute": str,
                   "saliency": str, "shap": str},
}
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rq3_faithfulness.py
import numpy as np
import torch
from src.rq3_faithfulness import run_probe
from src.interventions import feature_groups


def test_run_probe_identifies_signal_as_causal_driver_and_emits_verdict():
    # Ground-truth gate that uses ONLY the signal feature (index 42). Both the
    # numpy gate_fn and the differentiable gate_mean_fn express the same policy,
    # so causal ablation AND faithful attribution must both rank `signal` top.
    def gate_fn(observations):
        obs = np.atleast_2d(np.asarray(observations, dtype=np.float32))
        return 1.0 / (1.0 + np.exp(-4.0 * obs[:, 42]))

    def gate_mean_fn(obs_tensor):
        return torch.sigmoid(4.0 * obs_tensor[:, 42])

    rng = np.random.default_rng(0)
    obs = rng.normal(scale=0.3, size=(200, 43)).astype(np.float32)
    groups = feature_groups(window=20, n_assets=2)
    verdict = run_probe(gate_fn, gate_mean_fn, obs, groups, seed=0)

    # premise: the signal is the dominant CAUSAL driver (this is ground truth)
    assert verdict["top_group"]["causal_freeze"] == "signal"
    assert verdict["causal"]["freeze"]["signal"] > verdict["causal"]["freeze"]["returns"]
    # a faithful attribution recovers it too (this stub policy IS faithful)
    assert verdict["top_group"]["saliency"] == "signal"
    # verdict object has the full required shape
    assert set(verdict) == {"causal", "attribution", "spearman", "top_group"}
    assert set(verdict["spearman"]) == {"saliency_freeze", "saliency_permute",
                                        "shap_freeze", "shap_permute"}
    for rho in verdict["spearman"].values():
        assert -1.0 <= rho <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rq3_faithfulness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.rq3_faithfulness'`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/rq3_faithfulness.py
"""RQ3 / MC2 — causal-faithfulness experiment.

Probes a trained gate agent on the synthetic risky+safe market, where the
`signal` feature is the known ground-truth driver. Compares the CAUSAL track
(feature-group ablation) against the ATTRIBUTION track (saliency + KernelSHAP)
to test whether post-hoc attribution identifies the true mechanism (H3).

Run: uv run python -m src.rq3_faithfulness
"""
import json
import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from src.interventions import (feature_groups, rollout_observations, make_gate_fn,
                               causal_effect)
from src.attribution import (make_gate_mean_fn, saliency_importance, shap_importance,
                             aggregate_to_groups)

# Config — edit these directly
CONFIG = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
          "safe_asset_index": 1, "action_mode": "gate",
          "total_timesteps": 150_000, "n_steps": 6000, "signal_strength": 0.95}
N_SEEDS = 5
SHAP_BACKGROUND = 40
SHAP_EXPLAIN = 60


def _per_feature_causal(gate_fn, observations, obs_dim, mode, seed):
    """Causal effect of ablating each single feature -> a [obs_dim] vector, for
    the per-feature Spearman comparison against attribution."""
    return np.array([causal_effect(gate_fn, observations, [i], mode, seed=seed)
                     for i in range(obs_dim)], dtype=float)


def _top(group_importance: dict) -> str:
    return max(group_importance, key=group_importance.get)


def run_probe(gate_fn, gate_mean_fn, observations, groups, seed: int = 0) -> dict:
    """Pure verdict computation over one agent's replayed decisions. No training,
    no IO. See the module plan for the returned schema."""
    observations = np.asarray(observations, dtype=np.float32)
    obs_dim = observations.shape[1]

    causal_group = {
        mode: {name: causal_effect(gate_fn, observations, idx, mode, seed=seed)
               for name, idx in groups.items()}
        for mode in ("freeze", "permute")
    }
    saliency_vec = saliency_importance(gate_mean_fn, observations)
    shap_vec = shap_importance(gate_fn, observations, n_background=SHAP_BACKGROUND,
                               n_explain=SHAP_EXPLAIN, seed=seed)
    attribution_group = {
        "saliency": aggregate_to_groups(saliency_vec, groups),
        "shap": aggregate_to_groups(shap_vec, groups),
    }

    # per-feature Spearman agreement between causal effect and attribution
    spearman = {}
    for mode in ("freeze", "permute"):
        causal_vec = _per_feature_causal(gate_fn, observations, obs_dim, mode, seed)
        for attr_name, attr_vec in (("saliency", saliency_vec), ("shap", shap_vec)):
            rho, _ = spearmanr(causal_vec, attr_vec)
            # spearmanr returns nan if a vector is constant; report 0.0 (no monotone
            # agreement detectable) rather than letting nan flow downstream.
            spearman[f"{attr_name}_{mode}"] = float(rho) if np.isfinite(rho) else 0.0

    top_group = {
        "causal_freeze": _top(causal_group["freeze"]),
        "causal_permute": _top(causal_group["permute"]),
        "saliency": _top(attribution_group["saliency"]),
        "shap": _top(attribution_group["shap"]),
    }
    return {"causal": causal_group, "attribution": attribution_group,
            "spearman": spearman, "top_group": top_group}


def run_experiment(config: dict, n_seeds: int) -> dict:
    """Train n_seeds gate agents at the configured signal strength, probe each on
    a held-out market, aggregate the verdict across seeds, and save the run."""
    from src.train import train_agent
    from src.synthetic_market import generate_risky_safe_market

    now = datetime.datetime.now()
    print("=" * 60)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Config      : {config} | n_seeds={n_seeds}")
    print("=" * 60)

    groups = feature_groups(config["window"], n_assets=2)
    per_seed = []
    for seed in range(n_seeds):
        train_market = generate_risky_safe_market(config["n_steps"], seed=1000 + seed,
                                                  signal_strength=config["signal_strength"])
        model = train_agent(train_market, {**config, "seed": seed})
        eval_market = generate_risky_safe_market(config["n_steps"], seed=2000 + seed,
                                                 signal_strength=config["signal_strength"])
        observations = rollout_observations(model, eval_market, config)
        gate_fn = make_gate_fn(model)
        gate_mean_fn = make_gate_mean_fn(model)
        verdict = run_probe(gate_fn, gate_mean_fn, observations, groups, seed=seed)
        print(f"seed={seed}: causal_top={verdict['top_group']['causal_freeze']} "
              f"saliency_top={verdict['top_group']['saliency']} "
              f"shap_top={verdict['top_group']['shap']}", flush=True)
        per_seed.append(verdict)

    # Aggregate: fraction of seeds where each method ranks `signal` top, and mean
    # Spearman per method (computed over ALL seeds — the full experiment).
    def signal_top_fraction(method_key, block):
        return float(np.mean([v[block][method_key] == "signal" for v in per_seed]))

    summary = {
        "signal_is_causal_driver_fraction": signal_top_fraction("causal_freeze", "top_group"),
        "saliency_signal_top_fraction": signal_top_fraction("saliency", "top_group"),
        "shap_signal_top_fraction": signal_top_fraction("shap", "top_group"),
        "spearman_mean": {k: float(np.mean([v["spearman"][k] for v in per_seed]))
                          for k in per_seed[0]["spearman"]},
        "spearman_std": {k: float(np.std([v["spearman"][k] for v in per_seed]))
                         for k in per_seed[0]["spearman"]},
    }

    out_dir = (Path(__file__).resolve().parent.parent / "outputs"
               / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_rq3-faithfulness")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump({"config": config, "n_seeds": n_seeds}, f, indent=2)
    with open(out_dir / "results.json", "w") as f:
        json.dump({"summary": summary, "per_seed": per_seed}, f, indent=2)
    print(f"Summary: {summary}")
    print(f"Saved run to: {out_dir}")
    return {"summary": summary, "per_seed": per_seed, "out_dir": str(out_dir)}


if __name__ == "__main__":
    run_experiment(CONFIG, N_SEEDS)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rq3_faithfulness.py -v`
Expected: PASS (1 test). This exercises the full verdict pipeline on a signal-only stub in seconds — no training.

- [ ] **Step 5: Run the full experiment and inspect the artifact**

Run: `uv run python -u -m src.rq3_faithfulness`
Expected: trains 5 gate agents (minutes each on CPU), prints per-seed top groups, writes `outputs/<ts>_rq3-faithfulness/` with `config.json` and `results.json`. The premise check `signal_is_causal_driver_fraction` should be high (the agents key on the signal); the `saliency_signal_top_fraction` / `shap_signal_top_fraction` and `spearman_mean` are the measured faithfulness verdict — report whatever they are. If `signal_is_causal_driver_fraction` is low, the agents did not learn to use the signal: raise `total_timesteps` or check `signal_strength`; do NOT proceed to a faithfulness claim on agents with no signal-driven mechanism.

- [ ] **Step 6: Commit**

```bash
git add src/rq3_faithfulness.py tests/test_rq3_faithfulness.py
git commit -m "feat: RQ3 faithfulness verdict + experiment (causal vs attribution)"
```

---

### Task 7: Response-curve figures (legibility layer)

**Files:**
- Modify: `src/rq3_faithfulness.py`
- Test: `tests/test_rq3_faithfulness.py`

**Interfaces:**
- Produces:
  - `gate_response_to_vol_shock(model, market, config, t0, width, multiplier) -> dict` — `{"baseline": [gate...], "shocked": [gate...]}` gate trajectories around the shocked window, for the response-curve figure.

This is the spec §7 legibility/sanity layer (evidence the agent is causally responsive), NOT the verdict. Keep it to one function + one figure; if it proves fiddly, this task can be dropped without affecting the verdict (Tasks 1–6).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_rq3_faithfulness.py
from src.rq3_faithfulness import gate_response_to_vol_shock


def test_vol_shock_response_returns_aligned_gate_trajectories():
    from src.synthetic_market import generate_risky_safe_market
    from src.interventions import make_gate_fn

    class _RiskGate:
        # a gate that de-risks when recent risky-asset moves are large (uses vol feature 40)
        def predict(self, obs, deterministic=True):
            obs = np.atleast_2d(np.asarray(obs, dtype=np.float32))
            return np.clip(obs[:, 40:41] * 20.0, 0.0, 1.0), None

    market = generate_risky_safe_market(400, seed=4, signal_strength=0.95)
    config = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
              "safe_asset_index": 1, "action_mode": "gate"}
    curves = gate_response_to_vol_shock(_RiskGate(), market, config,
                                        t0=200, width=5, multiplier=6.0)
    assert set(curves) == {"baseline", "shocked"}
    assert len(curves["baseline"]) == len(curves["shocked"]) > 0
    assert np.isfinite(curves["baseline"]).all() and np.isfinite(curves["shocked"]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rq3_faithfulness.py -v`
Expected: FAIL — `ImportError: cannot import name 'gate_response_to_vol_shock'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/rq3_faithfulness.py (import at top: from src.interventions import inject_vol_shock)
def gate_response_to_vol_shock(model, market, config, t0: int, width: int,
                               multiplier: float) -> dict:
    """Gate trajectory on the original vs. vol-shocked market, aligned on the same
    decision timeline. Evidence the agent is causally responsive to volatility."""
    from src.interventions import inject_vol_shock  # local import keeps the top clean
    gate_fn = make_gate_fn(model)
    baseline_obs = rollout_observations(model, market, config)
    shocked_market = inject_vol_shock(market, t0=t0, width=width, multiplier=multiplier)
    shocked_obs = rollout_observations(model, shocked_market, config)
    return {"baseline": [float(x) for x in gate_fn(baseline_obs)],
            "shocked": [float(x) for x in gate_fn(shocked_obs)]}
```

Also add, inside `run_experiment` before the save block, a figure on the first
trained agent (guarded so a plotting failure never sinks the verdict):

```python
    # Legibility figure: gate response to a vol shock on the first agent (spec §7).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curves = gate_response_to_vol_shock(first_model, eval_market, config,
                                        t0=config["n_steps"] // 2, width=5, multiplier=6.0)
    plt.figure()
    plt.plot(curves["baseline"], label="baseline", linewidth=0.8)
    plt.plot(curves["shocked"], label="vol shock", linewidth=0.8)
    plt.xlabel("decision step"); plt.ylabel("de-risking gate g"); plt.legend()
    plt.title("Gate response to an injected volatility shock")
    plt.savefig(out_dir / "gate_response_vol_shock.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"wrote {out_dir / 'gate_response_vol_shock.png'}")
```

To supply `first_model`/`eval_market` to the figure, capture the seed-0 model and
its eval market in the loop (e.g. `if seed == 0: first_model, first_eval = model, eval_market`)
and use `first_eval` in the figure call. Keep the change minimal.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rq3_faithfulness.py -v`
Expected: PASS (2 tests total).

- [ ] **Step 5: Commit**

```bash
git add src/rq3_faithfulness.py tests/test_rq3_faithfulness.py
git commit -m "feat: RQ3 vol-shock gate-response figure (legibility layer)"
```

---

## Plan Self-Review

**Spec coverage:**
- Synthetic-only testbed, gate agent, signal ground truth (spec §2) → Task 6 `run_experiment` (risky+safe, `signal_strength=0.95`).
- Feature groups over the 43-dim obs (spec §3) → Task 1 `feature_groups`.
- Causal track: per-feature-group ablation freeze+permute (spec §4.1) → Tasks 1–2; environment interventions (vol shock, signal flip) → Task 3; response curves → Task 7.
- Attribution track: gradient saliency + KernelSHAP, group aggregation (spec §4.2) → Tasks 4–5.
- Verdict: ground-truth anchor (signal-dominance) + Spearman rank agreement across seeds (spec §5) → Task 6 `run_probe` + `run_experiment` summary.
- Testing: known-answer attribution calibration, ablation semantics + causal ground truth, end-to-end verdict gate (spec §6) → Tasks 4–5, 2, 6.
- `shap` dependency, compute-light, reuse MC1 code, run as module (spec §7) → Task 5 `uv add`, Task 6 reuse + entrypoint.
- Out of scope (real data, tilt agent, extra attribution methods) → correctly absent.

**Placeholder scan:** none — every code step has complete, runnable content and exact commands with expected output.

**Type consistency:** `feature_groups` returns `dict[str, list[int]]`, consumed by `causal_effect` (indices) and `aggregate_to_groups` (indices) identically. `make_gate_fn` → numpy `gate_fn` consumed by `causal_effect`, `shap_importance`, `run_probe`, `gate_response_to_vol_shock`. `make_gate_mean_fn` → torch `gate_mean_fn` consumed by `saliency_importance`, `run_probe`. `run_probe` return schema (Task 6) matches the keys asserted in its test and read by `run_experiment`'s summary. `rollout_observations(model, market, config)` signature identical across Tasks 1, 6, 7. Consistent.
