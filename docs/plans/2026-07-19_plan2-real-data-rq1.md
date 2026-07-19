# Real-Data Skill-vs-Structure Verdict (Plan 2 of 3, RQ1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On real multi-asset ETFs, deliver a clean, cost-aware, walk-forward verdict on RQ1 — how much of the de-risking gate's out-of-sample tail/risk benefit is *learned skill* versus *inherited structure* — reported as skill **net of a phase-randomized placebo null** with confidence intervals, against a strong base ladder and literature baselines.

**Architecture:** A `yfinance` ETF panel (cached to a git-ignored `data/`) becomes the return matrix. A causal signal feature turns it into the `AllocationEnv` `market` dict (reusing the Plan-1 env/train unchanged). An anchored (expanding-window) walk-forward retrains the de-risking-gate PPO agent per fold and stitches out-of-sample results; the operational skill measure is the stitched mean structure-baselined reward. A phase-randomization placebo destroys timeable structure to produce the overfitting/luck null (the real-data analog of Plan-1's signal-off), and the headline statistic is `skill_net = OOS_skill − placebo_skill` with CIs. Literature baselines (1/N, minimum-variance, CVaR-minimizing) and the base ladder (equal-weight, vol-scaled, risk-parity) are evaluated on the same OOS window with matched costs.

**Tech Stack:** Python 3.11, `uv` (never pip), NumPy, pandas, SciPy, `yfinance`, Gymnasium, Stable-Baselines3 (PyTorch), pytest, tqdm, matplotlib.

## Global Constraints

- Package manager is `uv` only: `uv add <pkg>`, `uv run python -u <script>`. Never `pip`. (CLAUDE.md)
- **New dependency:** `yfinance` — added in Task 1 (`uv add yfinance`). Stated explicitly before use per CLAUDE.md.
- Code style: config constants at the TOP of each script, clearly labelled; `snake_case`; full words (`n_assets` not `na`); flat over nested; 4-space indent. (code-style.md)
- No silent failures: no bare `except`; every `continue`/`break` logged; `dict[key]` not `.get()` unless absence is explicitly handled; no `None` flowing downstream unguarded. (code-style.md)
- No absolute paths in any script; scripts runnable from any working directory (resolve paths relative to `__file__`). (code-style.md)
- Traceability: every training/experiment script prints a run header (timestamp, script, config), writes to `outputs/YYYY-MM-DD_HH-MM-SS_<desc>/`, saves `config.json` and `results.json`, never overwrites an old output dir. Wrap major loops in `tqdm`; run with `python -u`. (traceability.md)
- Crash recovery: long runs (walk-forward, placebo) must be restartable from partial outputs, and **final aggregate metrics are always recomputed over all folds**, never a partial subset. (code-style.md)
- Determinism: every stochastic function takes an explicit integer `seed`; no reliance on global RNG state.
- **No lookahead (the cardinal real-data rule):** every feature, signal, covariance, and base/baseline weight at decision time `t` must be computed from data strictly before the return it earns (`returns[:t]`). Any leakage invalidates the RQ1 verdict.
- All portfolio weights are long-only on the simplex: `w >= 0`, `sum(w) == 1`.
- `data/` (raw price/return cache) is git-ignored — it is large and reproducible from config; never commit it. Keep private/large artifacts out of the public repo.

## Interfaces reused from Plan 1 (do not re-implement)

- `src.simplex.project_to_simplex(v: np.ndarray) -> np.ndarray`
- `src.metrics`: `expected_shortfall(returns, alpha=0.99)`, `max_drawdown(returns)`, `tail_ratio(returns)`, `skewness(returns)`, `sharpe(returns, periods_per_year=252)`, `sortino(returns, periods_per_year=252)`, `turnover(weights)`
- `src.base_policies`: `equal_weight_base(return_window)`, `vol_scaled_base(return_window, target_vol=0.01)`, `BASE_POLICIES` dict (keys `"equal_weight"`, `"vol_scaled"`)
- `src.allocation_env.AllocationEnv(market, base_name, window=20, cost_bps=10.0, safe_asset_index=None)` — consumes `market["returns"]` `(T, n)` and `market["signal"]` `(T,)`; action is scalar gate `g∈[0,1]`; `w=(1-g)·base+g·safe`; reward = agent net log-return − base net log-return; `last_info` keys `weights`, `base_weights`, `gate`, `port_return`, `base_return`.
- `src.train.build_env(market, config)`, `src.train.train_agent(market, config) -> PPO` — `config` keys read: `base_name`, `window`, `cost_bps`, `safe_asset_index`, `total_timesteps`, `seed`.

## File structure (new files)

```
src/data.py            # yfinance ETF panel -> cached returns matrix (Task 1)
src/real_market.py     # returns matrix -> env market dict with a CAUSAL signal (Task 2)
src/base_policies.py   # +risk_parity_base, +register "risk_parity" (Task 3, modify)
src/baselines.py       # min-variance, CVaR-min baselines + cost-aware weight roller (Task 4)
src/walk_forward.py    # expanding-window folds, per-fold retrain, OOS stitching (Task 5)
src/placebo.py         # phase-randomization null (overfitting/luck baseline) (Task 6)
src/rq1_real_data.py   # the RQ1 experiment: skill net of null + metrics table + figures (Task 7)
tests/test_*.py        # one test module per task
```

---

### Task 1: ETF data pipeline (yfinance -> cached returns matrix)

**Files:**
- Create: `src/data.py`
- Test: `tests/test_data.py`
- Modify: `.gitignore` (add `data/`)

**Interfaces:**
- Consumes: nothing (project entrypoint for real data).
- Produces:
  - `compute_returns(prices: pd.DataFrame) -> pd.DataFrame` — daily simple returns from an adjusted-close price frame (dates × tickers); drops the first (NaN) row; drops any ticker with a NaN after the first row and logs which.
  - `download_prices(tickers: list[str], start: str, end: str, cache_path: Path) -> pd.DataFrame` — adjusted close via `yfinance`, cached to parquet; on re-call loads the cache instead of re-downloading.
  - `load_etf_panel(cache_path: Path = DEFAULT_CACHE, start: str = START, end: str = END) -> dict` — returns `{"returns": np.ndarray (T, n), "dates": np.ndarray (T,), "tickers": list[str]}`.
  - Module constants `TICKERS`, `START`, `END`, `DEFAULT_CACHE`.

- [ ] **Step 1: Add the dependency**

```bash
cd /dccstor/meghanadhp/projects/Helix/rl-allocation-audit
uv add yfinance
```

- [ ] **Step 2: Add `data/` to `.gitignore`**

Append a line so the raw cache is never committed (large, reproducible from config). The file currently ends after `references/`; add:

```gitignore
data/
```

- [ ] **Step 3: Write the failing test**

```python
# tests/test_data.py
import numpy as np
import pandas as pd
from src.data import compute_returns, load_etf_panel


def test_compute_returns_drops_first_row_and_shapes():
    prices = pd.DataFrame(
        {"A": [100.0, 101.0, 99.0], "B": [50.0, 50.0, 55.0]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
    )
    rets = compute_returns(prices)
    assert list(rets.columns) == ["A", "B"]
    assert rets.shape == (2, 2)                       # first row (NaN) dropped
    np.testing.assert_allclose(rets["A"].iloc[0], 0.01, atol=1e-9)
    np.testing.assert_allclose(rets["B"].iloc[1], 0.10, atol=1e-9)


def test_compute_returns_drops_ticker_with_gap():
    prices = pd.DataFrame(
        {"A": [100.0, 101.0, 102.0], "B": [50.0, np.nan, 55.0]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
    )
    rets = compute_returns(prices)
    assert list(rets.columns) == ["A"]                # B dropped for the NaN gap


def test_load_etf_panel_uses_cache(tmp_path):
    # A pre-existing cache must be read without any network download.
    cache = tmp_path / "panel.parquet"
    prices = pd.DataFrame(
        {"A": [100.0, 101.0, 102.0, 103.0], "B": [50.0, 51.0, 50.0, 52.0]},
        index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
    )
    prices.to_parquet(cache)
    panel = load_etf_panel(cache_path=cache)
    assert panel["returns"].shape == (3, 2)           # 4 prices -> 3 returns
    assert panel["tickers"] == ["A", "B"]
    assert panel["dates"].shape == (3,)
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL with import error on `src.data`.

- [ ] **Step 5: Write minimal implementation**

```python
# src/data.py
"""ETF price/return panel via yfinance, cached to a git-ignored data/ dir.

Universe (spec §8): liquid multi-asset ETFs spanning equity sectors, duration,
gold, and international equity — chosen for a long survivorship-clean daily
history through 2008 / 2020 / 2022 stress. Adjusted close only (splits/dividends
handled by the vendor). The cache makes runs restartable and keeps the large raw
data out of the repo.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Config — edit these directly
TICKERS = [
    "SPY",                                  # US large-cap equity
    "XLK", "XLF", "XLE", "XLV", "XLU",      # sector SPDRs (tech/fin/energy/health/utilities)
    "IEF", "TLT",                           # 7-10y and 20y+ US Treasuries (the safe-haven sleeve)
    "GLD",                                  # gold
    "EFA", "EEM",                           # developed-ex-US and emerging equity
]
START = "2005-01-01"
END = "2025-01-01"
DEFAULT_CACHE = Path(__file__).resolve().parent.parent / "data" / "etf_panel.parquet"


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().iloc[1:]
    # A NaN after the first row means a ticker lacks full history over the window.
    # Drop it loudly rather than forward-filling (which would fabricate returns).
    bad = returns.columns[returns.isna().any()].tolist()
    if bad:
        print(f"compute_returns: dropping {len(bad)} tickers with gaps: {bad}")
        returns = returns.drop(columns=bad)
    return returns


def download_prices(tickers: list, start: str, end: str, cache_path: Path) -> pd.DataFrame:
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"download_prices: loading cache {cache_path}")
        return pd.read_parquet(cache_path)
    import yfinance as yf
    print(f"download_prices: downloading {len(tickers)} tickers {start}..{end} from yfinance")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"].dropna(how="all")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(cache_path)
    print(f"download_prices: cached {prices.shape[0]} rows x {prices.shape[1]} tickers to {cache_path}")
    return prices


def load_etf_panel(cache_path: Path = DEFAULT_CACHE, start: str = START, end: str = END) -> dict:
    prices = download_prices(TICKERS, start, end, cache_path)
    returns = compute_returns(prices)
    print(f"load_etf_panel: {returns.shape[0]} return rows x {returns.shape[1]} tickers "
          f"({returns.index[0].date()}..{returns.index[-1].date()})")
    return {
        "returns": returns.to_numpy(dtype=float),
        "dates": returns.index.to_numpy(),
        "tickers": list(returns.columns),
    }
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/test_data.py -v`
Expected: 3 passed.

- [ ] **Step 7: Download the real panel (manual, one-time; network required)**

Run: `uv run python -u -c "from src.data import load_etf_panel; p = load_etf_panel(); print(p['tickers']); print(p['returns'].shape)"`
Expected: prints the surviving tickers and a shape like `(~5000, ~10-12)`; creates `data/etf_panel.parquet`. If some tickers are dropped for gaps, that is expected and logged — the survivors define the universe. (This step is not a unit test because it needs network; it is required before Task 5/7 real runs.)

- [ ] **Step 8: Commit**

```bash
git add src/data.py tests/test_data.py .gitignore pyproject.toml uv.lock
git commit -m "feat: ETF data pipeline (yfinance -> cached returns matrix)"
```

---

### Task 2: Real-data market builder (causal signal, no lookahead)

**Files:**
- Create: `src/real_market.py`
- Test: `tests/test_real_market.py`

**Interfaces:**
- Consumes: nothing (operates on a raw returns matrix from Task 1).
- Produces: `build_real_market(returns: np.ndarray, safe_asset_index: int, window: int = 20) -> dict` returning `{"returns": (T, n), "signal": (T,), "safe_asset_index": int}`. The signal is a **causal** crisis-proxy: for each `t`, the trailing realized volatility of the equal-weight portfolio over `returns[t-window:t]` (data strictly before step `t`), z-scored using only trailing history, squashed to `[0, 1]` via a logistic. `signal[t] = 0.5` for `t < window` (no history yet). Higher signal ⇒ more crisis-like ⇒ the observable a de-risking rule would key on.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_real_market.py
import numpy as np
from src.real_market import build_real_market


def test_shapes_and_keys():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, size=(500, 4))
    m = build_real_market(returns, safe_asset_index=3, window=20)
    assert m["returns"].shape == (500, 4)
    assert m["signal"].shape == (500,)
    assert m["safe_asset_index"] == 3
    assert np.all((m["signal"] >= 0.0) & (m["signal"] <= 1.0))


def test_signal_is_causal():
    # THE NO-LOOKAHEAD TEST. Altering returns at/after time t must not change
    # signal[t] — the signal at t may depend only on data strictly before t.
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.01, size=(300, 3))
    base = build_real_market(returns, safe_asset_index=0, window=20)["signal"]

    perturbed = returns.copy()
    perturbed[150:] += 5.0                       # huge change from t=150 onward
    after = build_real_market(perturbed, safe_asset_index=0, window=20)["signal"]

    np.testing.assert_allclose(base[:151], after[:151], atol=1e-12)  # <= t=150 unchanged


def test_signal_rises_in_high_vol_window():
    # Calm then turbulent: signal should be higher in the turbulent tail.
    rng = np.random.default_rng(2)
    calm = rng.normal(0, 0.005, size=(400, 3))
    turbulent = rng.normal(0, 0.05, size=(400, 3))
    returns = np.vstack([calm, turbulent])
    signal = build_real_market(returns, safe_asset_index=0, window=20)["signal"]
    assert signal[600:].mean() > signal[100:380].mean()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_real_market.py -v`
Expected: FAIL with import error on `src.real_market`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/real_market.py
"""Turn a real returns matrix into the AllocationEnv market dict.

The env expects market["returns"] and market["signal"]. On real data the signal
is NOT a look-ahead oracle (that was the synthetic ground truth); it is a CAUSAL
crisis proxy — trailing equal-weight realized volatility — that a transparent
de-risking rule could observe. The gate agent then either learns to time
de-risking off it or (RQ1's H1) fails to beat the base after costs.
"""
import numpy as np


def build_real_market(returns: np.ndarray, safe_asset_index: int, window: int = 20) -> dict:
    returns = np.asarray(returns, dtype=float)
    n_steps, n_assets = returns.shape
    if not 0 <= safe_asset_index < n_assets:
        raise ValueError(f"safe_asset_index {safe_asset_index} out of range for {n_assets} assets")

    eq_weight_return = returns.mean(axis=1)          # 1/N portfolio return per step
    signal = np.full(n_steps, 0.5)                   # neutral until a full window exists

    # Causal: signal[t] uses ONLY returns strictly before t (indices t-window..t-1).
    for t in range(window, n_steps):
        trailing = eq_weight_return[t - window:t]
        trailing_vol = trailing.std()
        history = eq_weight_return[:t]               # all data before t, for standardization
        hist_mean = history.std() if history.size > 1 else trailing_vol
        # z-score current vol against trailing history's vol scale, then logistic-squash.
        scale = hist_mean if hist_mean > 1e-12 else 1e-12
        z = (trailing_vol - scale) / scale
        signal[t] = 1.0 / (1.0 + np.exp(-z))

    return {"returns": returns, "signal": signal, "safe_asset_index": safe_asset_index}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_real_market.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/real_market.py tests/test_real_market.py
git commit -m "feat: real-data market builder with causal crisis-proxy signal"
```

---

### Task 3: Risk-parity base (§5.1 item 3, correlation-aware structural floor)

**Files:**
- Modify: `src/base_policies.py`
- Test: `tests/test_base_policies.py` (append)

**Interfaces:**
- Consumes: nothing new (pure NumPy).
- Produces:
  - `risk_parity_base(return_window: np.ndarray, max_iter: int = 200, tol: float = 1e-8) -> np.ndarray` — long-only equal-risk-contribution (ERC) weights over the trailing window's covariance, on the simplex.
  - `BASE_POLICIES` gains key `"risk_parity"`.
- **Design note (record in `context/decisions.md` at commit):** spec §5.1(3) names an "SPT / diversity-weighted (growth-optimal-oriented)" base. Diversity-weighting needs market-cap weights we do not have, and its growth benefit (rebalancing premium) is already spanned by `equal_weight`. The unspanned structural channel left by equal-weight + inverse-vol is the **cross-asset correlation** structure; ERC/risk-parity closes it transparently and non-learned. We realize the §5.1(3) slot as ERC and note the deviation.

- [ ] **Step 1: Write the failing test (append to `tests/test_base_policies.py`)**

```python
# append to tests/test_base_policies.py
from src.base_policies import risk_parity_base


def test_risk_parity_registered():
    from src.base_policies import BASE_POLICIES
    assert set(BASE_POLICIES.keys()) == {"equal_weight", "vol_scaled", "risk_parity"}


def test_risk_parity_downweights_high_vol_and_equalizes_risk():
    rng = np.random.default_rng(0)
    win = np.column_stack([
        rng.normal(0, 0.005, 200),   # low-vol asset
        rng.normal(0, 0.05, 200),    # high-vol asset (10x vol)
    ])
    w = risk_parity_base(win)
    assert w[0] > w[1]                                   # low-vol asset gets more weight
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-8)
    assert np.all(w >= -1e-9)
    # Equal risk contribution: w_i * (Sigma w)_i should be ~equal across assets.
    cov = np.cov(win, rowvar=False)
    rc = w * (cov @ w)
    np.testing.assert_allclose(rc[0], rc[1], rtol=0.1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_base_policies.py -v`
Expected: FAIL with `cannot import name 'risk_parity_base'`.

- [ ] **Step 3: Add the implementation to `src/base_policies.py`**

Insert after `vol_scaled_base` and update the registry:

```python
def risk_parity_base(return_window: np.ndarray, max_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
    # Equal-risk-contribution (ERC) weights: each asset contributes the same share
    # of portfolio variance. Closes the correlation-structure gap that equal-weight
    # and inverse-vol leave open (spec §5.1 item 3, realized as ERC). Solved by the
    # standard fixed-point iteration on the trailing-window covariance.
    cov = np.cov(return_window, rowvar=False)
    n_assets = cov.shape[0]
    weights = np.full(n_assets, 1.0 / n_assets)
    for _ in range(max_iter):
        marginal = cov @ weights                       # (Sigma w)_i
        marginal = np.where(np.abs(marginal) < 1e-16, 1e-16, marginal)
        updated = 1.0 / marginal                       # target inversely to marginal risk
        updated = updated / updated.sum()
        if np.max(np.abs(updated - weights)) < tol:
            weights = updated
            break
        weights = updated
    return project_to_simplex(weights)
```

Update the registry dict at the bottom:

```python
BASE_POLICIES = {
    "equal_weight": equal_weight_base,
    "vol_scaled": vol_scaled_base,
    "risk_parity": risk_parity_base,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_base_policies.py -v`
Expected: all base-policy tests pass (the original 3 + 2 new).

- [ ] **Step 5: Record the design deviation and commit**

Append the design note above to `context/decisions.md` (newest first), then:

```bash
git add src/base_policies.py tests/test_base_policies.py context/decisions.md
git commit -m "feat: risk-parity (ERC) base policy; realize spec 5.1(3) as ERC"
```

---

### Task 4: Literature baselines + cost-aware weight roller

**Files:**
- Create: `src/baselines.py`
- Test: `tests/test_baselines.py`

**Interfaces:**
- Consumes: `src.simplex.project_to_simplex`, `src.base_policies.equal_weight_base`.
- Produces (each weight fn maps a trailing `(window, n_assets)` return window to next-step target weights, matching the base-policy signature):
  - `minimum_variance_base(return_window: np.ndarray) -> np.ndarray` — long-only min-variance via SLSQP on the trailing covariance.
  - `cvar_min_base(return_window: np.ndarray, alpha: float = 0.95) -> np.ndarray` — long-only CVaR-minimizing weights via the Rockafellar–Uryasev LP over the window's empirical returns (the RMZ/KP-style tail-minimizing strategy, spec §8).
  - `roll_weights(weight_fn, returns: np.ndarray, window: int = 20, cost_bps: float = 10.0) -> np.ndarray` — rolls `weight_fn` causally over `returns`, returns the length-`(T-window)` **net-of-cost simple return** series (turnover charged at `cost_bps`), for feeding `src.metrics`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_baselines.py
import numpy as np
from src.baselines import minimum_variance_base, cvar_min_base, roll_weights
from src.base_policies import equal_weight_base
from src.metrics import expected_shortfall


def test_min_variance_downweights_the_volatile_asset():
    rng = np.random.default_rng(0)
    win = np.column_stack([
        rng.normal(0, 0.005, 300),   # calm asset
        rng.normal(0, 0.05, 300),    # volatile asset
    ])
    w = minimum_variance_base(win)
    assert w[0] > w[1]
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)
    assert np.all(w >= -1e-6)


def test_cvar_min_reduces_tail_vs_equal_weight():
    # One asset has a fat left tail; CVaR-min should avoid it, giving a
    # less-negative expected shortfall than equal weight on the same window.
    rng = np.random.default_rng(1)
    calm = rng.normal(0.0, 0.01, 500)
    fat = rng.standard_t(df=3, size=500) * 0.02      # heavy left tail
    win = np.column_stack([calm, fat])
    w_cvar = cvar_min_base(win, alpha=0.95)
    np.testing.assert_allclose(w_cvar.sum(), 1.0, atol=1e-6)
    assert np.all(w_cvar >= -1e-6)
    es_cvar = expected_shortfall(win @ w_cvar, alpha=0.95)
    es_eq = expected_shortfall(win @ equal_weight_base(win), alpha=0.95)
    assert es_cvar >= es_eq                           # less-negative (better) tail


def test_roll_weights_charges_cost_and_shapes():
    rng = np.random.default_rng(2)
    returns = rng.normal(0, 0.01, size=(120, 3))
    net = roll_weights(equal_weight_base, returns, window=20, cost_bps=10.0)
    assert net.shape == (100,)                        # T - window
    assert np.isfinite(net).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: FAIL with import error on `src.baselines`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/baselines.py
"""Literature baselines (spec §8) and a cost-aware weight roller.

minimum_variance and cvar_min are the standard long-only risk baselines the RL
agent must be measured against; roll_weights turns any weight rule into a
net-of-cost realized return series so src.metrics can score it on the same OOS
window and cost model as the gate agent.
"""
import numpy as np
from scipy.optimize import minimize, linprog

from src.simplex import project_to_simplex


def minimum_variance_base(return_window: np.ndarray) -> np.ndarray:
    cov = np.cov(return_window, rowvar=False)
    n_assets = cov.shape[0]

    def portfolio_variance(weights):
        return float(weights @ cov @ weights)

    start = np.full(n_assets, 1.0 / n_assets)
    constraints = ({"type": "eq", "fun": lambda w: w.sum() - 1.0},)
    bounds = [(0.0, 1.0)] * n_assets
    result = minimize(portfolio_variance, start, method="SLSQP",
                      bounds=bounds, constraints=constraints)
    if not result.success:
        # Do not silently return a bad optimum; fall back to equal weight and log.
        print(f"minimum_variance_base: SLSQP failed ({result.message}); using equal weight")
        return start
    return project_to_simplex(result.x)


def cvar_min_base(return_window: np.ndarray, alpha: float = 0.95) -> np.ndarray:
    # Rockafellar-Uryasev CVaR minimization as an LP. Loss per scenario s is
    # -(returns_s @ w). Variables: [w (n), var (1, the VaR level), u (T, tail slacks)].
    # min  var + 1/((1-alpha)*T) * sum(u)
    # s.t. u_s >= -(returns_s @ w) - var ; u_s >= 0 ; sum(w)=1 ; w>=0.
    scenarios = np.asarray(return_window, dtype=float)
    n_scen, n_assets = scenarios.shape
    n_vars = n_assets + 1 + n_scen

    cost = np.zeros(n_vars)
    cost[n_assets] = 1.0                                       # var coefficient
    cost[n_assets + 1:] = 1.0 / ((1.0 - alpha) * n_scen)       # u coefficients

    # u_s + (returns_s @ w) + var >= 0  ->  -(returns_s@w) - var - u_s <= 0
    a_ub = np.zeros((n_scen, n_vars))
    a_ub[:, :n_assets] = -scenarios
    a_ub[:, n_assets] = -1.0
    a_ub[np.arange(n_scen), n_assets + 1 + np.arange(n_scen)] = -1.0
    b_ub = np.zeros(n_scen)

    a_eq = np.zeros((1, n_vars))
    a_eq[0, :n_assets] = 1.0
    b_eq = np.array([1.0])

    bounds = [(0.0, 1.0)] * n_assets + [(None, None)] + [(0.0, None)] * n_scen
    result = linprog(cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")
    if not result.success:
        print(f"cvar_min_base: LP failed ({result.message}); using equal weight")
        return np.full(n_assets, 1.0 / n_assets)
    return project_to_simplex(result.x[:n_assets])


def roll_weights(weight_fn, returns: np.ndarray, window: int = 20, cost_bps: float = 10.0) -> np.ndarray:
    returns = np.asarray(returns, dtype=float)
    n_steps, n_assets = returns.shape
    cost_rate = cost_bps * 1e-4
    prev_weights = np.full(n_assets, 1.0 / n_assets)
    net_returns = []
    for t in range(window, n_steps):
        win = returns[t - window:t]                    # causal: strictly before t
        weights = weight_fn(win)
        turnover = 0.5 * np.abs(weights - prev_weights).sum()
        gross = float(weights @ returns[t])
        net_returns.append(gross - cost_rate * turnover)
        prev_weights = weights
    return np.asarray(net_returns, dtype=float)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_baselines.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/baselines.py tests/test_baselines.py
git commit -m "feat: min-variance & CVaR-min baselines + cost-aware weight roller"
```

---

### Task 5: Walk-forward harness (per-fold retrain, OOS stitching, restartable)

**Files:**
- Create: `src/walk_forward.py`
- Test: `tests/test_walk_forward.py`

**Interfaces:**
- Consumes: `src.real_market.build_real_market`, `src.train.build_env` + `src.train.train_agent`.
- Produces:
  - `make_folds(n_steps: int, initial_train: int, test_block: int) -> list[tuple[range, range]]` — expanding-window folds: fold `k` trains on `[0, initial_train + k*test_block)` and tests on the next `test_block` steps; the final partial block is included.
  - `roll_policy(model, market: dict, config: dict) -> dict` — rolls the deterministic gate policy over `market`, returns arrays `{"baselined_reward", "port_return", "base_return", "gate"}` (each length `T-window`).
  - `walk_forward_gate(returns: np.ndarray, config: dict, run_dir=None) -> dict` — for each fold: build the train market from the train slice, train the gate, build the test market from the test slice, roll the policy; **stitch** OOS arrays across folds. Returns `{"oos_baselined_reward", "oos_port_return", "oos_base_return", "oos_gate", "fold_mean_skill": list, "mean_skill": float}`. If `run_dir` is given, each fold's arrays are cached to `run_dir/fold_XX.npz` and completed folds are skipped on restart; `mean_skill` is always recomputed over all folds.
  - `config` keys read (in addition to the Plan-1 training keys): `initial_train`, `test_block`, `safe_asset_index`, `window`, `base_name`, `cost_bps`, `total_timesteps`, `seed`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_walk_forward.py
import numpy as np
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_walk_forward.py -v`
Expected: FAIL with import error on `src.walk_forward`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/walk_forward.py
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

    per_fold = []
    for fold_index, (train_slice, test_slice) in enumerate(folds):
        cache = run_dir / f"fold_{fold_index:02d}.npz" if run_dir is not None else None
        if cache is not None and cache.exists():
            print(f"walk_forward_gate: fold {fold_index} cached; loading {cache}")
            per_fold.append(dict(np.load(cache)))
            continue

        train_returns = returns[train_slice.start:train_slice.stop]
        test_returns = returns[test_slice.start:test_slice.stop]
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_walk_forward.py -v -s`
Expected: 2 passed. (Trains a few tiny PPO agents; up to a minute on CPU, seconds on GPU.)

- [ ] **Step 5: Commit**

```bash
git add src/walk_forward.py tests/test_walk_forward.py
git commit -m "feat: expanding-window walk-forward with per-fold retrain and OOS stitching"
```

---

### Task 6: Phase-randomization placebo null (overfitting/luck baseline)

**Files:**
- Create: `src/placebo.py`
- Test: `tests/test_placebo.py`

**Interfaces:**
- Consumes: `src.walk_forward.walk_forward_gate`.
- Produces:
  - `phase_randomize(returns: np.ndarray, seed: int) -> np.ndarray` — returns a surrogate series that preserves each asset's marginal variance and power spectrum and the cross-asset contemporaneous correlation (shared random phases across assets) but **destroys timeable temporal structure**. Same shape as input.
  - `placebo_null(returns: np.ndarray, config: dict, n_placebo: int, seed: int, run_dir=None) -> dict` — runs the full walk-forward on `n_placebo` independent phase-randomized surrogates; returns `{"placebo_skills": list[float], "mean": float, "std": float}`. This is the real-data analog of Plan-1's signal-OFF null (open-questions.md): the skill the pipeline extracts when no timeable structure exists.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_placebo.py
import numpy as np
from src.placebo import phase_randomize, placebo_null


def test_phase_randomize_preserves_variance_and_shape_and_is_deterministic():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.01, size=(512, 3))
    surrogate = phase_randomize(returns, seed=7)
    assert surrogate.shape == returns.shape
    np.testing.assert_allclose(surrogate.std(axis=0), returns.std(axis=0), rtol=0.05)
    again = phase_randomize(returns, seed=7)
    np.testing.assert_allclose(surrogate, again)               # deterministic in seed


def test_phase_randomize_destroys_autocorrelation():
    # A strongly autocorrelated series should lose its lag-1 autocorrelation.
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, size=2000)
    ar = np.zeros(2000)
    for t in range(1, 2000):
        ar[t] = 0.9 * ar[t - 1] + noise[t]                     # strong AR(1)
    returns = ar.reshape(-1, 1)
    surrogate = phase_randomize(returns, seed=3).ravel()

    def lag1(x):
        return np.corrcoef(x[:-1], x[1:])[0, 1]

    assert lag1(ar) > 0.7
    assert abs(lag1(surrogate)) < 0.3                          # structure destroyed


def test_placebo_null_returns_requested_number_of_skills(tmp_path):
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0003, 0.01, size=(900, 3))
    config = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
              "safe_asset_index": 2, "total_timesteps": 1500, "seed": 0,
              "initial_train": 400, "test_block": 250}
    null = placebo_null(returns, config, n_placebo=2, seed=0, run_dir=tmp_path)
    assert len(null["placebo_skills"]) == 2
    assert np.isfinite(null["mean"]) and np.isfinite(null["std"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_placebo.py -v`
Expected: FAIL with import error on `src.placebo`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/placebo.py
"""Phase-randomization placebo: the real-data overfitting/luck null.

Destroys any timeable temporal structure while preserving each asset's marginal
variance, spectrum, and contemporaneous cross-asset correlation. Running the full
walk-forward on these surrogates measures the skill the pipeline manufactures from
noise; the RQ1 headline is real OOS skill NET of this null (open-questions.md).
"""
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.walk_forward import walk_forward_gate


def phase_randomize(returns: np.ndarray, seed: int) -> np.ndarray:
    returns = np.asarray(returns, dtype=float)
    n_steps, n_assets = returns.shape
    rng = np.random.default_rng(seed)

    demeaned = returns - returns.mean(axis=0)
    spectrum = np.fft.rfft(demeaned, axis=0)
    magnitude = np.abs(spectrum)
    n_freq = spectrum.shape[0]

    # SHARED random phases across assets preserve cross-asset correlation.
    random_phase = rng.uniform(0, 2 * np.pi, size=n_freq)
    random_phase[0] = 0.0                                      # keep the DC term real
    if n_steps % 2 == 0:
        random_phase[-1] = 0.0                                 # Nyquist term stays real
    phased = magnitude * np.exp(1j * random_phase)[:, None]
    surrogate = np.fft.irfft(phased, n=n_steps, axis=0)
    return surrogate + returns.mean(axis=0)                    # restore per-asset mean


def placebo_null(returns: np.ndarray, config: dict, n_placebo: int, seed: int, run_dir=None) -> dict:
    run_dir = Path(run_dir) if run_dir is not None else None
    placebo_skills = []
    for draw in tqdm(range(n_placebo), desc="placebo"):
        surrogate = phase_randomize(returns, seed=seed + draw)
        draw_dir = run_dir / f"placebo_{draw:02d}" if run_dir is not None else None
        result = walk_forward_gate(surrogate, {**config, "seed": config["seed"] + draw}, run_dir=draw_dir)
        placebo_skills.append(result["mean_skill"])
        print(f"placebo_null: draw {draw} skill = {result['mean_skill']:.6e}")
    return {
        "placebo_skills": [float(x) for x in placebo_skills],
        "mean": float(np.mean(placebo_skills)),
        "std": float(np.std(placebo_skills)),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_placebo.py -v -s`
Expected: 3 passed. (The last test trains a few tiny agents; a minute or so on CPU.)

- [ ] **Step 5: Commit**

```bash
git add src/placebo.py tests/test_placebo.py
git commit -m "feat: phase-randomization placebo null (real-data overfitting baseline)"
```

---

### Task 7: RQ1 experiment — skill net of null + metrics table + figures

**Files:**
- Create: `src/rq1_real_data.py`
- Test: `tests/test_rq1_real_data.py`

**Interfaces:**
- Consumes: `src.data.load_etf_panel`, `src.walk_forward.walk_forward_gate` + `make_folds`, `src.placebo.placebo_null`, `src.baselines` (`minimum_variance_base`, `cvar_min_base`, `roll_weights`), `src.base_policies` (`equal_weight_base`, `vol_scaled_base`, `risk_parity_base`), `src.metrics`.
- Produces:
  - `compute_metrics_bundle(net_returns: np.ndarray) -> dict` — `{"es_99", "max_drawdown", "tail_ratio", "skewness", "sharpe", "sortino"}` on a net simple-return series.
  - `run_rq1(config: dict, returns: np.ndarray = None, run_dir=None) -> dict` — the RQ1 pipeline. For the base named in `config["base_name"]`: run `walk_forward_gate` (agent OOS) and `placebo_null`; compute `skill_net = mean_skill − placebo_mean` with a CI from `fold_mean_skill` (agent) and `placebo_skills` (null); on the OOS index region, roll the base, 1/N, min-variance, and CVaR-min baselines via `roll_weights` and score each with `compute_metrics_bundle`; also score the stitched agent `oos_port_return`. Writes a traceable output dir with `config.json`, `results.json`, `metrics_table.csv`, and figures `skill_net.png` (skill with CI vs. the null), `cumulative_wealth.png`, `gate_timeseries.png`. If `returns` is None, loads the real panel via `load_etf_panel`.
  - Running `uv run python -u -m src.rq1_real_data` runs the full real-data RQ1 experiment.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rq1_real_data.py
import numpy as np
from src.rq1_real_data import compute_metrics_bundle, run_rq1


def test_metrics_bundle_keys_and_finite():
    rng = np.random.default_rng(0)
    net = rng.normal(0.0003, 0.01, size=1000)
    bundle = compute_metrics_bundle(net)
    assert set(bundle) == {"es_99", "max_drawdown", "tail_ratio", "skewness", "sharpe", "sortino"}
    assert all(np.isfinite(v) for v in bundle.values())


def test_rq1_no_skill_on_structureless_input(tmp_path):
    # INTENT TEST (real-data analog of RQ2's signal-OFF): on i.i.d. noise there is
    # no timeable structure, so the agent's OOS skill must not exceed the placebo
    # null in a way the method would call real. skill_net should be ~0 (within CI).
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0003, 0.01, size=(900, 3))
    config = {"base_name": "equal_weight", "window": 20, "cost_bps": 10.0,
              "safe_asset_index": 2, "total_timesteps": 1500, "seed": 0,
              "initial_train": 400, "test_block": 250, "n_placebo": 2}
    result = run_rq1(config, returns=returns, run_dir=tmp_path)

    assert "skill_net" in result and np.isfinite(result["skill_net"])
    assert "skill_net_ci" in result and len(result["skill_net_ci"]) == 2
    # metrics table must include the agent, the base, and the literature baselines
    assert {"agent", "base", "one_over_n", "min_variance", "cvar_min"}.issubset(result["metrics_table"].keys())
    # On structureless noise the net skill CI must straddle ~0 (no manufactured skill).
    low, high = result["skill_net_ci"]
    assert low <= 5e-5 and high >= -5e-5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rq1_real_data.py -v`
Expected: FAIL with import error on `src.rq1_real_data`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/rq1_real_data.py
"""RQ1: how much of the gate agent's OOS tail/risk benefit is learned skill vs.
inherited structure — on real ETFs, cost-aware, walk-forward.

Headline statistic (open-questions.md): skill NET of the phase-randomized placebo
null, with a confidence interval. A verdict of "skill_net ~ 0" (residual adds
nothing above the structural base after costs) is a valid, publishable RQ1 answer
(spec §9), not a failure. The metrics table places the agent beside its base and
the literature baselines (1/N, min-variance, CVaR-min) on the same OOS window.
"""
import csv
import json
import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_etf_panel
from src.walk_forward import walk_forward_gate, make_folds
from src.placebo import placebo_null
from src.baselines import minimum_variance_base, cvar_min_base, roll_weights
from src.base_policies import equal_weight_base, vol_scaled_base, risk_parity_base
from src.metrics import (expected_shortfall, max_drawdown, tail_ratio,
                         skewness, sharpe, sortino)

BASE_FNS = {
    "equal_weight": equal_weight_base,
    "vol_scaled": vol_scaled_base,
    "risk_parity": risk_parity_base,
}


def compute_metrics_bundle(net_returns: np.ndarray) -> dict:
    net_returns = np.asarray(net_returns, dtype=float)
    return {
        "es_99": expected_shortfall(net_returns, alpha=0.99),
        "max_drawdown": max_drawdown(net_returns),
        "tail_ratio": tail_ratio(net_returns),
        "skewness": skewness(net_returns),
        "sharpe": sharpe(net_returns),
        "sortino": sortino(net_returns),
    }


def _oos_start(n_steps: int, config: dict) -> int:
    # The first index of the stitched OOS region (start of the first test block).
    return make_folds(n_steps, config["initial_train"], config["test_block"])[0][1].start


def run_rq1(config: dict, returns: np.ndarray = None, run_dir=None) -> dict:
    now = datetime.datetime.now()
    print("=" * 60)
    print(f"Run started : {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Script      : {__file__}")
    print(f"Config      : {config}")
    print("=" * 60)

    if returns is None:
        panel = load_etf_panel()
        returns = panel["returns"]
        print(f"run_rq1: loaded real panel {returns.shape}")

    if run_dir is None:
        run_dir = Path(__file__).resolve().parent.parent / "outputs" / f"{now.strftime('%Y-%m-%d_%H-%M-%S')}_rq1-real-data"
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Agent: per-fold retrained gate, stitched OOS.
    agent = walk_forward_gate(returns, config, run_dir=run_dir / "agent")
    null = placebo_null(returns, config, n_placebo=config["n_placebo"],
                        seed=config["seed"] + 100, run_dir=run_dir / "placebo")

    # skill net of the luck/overfitting floor, with a CI combining agent-fold and
    # placebo spread (added in quadrature; both are estimates of a mean skill).
    skill_net = agent["mean_skill"] - null["mean"]
    fold_skills = np.asarray(agent["fold_mean_skill"], dtype=float)
    agent_se = fold_skills.std(ddof=1) / np.sqrt(len(fold_skills)) if len(fold_skills) > 1 else 0.0
    placebo_arr = np.asarray(null["placebo_skills"], dtype=float)
    placebo_se = placebo_arr.std(ddof=1) / np.sqrt(len(placebo_arr)) if len(placebo_arr) > 1 else 0.0
    half_width = 1.96 * float(np.sqrt(agent_se ** 2 + placebo_se ** 2))
    skill_net_ci = [skill_net - half_width, skill_net + half_width]

    # Baselines on the SAME OOS index region and cost model.
    oos_start = _oos_start(returns.shape[0], config)
    oos_returns = returns[oos_start:]
    window = config["window"]
    cost_bps = config["cost_bps"]
    metrics_table = {
        "agent": compute_metrics_bundle(agent["oos_port_return"]),
        "base": compute_metrics_bundle(agent["oos_base_return"]),
        "one_over_n": compute_metrics_bundle(roll_weights(equal_weight_base, oos_returns, window, cost_bps)),
        "min_variance": compute_metrics_bundle(roll_weights(minimum_variance_base, oos_returns, window, cost_bps)),
        "cvar_min": compute_metrics_bundle(roll_weights(cvar_min_base, oos_returns, window, cost_bps)),
    }

    result = {
        "mean_skill": agent["mean_skill"],
        "placebo_mean": null["mean"],
        "skill_net": float(skill_net),
        "skill_net_ci": [float(skill_net_ci[0]), float(skill_net_ci[1])],
        "fold_mean_skill": agent["fold_mean_skill"],
        "placebo_skills": null["placebo_skills"],
        "mean_gate": float(agent["oos_gate"].mean()),
        "metrics_table": metrics_table,
    }

    # Persist config + results + a flat metrics CSV.
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(run_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(run_dir / "metrics_table.csv", "w", newline="") as f:
        metric_names = list(next(iter(metrics_table.values())).keys())
        writer = csv.writer(f)
        writer.writerow(["strategy"] + metric_names)
        for strategy, bundle in metrics_table.items():
            writer.writerow([strategy] + [bundle[name] for name in metric_names])

    # Figure 1: skill net of the null, with CI.
    plt.figure()
    plt.axhline(0.0, color="grey", linewidth=0.8)
    plt.bar(["skill_net"], [skill_net],
            yerr=[[skill_net - skill_net_ci[0]], [skill_net_ci[1] - skill_net]], capsize=6)
    plt.ylabel("OOS skill net of placebo null")
    plt.title("RQ1: learned skill above structure (real ETFs)")
    plt.savefig(run_dir / "skill_net.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Figure 2: cumulative wealth, agent vs base.
    plt.figure()
    plt.plot(np.cumprod(1.0 + agent["oos_port_return"]), label="agent (gate)")
    plt.plot(np.cumprod(1.0 + agent["oos_base_return"]), label="base")
    plt.legend()
    plt.ylabel("cumulative wealth (OOS)")
    plt.title("Agent vs structural base, net of costs")
    plt.savefig(run_dir / "cumulative_wealth.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Figure 3: gate over time (when does the agent de-risk?).
    plt.figure()
    plt.plot(agent["oos_gate"])
    plt.ylabel("de-risking gate g")
    plt.xlabel("OOS step")
    plt.title("Learned de-risking gate over time")
    plt.savefig(run_dir / "gate_timeseries.png", dpi=120, bbox_inches="tight")
    plt.close()

    print(f"run_rq1: mean_skill={agent['mean_skill']:.6e} placebo={null['mean']:.6e} "
          f"skill_net={skill_net:.6e} CI={skill_net_ci}")
    print(f"Saved run to: {run_dir}")
    return result


if __name__ == "__main__":
    _config = {"base_name": "risk_parity", "window": 20, "cost_bps": 10.0,
               "safe_asset_index": None, "total_timesteps": 150_000, "seed": 0,
               "initial_train": 1260, "test_block": 252, "n_placebo": 5}
    # safe_asset_index None -> AllocationEnv defaults to the last asset; set to the
    # Treasury ETF's column index once the surviving universe is known (Task 1 print).
    run_rq1(_config)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_rq1_real_data.py -v -s`
Expected: 2 passed. (Trains several tiny agents across folds and placebo draws; a few minutes on CPU.)

- [ ] **Step 5: Set the safe-asset index and run the real experiment (manual)**

From the Task-1 download print, find the column index of the Treasury ETF (`IEF` preferred; else `TLT`) in the surviving `tickers` list and set `safe_asset_index` in the `__main__` config (and, if desired, sweep `base_name` over `equal_weight`/`vol_scaled`/`risk_parity`). Then:

Run: `uv run python -u -m src.rq1_real_data`
Expected: prints per-fold and placebo skills; writes `outputs/<ts>_rq1-real-data/` with `results.json`, `metrics_table.csv`, and the three figures. The RQ1 verdict is `skill_net` and its CI: a CI straddling 0 supports H1 (benefit is inherited structure); a clearly positive CI is genuine learned skill.

- [ ] **Step 6: Commit**

```bash
git add src/rq1_real_data.py tests/test_rq1_real_data.py
git commit -m "feat: RQ1 real-data experiment (skill net of placebo null + metrics table)"
```

---

## Plan 2 Self-Review

**Spec coverage (spec §8 real-data demonstration + RQ1 §4):**
- Universe: multi-asset liquid ETFs via yfinance, ~2005–2025 → Task 1 (`TICKERS`, `START`, `END`).
- Anchored walk-forward, no full-sample tuning → Task 5 (`make_folds` expanding-window, per-fold retrain).
- Turnover-aware transaction costs from day one → env reward (Plan 1) + `roll_weights` cost charge (Task 4).
- Tail/risk metrics (99% ES/CVaR, max drawdown, tail ratio, skew, Sharpe, Sortino, turnover) → `compute_metrics_bundle` (Task 7) over `src.metrics`.
- Baselines: structural ladder (equal-weight, vol-scaled, risk-parity) + 1/N + minimum-variance + RMZ/KP tail-minimizing → Tasks 3, 4, 7.
- SPT/growth-optimal base added to the ladder (spec §5.1 item 3) → Task 3 (realized as ERC/risk-parity; deviation documented).
- RQ1 skill-vs-structure verdict, reported **net of the matched null** with CIs (open-questions.md) → Tasks 6 + 7.
- Traceability / uv / no-silent-failure / no-lookahead / restartable → Global Constraints, embedded in Tasks 1, 2, 5, 6, 7.
- **Deferred (correctly out of Plan-2 scope):** MC2 causal probing / SHAP comparison (Plan 3, RQ3); algorithm zoo beyond PPO (spec §12 YAGNI).

**Placeholder scan:** none — every code step is runnable; every command has expected output. RL/real-data tests use plumbing + property + intent assertions (shapes, causality/no-lookahead, sign relationships, restartability, and the structureless-input "no manufactured skill" intent test), which is the correct verification for stochastic RL on non-synthetic data.

**Type consistency:** base/baseline weight fns all share the `(window, n_assets) -> (n_assets,)` signature (Tasks 3, 4) and are consumed identically by `roll_weights` (Task 4) and the env's `BASE_POLICIES` (Task 5 via train). `build_real_market` returns `returns`/`signal`/`safe_asset_index` (Task 2), consumed by `walk_forward_gate` (Task 5). `walk_forward_gate` returns `oos_*`/`fold_mean_skill`/`mean_skill` (Task 5), consumed by `placebo_null` (Task 6, uses `mean_skill`) and `run_rq1` (Task 7, uses all). `config` keys (`initial_train`, `test_block`, `safe_asset_index`, `n_placebo`, plus Plan-1 training keys) are used consistently across Tasks 5–7. `compute_metrics_bundle` keys match the CSV writer and the test. Consistent.

---

## Next plan (not yet written)
- **Plan 3 — MC2 causal probing (RQ3):** intervention harness (vol shocks, regime flips, feature freeze/permute) on the trained gate agent + SHAP/saliency comparison and a causal-faithfulness verdict. Write after Plan 2 executes; staged so a slip degrades gracefully to the MC1 (synthetic) + RQ1 (real-data) paper.
