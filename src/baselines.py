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
        prev_weights = weights   # turnover measured vs chosen weights; intra-period drift not tracked (deliberate simplification)
    return np.asarray(net_returns, dtype=float)
