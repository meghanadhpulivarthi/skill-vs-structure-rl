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


def risk_parity_base(return_window: np.ndarray, max_iter: int = 200, tol: float = 1e-8) -> np.ndarray:
    # Equal-risk-contribution (ERC) weights: each asset contributes the same share
    # of portfolio variance. Closes the correlation-structure gap that equal-weight
    # and inverse-vol leave open (spec §5.1 item 3, realized as ERC). Uses the
    # sqrt-damped multiplicative fixed-point iteration: it shares the ERC fixed
    # point w_i*(Sigma w)_i = const with the raw 1/(Sigma w) map but, unlike that
    # map, does not oscillate (the undamped map flips between corner and center
    # allocations for high vol-ratio diagonal covariances and never converges).
    cov = np.cov(return_window, rowvar=False)
    n_assets = cov.shape[0]
    weights = np.full(n_assets, 1.0 / n_assets)
    converged = False
    change = np.inf
    for _ in range(max_iter):
        marginal = cov @ weights                       # (Sigma w)_i
        if np.any(marginal < 0):
            # (Sigma w)_i < 0 makes ERC ill-defined (sqrt of a negative). Fall back
            # loudly to inverse-variance weights (exact ERC when cov is diagonal).
            print("risk_parity_base: negative marginal risk encountered; "
                  "falling back to inverse-variance weights")
            diag = np.diag(cov)
            inv_var = 1.0 / np.where(diag < 1e-16, 1e-16, diag)
            return project_to_simplex(inv_var / inv_var.sum())
        marginal = np.where(marginal < 1e-16, 1e-16, marginal)
        updated = np.sqrt(weights / marginal)          # sqrt damping -> stable ERC fixed point
        updated = updated / updated.sum()
        change = np.max(np.abs(updated - weights))
        weights = updated
        if change < tol:
            converged = True
            break
    if not converged:
        print(f"risk_parity_base: ERC iteration did not converge in {max_iter} iters "
              f"(last weight change {change:.2e}); returning last iterate")
    return project_to_simplex(weights)


BASE_POLICIES = {
    "equal_weight": equal_weight_base,
    "vol_scaled": vol_scaled_base,
    "risk_parity": risk_parity_base,
}
