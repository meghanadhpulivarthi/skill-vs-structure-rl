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


BASE_POLICIES = {
    "equal_weight": equal_weight_base,
    "vol_scaled": vol_scaled_base,
    "risk_parity": risk_parity_base,
}
