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
