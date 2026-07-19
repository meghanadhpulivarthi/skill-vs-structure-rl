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
