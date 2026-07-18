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
