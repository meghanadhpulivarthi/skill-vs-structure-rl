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


# Config — risky+safe world (asset 0 = risky, asset 1 = safe haven)
# Used for the minimal RQ2 validation: the gate can de-risk from risky into safe.
RISKY_CALM_DRIFT = 0.0006
RISKY_CALM_VOL = 0.008
RISKY_CRISIS_DRIFT = -0.004
RISKY_CRISIS_VOL = 0.030
SAFE_DRIFT = 0.00005   # safe haven: tiny positive drift, both regimes
SAFE_VOL = 0.001       # and very low volatility, no crash


def generate_risky_safe_market(
    n_steps: int,
    seed: int,
    signal_strength: float,
    crisis_persistence: float = 0.9,
    calm_persistence: float = 0.98,
) -> dict:
    """Minimal 2-asset world: asset 0 risky (crashes in crisis), asset 1 safe.

    Same regime/signal machinery as generate_market. The gate de-risks from the
    risky asset toward the safe asset; timing it with `signal` is the only source
    of skill, so signal_strength=0 makes skill impossible (the RQ2 null).
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

    returns = np.zeros((n_steps, 2))
    for t in range(n_steps):
        if regime[t] == 0:
            returns[t, 0] = rng.normal(RISKY_CALM_DRIFT, RISKY_CALM_VOL)
        else:
            returns[t, 0] = rng.normal(RISKY_CRISIS_DRIFT, RISKY_CRISIS_VOL)
        returns[t, 1] = rng.normal(SAFE_DRIFT, SAFE_VOL)

    return {"returns": returns, "signal": signal, "regime": regime}
