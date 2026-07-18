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
