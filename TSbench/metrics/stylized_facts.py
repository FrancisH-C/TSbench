"""Stylized facts metrics for evaluating synthetic time series quality.

These metrics test whether generated series exhibit the statistical properties
(stylized facts) commonly observed in financial returns: fat tails, volatility
clustering, and absence of linear autocorrelation with presence of nonlinear
dependence.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def excess_kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (fat tails test).

    Financial returns typically have excess kurtosis > 0, indicating
    heavier tails than a Gaussian distribution (Cont, 2001).

    Args:
        x: 1-D array of returns.

    Returns:
        Excess kurtosis (kurtosis - 3).

    """
    x = np.asarray(x, dtype=float).ravel()
    return float(stats.kurtosis(x, fisher=True))


def volatility_clustering(x: np.ndarray, max_lag: int = 10) -> np.ndarray:
    """Autocorrelation of squared returns (volatility clustering).

    Financial returns show little autocorrelation in raw values but
    significant autocorrelation in squared (or absolute) returns,
    indicating volatility clustering (Cont, 2001).

    Args:
        x: 1-D array of returns.
        max_lag: Maximum lag for autocorrelation computation.

    Returns:
        Array of autocorrelations of x² at lags 1..max_lag.

    """
    x = np.asarray(x, dtype=float).ravel()
    x2 = x**2
    x2_centered = x2 - np.mean(x2)
    var = np.var(x2)
    if var == 0:
        return np.zeros(max_lag)
    n = len(x2)
    acf = np.array(
        [
            np.sum(x2_centered[: n - lag] * x2_centered[lag:]) / (n * var)
            for lag in range(1, max_lag + 1)
        ]
    )
    return acf


def ljung_box(x: np.ndarray, max_lag: int = 10) -> tuple[float, float]:
    """Ljung-Box test for autocorrelation.

    Tests the null hypothesis that the first ``max_lag`` autocorrelations
    are jointly zero. A low p-value indicates significant autocorrelation.

    Args:
        x: 1-D array of values.
        max_lag: Number of lags to test.

    Returns:
        Tuple of (test statistic Q, p-value).

    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    x_centered = x - np.mean(x)
    var = np.var(x)
    if var == 0 or n <= max_lag:
        return 0.0, 1.0
    acf = np.array(
        [
            np.sum(x_centered[: n - lag] * x_centered[lag:]) / (n * var)
            for lag in range(1, max_lag + 1)
        ]
    )
    Q = float(n * (n + 2) * np.sum(acf**2 / np.arange(n - 1, n - max_lag - 1, -1)))
    p_value = float(1.0 - stats.chi2.cdf(Q, df=max_lag))
    return Q, p_value
