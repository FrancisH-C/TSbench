"""Metric utilities for converting between regression and classification."""

from __future__ import annotations

import numpy as np


def get_bins(values: np.ndarray, number_of_bins: int) -> np.ndarray:
    """Compute percentile-based bin edges for discretizing continuous values.

    Args:
        values: 1-D array of continuous values.
        number_of_bins: Number of bins (edges returned = ``number_of_bins - 1``).

    Returns:
        Array of bin edge values.

    """
    number_of_edges = number_of_bins - 1
    mult = 100 // number_of_bins
    percentiles = list(range(mult, mult * number_of_edges + mult, mult))
    bins_value = np.percentile(values, percentiles)
    # Ensure unique edges by extending duplicates
    while len(set(bins_value)) != number_of_edges:
        bins_value = np.insert(
            bins_value, -1, bins_value[-1] + (1 - bins_value[-1] / 2)
        )
    return bins_value


def regression_to_classification(
    values: np.ndarray,
    bins_value: np.ndarray | None = None,
    number_of_bins: int = 3,
) -> np.ndarray:
    """Convert continuous values to one-hot encoded bin vectors.

    When ``bins_value`` is ``None``, default half-tick bin edges are
    generated (suitable for mid-price prediction).

    Args:
        values: 1-D array of continuous values.
        bins_value: Pre-computed bin edges, or ``None`` for defaults.
        number_of_bins: Number of bins (used only when ``bins_value`` is ``None``).

    Returns:
        Array of shape ``(len(values), number_of_bins)`` with one-hot encoding.

    """
    if bins_value is None:
        number_of_edges = number_of_bins - 1
        tick = 0.005
        edges = [
            tick / 2 * (bin_index - number_of_edges // 2)
            for bin_index in range(number_of_edges)
        ]
        for i in range(len(edges)):
            if edges[i] < 0:
                edges[i] -= 10 ** (-4)
            if edges[i] > 0:
                edges[i] += 10 ** (-4)
        bins_value = np.asarray(edges, dtype=float)
    else:
        number_of_bins = len(bins_value) + 1

    n = values.shape[0]
    bins = np.zeros([n, number_of_bins])
    for i in range(n):
        j = np.searchsorted(bins_value, values[i])
        bins[i][j] = 1
    return bins
