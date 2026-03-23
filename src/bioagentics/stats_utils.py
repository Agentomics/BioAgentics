"""Shared statistical utility functions."""

from __future__ import annotations

import numpy as np


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvalues : np.ndarray
        Raw p-values.

    Returns
    -------
    np.ndarray
        Adjusted p-values clipped to [0, 1].
    """
    n = len(pvalues)
    if n == 0:
        return np.array([])

    ranked = np.argsort(pvalues)
    adjusted = np.ones(n)
    for i, rank_idx in enumerate(reversed(ranked)):
        rank = n - i
        if i == 0:
            adjusted[rank_idx] = min(1.0, pvalues[rank_idx] * n / rank)
        else:
            prev_idx = ranked[n - i]
            adjusted[rank_idx] = min(adjusted[prev_idx], pvalues[rank_idx] * n / rank)

    return np.clip(adjusted, 0, 1)
