"""
Data scaling utilities for numerical stability in EM fitting.

Provides median / MAD (Median Absolute Deviation) normalization that
prevents extreme parameter magnitudes when fitting mixture distributions
to data with very different column scales.
"""

import numpy as np
from numpy.typing import NDArray


def column_median_mad(X: NDArray) -> tuple[NDArray, NDArray]:
    """
    Compute per-column median and MAD (Median Absolute Deviation).

    MAD is defined as :math:`\\text{median}(|X_j - \\text{median}(X_j)|)` for
    each column :math:`j`, scaled by 1.4826 to be a consistent estimator of
    the standard deviation for normal data.

    Columns with zero MAD (constant columns) get MAD = 1 to avoid division
    by zero.

    Parameters
    ----------
    X : ndarray, shape (n, d)
        Data matrix.

    Returns
    -------
    median : ndarray, shape (d,)
        Per-column medians.
    mad : ndarray, shape (d,)
        Per-column scaled MAD values (positive, at least 1e-15).
    """
    median = np.median(X, axis=0)
    mad = 1.4826 * np.median(np.abs(X - median), axis=0)
    mad = np.where(mad < 1e-15, 1.0, mad)
    return median, mad
