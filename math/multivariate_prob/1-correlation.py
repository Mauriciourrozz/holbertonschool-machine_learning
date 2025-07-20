#!/usr/bin/env python3
"""
1-correlation.py
"""
import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix from a covariance matrix.

    Parameters:
    - C: numpy.ndarray of shape (d, d)

    Returns:
    - numpy.ndarray of shape (d, d) containing the correlation matrix
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    stddev = np.sqrt(np.diag(C))
    outer_stddev = np.outer(stddev, stddev)

    corr = C / outer_stddev
    corr[np.isnan(corr)] = 0

    return corr
