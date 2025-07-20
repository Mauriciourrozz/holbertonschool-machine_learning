#!/usr/bin/env python3
"""
0-mean_cov.py
"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set.

    Parameters:
    - X: numpy.ndarray of shape (n, d)

    Returns:
    - mean: numpy.ndarray of shape (1, d)
    - cov: numpy.ndarray of shape (d, d)
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")

    mean = np.mean(X, axis=0, keepdims=True)
    x = X - mean
    cov = np.matmul(x.T, x) / (n - 1)

    return mean, cov
