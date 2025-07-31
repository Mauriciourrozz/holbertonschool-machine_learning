#!/usr/bin/env python3
"""
4-initialize.py
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializate pi, m and S to GMM
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0 or k >= X.shape[0]:
        return None, None, None

    pi = np.full((k,), 1 / k)
    m, _ = kmeans(X, k)
    d = X.shape[1]
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
