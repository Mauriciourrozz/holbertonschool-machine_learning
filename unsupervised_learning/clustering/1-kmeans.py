#!/usr/bin/env python3
"""
1-kmeans.py
"""
import numpy as np


def initialize(X, k):
    """
    Initialize cluster centroids for K-means using a multivariate
    uniform distribution.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None, None

    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)

    centroids = np.random.uniform(
        low=min_val, high=max_val, size=(k, X.shape[1]))

    return centroids, min_val, max_val


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Inicializo centroides y valores para reuso
    C, min_val, max_val = initialize(X, k)
    if C is None:
        return None, None

    for i in range(iterations):
        dist = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        clss = np.argmin(dist, axis=1)

        C_prev = C.copy()

        for j in range(k):
            puntos = X[clss == j]
            if puntos.size > 0:
                C[j] = np.mean(puntos, axis=0)

        empty = np.where(np.bincount(clss, minlength=k) == 0)[0]
        if empty.size > 0:
            C[empty] = np.random.uniform(low=min_val, high=max_val, size=(len(empty), X.shape[1]))

        if np.allclose(C, C_prev):
            break

    return C, clss
