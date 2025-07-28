#!/usr/bin/env python3
"""
0-initialize.py
"""
import numpy as np


def initialize(X, k):
    """
    Initialize cluster centroids for K-means using a multivariate
    uniform distribution.

    Parameters:
    -----------
    X : numpy.ndarray of shape (n, d)
        Dataset containing n data points with d dimensions.
    k : int
        Positive integer representing the number of clusters.

    Returns:
    --------
    numpy.ndarray of shape (k, d)
        Array containing the initialized centroids for each cluster.
        Each centroid is sampled uniformly between the minimum and maximum
        values of X along each dimension.
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None

    # obtengo el maximo y minimo valor de cada columna de x
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)

    # Generar k centroides aleatorios dentro del rango y
    # que tenga d dimensiones
    centroids = np.random.uniform(
        low=min_val, high=max_val, size=(k, X.shape[1]))

    return centroids
