#!/usr/bin/env python3
"""
3-optimum.py
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Finds the optimal number of clusters based on the intra-cluster variance.

    Parameters:
    - X: numpy.ndarray of shape (n, d), dataset
    - kmin: positive integer, minimum number of clusters (inclusive)
    - kmax: positive integer, maximum number of clusters (inclusive)
    - iterations: positive integer, maximum number of iterations for K-means

    Returns:
    - results: list of K-means outputs for each cluster size
    - d_vars: list of the variance difference with respect to the minimum
        number of clusters
    - or (None, None) on error
    """

    # Valido que X sea un ndarray 2D
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    n = X.shape[0]

    # Valido kmin y kmax
    if not isinstance(kmin, int) or kmin <= 0 or kmin >= n:
        return None, None

    if kmax is None:
        kmax = n

    if not isinstance(kmax, int) or kmax <= 0 or kmax > n or kmax < kmin + 1:
        return None, None

    results = []
    d_vars = []

    # Iterar sobre cada número de clusters desde kmin hasta kmax
    for k in range(kmin, kmax + 1):
        # Ejecutar kmeans con el número actual de clusters
        C, clss = kmeans(X, k, iterations)
        if C is None or clss is None:
            return None, None

        # Guardar resultados
        results.append((C, clss))

        # Calcular varianza total con la función variance
        var = variance(X, C)
        if var is None:
            return None, None

        d_vars.append(var)

    # Calcular diferencia de varianza respecto a varianza mínima
    base_var = d_vars[0]
    d_vars = [base_var - v for v in d_vars]

    return results, d_vars
