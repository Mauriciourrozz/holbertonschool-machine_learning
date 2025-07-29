#!/usr/bin/env python3
"""
1-kmeans.py
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


def kmeans(X, k, iterations=1000):
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None

    np.random.seed(0)  # fijar semilla para reproducibilidad

    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)

    C = np.random.uniform(low=min_val, high=max_val, size=(k, X.shape[1]))  # inicializar centroides

    for i in range(iterations):
        distancias = np.linalg.norm(X[:, np.newaxis, :] - C[np.newaxis, :, :], axis=2)
        clss = np.argmin(distancias, axis=1)

        C_prev = C.copy()

        for j in range(k):
            puntos = X[clss == j]

            if len(puntos) > 0:
                C[j] = np.mean(puntos, axis=0)
            else:
                C[j] = np.random.uniform(low=min_val, high=max_val)  # reinicializar centroide vacio

        if np.allclose(C, C_prev):
            break

    return C, clss
