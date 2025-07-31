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
    """
    Performs K-means clustering on a dataset.

    Parameters:
    X : numpy.ndarray of shape (n, d)
        The dataset, where n is the number of data points and
        d is the number of dimensions.
    k : int
        The number of clusters to form.
    iterations : int, optional (default=1000)
        The maximum number of iterations to perform.

    Returns:
    C : numpy.ndarray of shape (k, d)
        The final cluster centroids.
    clss : numpy.ndarray of shape (n,)
        The cluster index assigned to each data point.
    Returns (None, None) if the algorithm fails.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    C = initialize(X, k)
    if C is None:
        return None, None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    clss = np.zeros(X.shape[0], dtype=int)

    for _ in range(iterations):
        X_vectors = np.repeat(X[:, np.newaxis], k, axis=1)
        X_vectors = np.reshape(X_vectors, (X.shape[0], k, X.shape[1]))
        C_vectors = np.tile(C[np.newaxis, :], (X.shape[0], 1, 1))
        C_vectors = np.reshape(C_vectors, (X.shape[0], k, X.shape[1]))
        # Calculate Euclidean distances
        distances = np.linalg.norm(X_vectors - C_vectors, axis=2)
        new_clss = np.argmin(distances, axis=1)
        old_C = C.copy()
        # Update centroids
        for j in range(k):
            mask = (new_clss == j)
            if np.any(mask):
                C[j] = X[mask].mean(axis=0)
            else:
                C[j] = np.random.uniform(
                    low=min_vals, high=max_vals, size=X.shape[1])

        if np.all(C == old_C):
            return C, clss
        C_vectors = np.tile(C, (X.shape[0], 1))
        C_vectors = C_vectors.reshape(X.shape[0], k, X.shape[1])
        distance = np.linalg.norm(X_vectors - C_vectors, axis=2)
        clss = np.argmin(distance ** 2, axis=1)

    return C, clss