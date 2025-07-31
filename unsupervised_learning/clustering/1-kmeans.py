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
    try:
        centroids = initialize(X, k)
        for _ in range(iterations):
            old_centroids = centroids.copy()
            
            # Calcular distancias (broadcasting correcto)
            distances = np.linalg.norm(X[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            
            # Asignar puntos al centroide más cercano
            labels = np.argmin(distances, axis=1)

            # Actualizar centroides
            for i in range(k):
                cluster_points = X[labels == i]
                if cluster_points.size == 0:
                    centroids[i] = initialize(X, 1)[0]
                else:
                    centroids[i] = np.mean(cluster_points, axis=0)

            # Verificar convergencia
            if np.allclose(old_centroids, centroids):
                break

        return centroids, labels

    except Exception:
        return None, None
