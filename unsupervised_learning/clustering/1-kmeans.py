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

    if not isinstance(k, int) or k <= 0:
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
    # Validaciones iniciales
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
            not isinstance(k, int) or k <= 0 or
            not isinstance(iterations, int) or iterations <= 0):
        return None, None

    n, d = X.shape
    centroids = initialize(X, k)
    if centroids is None:
        return None, None

    for _ in range(iterations):
        # Calcular las distancias de cada punto a cada centroide
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        # Asignar cada punto al centroide más cercano
        labels = np.argmin(distances, axis=1)

        # Guardar una copia de los centroides actuales para comparar cambios
        prev_centroids = centroids.copy()

        # Actualizar los centroides
        for i in range(k):
            puntos = X[labels == i]
            if puntos.shape[0] == 0:
                # Si un cluster quedó vacío, se reinicializa su centroide
                centroids[i] = initialize(X, 1)[0]
            else:
                # Calcular el nuevo centroide como el promedio de sus puntos
                centroids[i] = np.mean(puntos, axis=0)

        # Verificar si los centroides dejaron de cambiar (convergencia)
        if np.allclose(centroids, prev_centroids):
            break

    return centroids, labels
