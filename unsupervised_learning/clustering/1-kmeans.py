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
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None

    # Inicializo centroides
    C = initialize(X, k)

    for i in range(iterations):
        # Calcular distancias y asignar puntos al centroide más cercano
        distancias = np.linalg.norm(X[:, np.newaxis, :] - C[np.newaxis, :, :], axis=2)
        clss = np.argmin(distancias, axis=1)

        # Guardar copia de los centroides actuales
        C_prev = C.copy()

        # Actualizar centroides
        for j in range(k):
            puntos = X[clss == j]

            if len(puntos) > 0:
                C[j] = np.mean(puntos, axis=0)
            else:
                # Reinicializar centroide si no tiene puntos asignados
                C[j] = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0))

        # Comparar si los centroides cambiaron
        if np.allclose(C, C_prev):
            break

    # Verificar y devolver resultados
    if C.shape == (k, X.shape[1]) and clss.shape == (X.shape[0],):
        return C, clss
    else:
        return None, None
