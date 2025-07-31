#!/usr/bin/env python3
"""
2-variance.py
"""
import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance without loops.
    X: numpy.ndarray with shape (n, d) - data
    C: numpy.ndarray with shape (k, d) - centroids
    Returns the total variance or None on failure.
    """
    try:
        # calcular distancias entre cada punto y cada centroide
        diff = X[:, np.newaxis, :] - C[np.newaxis, :, :]

        # Calculo la distancia al cuadrado (norma al cuadrado) en cada dimensión
        # Sumo por dimensión para obtener matriz (n, k) de distancias al cuadrado
        dist_sq = np.sum(diff ** 2, axis=2)

        # para cada punto asigno el índice del centroide más cercano
        clss = np.argmin(dist_sq, axis=1)

        # tomo la distancia al cuadrado al centroide asignado para cada punto
        # Para eso, uso dist_sq y clss para seleccionar solo la distancia del cluster asignado
        min_dist_sq = dist_sq[np.arange(X.shape[0]), clss]

        # Sumo todas las distancias al cuadrado para obtener la varianza total
        return np.sum(min_dist_sq)


    except Exception:
        return None