#!/usr/bin/env python3
"""
0-norm_constants.py
"""
import numpy as np


def constantes_normalizacion(X):
    """
    Calculate the normalization constants (mean and standard deviation)
    of each characteristic in the matrix.

    Parameters:
    X (list of lists): Data array (m, nx)

    Returns:
    tuple: mean and standard deviation of each feature (nx,)
    """
    m = len(X)
    nx = len(X[0])

    medias = [0] * nx
    desviaciones = [0] * nx

    for j in range(nx):
        suma_columna = 0
        for i in range(m):
            suma_columna += X[i][j]
        medias[j] = suma_columna / m

    for j in range(nx):
        suma_varianza = 0
        for i in range(m):
            suma_varianza += (X[i][j] - medias[j]) ** 2
        desviaciones[j] = (suma_varianza / m) ** 0.5

    return medias, desviaciones
