#!/usr/bin/env python3
"""
6-expectation.py
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Performs the expectation step in the EM algorithm for a GMM.
    
    Returns:
        g: posterior probabilities (k, n)
        l: total log likelihood
    """
    try:
        n, d = X.shape
        k = pi.shape[0]

        # Creamos una matriz vacía para guardar las probabilidades
        g = np.zeros((k, n))

        # iterar sobre los k clústeres
        for i in range(k):
            g[i] = pi[i] * pdf(X, m[i], S[i])

        # Sumar todas las probabilidades por punto
        total = np.sum(g, axis=0)

        # Normalizamos dividiendo cada fila por la suma de su columna
        g /= total

        # Log likelihood: suma del log de la suma de probabilidades por punto
        l = np.sum(np.log(total))

        return g, l

    except Exception:
        return None, None
