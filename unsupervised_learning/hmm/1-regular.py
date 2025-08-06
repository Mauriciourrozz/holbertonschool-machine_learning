#!/usr/bin/env python3
"""
1-regular.py
"""
import numpy as np


def regular(P):
    """
    Calculates the steady state probabilities of a regular Markov chain.

    Parameters:
    P (numpy.ndarray): Square 2D array of shape (n, n) representing
    the transition matrix.

    Returns:
    numpy.ndarray: Row vector of shape (1, n) with steady state probabilities,
                   or None if input is invalid or computation fails.
    """
    try:
        # Validaciones
        if not isinstance(P, np.ndarray):
            return None
        if P.ndim != 2:
            return None
        n, m = P.shape
        if n != m:
            return None
        if not np.allclose(np.sum(P, axis=1), 1.0):
            return None
        if (P <= 0).any() or (P >= 1).any():
            return None

        # Verificación de regularidad
        P_pow = np.linalg.matrix_power(P, n * n)
        if (P_pow <= 0).any():
            return None

        # Armado del sistema lineal
        A = P.T - np.identity(n)
        A = np.concatenate((A, np.ones((1, n))), axis=0)
        b = np.zeros(n + 1)
        b[-1] = 1

        # Resolución por mínimos cuadrados
        pi, *_ = np.linalg.lstsq(A, b, rcond=None)
        return pi[np.newaxis, :]
    except Exception:
        return
