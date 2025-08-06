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
    if not np.allclose(P.sum(axis=1), 1):
        return None
    if np.any(P < 0) or np.any(P > 1):
        return None
    n = P.shape[0]
    try:
        P_T = P.T
        Id = np.eye(n)
        A = P_T - Id

        A = np.vstack([A, np.ones(n)])

        b = np.zeros(n + 1)
        b[-1] = 1

        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        return x.reshape(1, n)

    except Exception:
        return None
