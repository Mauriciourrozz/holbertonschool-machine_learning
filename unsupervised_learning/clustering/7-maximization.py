#!/usr/bin/env python3
"""
7-maximization.py
"""
import numpy as np


def maximization(X, g):
    """
    Performs the maximization step in the EM algorithm for a GMM.

    Parameters:
    - X: (n, d) data points
    - g: (k, n) posterior probabilities

    Returns:
    - pi: (k,) updated priors
    - m: (k, d) updated means
    - S: (k, d, d) updated covariances
    """
    try:
        if (not isinstance(X, np.ndarray) or not isinstance(g, np.ndarray) or
            len(X.shape) != 2 or len(g.shape) != 2):
            return None, None, None

        n, d = X.shape
        k, n2 = g.shape

        if n != n2:
            return None, None, None

        # Sumamos las responsabilidades por cluster
        Nk = np.sum(g, axis=1)

        # Nuevas medias: m_j = sum_i (gamma_ji * x_i) / Nk_j
        m = np.dot(g, X) / Nk[:, np.newaxis]

        # Nuevas covarianzas
        S = np.zeros((k, d, d))
        for i in range(k):
            diff = X - m[i]
            weighted = g[i][:, np.newaxis] * diff
            S[i] = np.dot(weighted.T, diff) / Nk[i]

        # Nuevos priors
        pi = Nk / n

        return pi, m, S

    except Exception:
        return None, None, None
