#!/usr/bin/env python3
"""
6-expectation.py
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Performs the expectation step in the EM algorithm for a GMM.

    Parameters:
    - X: (n, d) data
    - pi: (k,) priors
    - m: (k, d) means
    - S: (k, d, d) covariances

    Returns:
    - g: (k, n) posterior probabilities
    - l: total log likelihood
    """
    try:
        n, d = X.shape
        k = pi.shape[0]

        # Validaciones básicas
        if (pi.shape != (k,) or
            m.shape != (k, d) or
            S.shape != (k, d, d) or
            not np.isclose(np.sum(pi), 1)):
            return None, None

        g = np.zeros((k, n))

        for i in range(k):
            P = pdf(X, m[i], S[i])
            if P is None:
                return None, None
            g[i] = pi[i] * P

        total = np.sum(g, axis=0)

        # Verificar que ninguna probabilidad total sea cero
        if np.any(total == 0):
            return None, None

        # Normalizar
        g /= total

        # Log likelihood
        l = np.sum(np.log(total))

        if np.isnan(l) or np.isinf(l) or np.any(np.isnan(g)) or np.any(np.isinf(g)):
            return None, None

        return g, l

    except Exception:
        return None, None
