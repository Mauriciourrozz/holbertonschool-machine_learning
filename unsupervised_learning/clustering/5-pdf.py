#!/usr/bin/env python3
"""
5-pdf.py
"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the PDF of a multivariate normal distribution.

    Parameters:
    - X: np.ndarray of shape (n, d), data points
    - m: np.ndarray of shape (d,), mean vector
    - S: np.ndarray of shape (d, d), covariance matrix

    Returns:
    - P: np.ndarray of shape (n,), PDF values for each point
    """
    if not isinstance(X, np.ndarray) or not isinstance(
            m, np.ndarray) or not isinstance(S, np.ndarray):
        return None
    if len(X.shape) != 2 or len(m.shape) != 1 or len(S.shape) != 2:
        return None
    n, d = X.shape
    if m.shape[0] != d or S.shape != (d, d):
        return None

    # Restar la media
    diff = X - m

    # Inversa y determinante de S
    try:
        inv_S = np.linalg.inv(S)
        det_S = np.linalg.det(S)
    except np.linalg.LinAlgError:
        return None

    # Exponente de la función gaussiana
    exponent = np.einsum('ij,jk,ik->i', diff, inv_S, diff)

    # Denominador: normalización
    denom = np.sqrt((2 * np.pi) ** d * det_S)

    # PDF final
    P = np.exp(-0.5 * exponent) / denom

    # Asegurarse de que ningún valor sea menor que 1e-300
    P = np.maximum(P, 1e-300)

    return P
