#!/usr/bin/env python3
"""
5-definiteness.py
"""
import numpy as np


def definiteness(matrix):
    """
    Determines the definiteness of a square numpy matrix.

    Args:
        matrix (numpy.ndarray): A square numpy array of shape (n, n).

    Raises:
        TypeError: If matrix is not a numpy.ndarray.

    Returns:
        str or None: One of
            'Positive definite',
            'Positive semi-definite',
            'Negative semi-definite',
            'Negative definite',
            'Indefinite',
        or None if the matrix is invalid or does not fit any category.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None

    if not np.allclose(matrix, matrix.T, atol=1e-8):
        return None

    eigvals = np.linalg.eigvalsh(matrix)

    pos = np.all(eigvals > 0)
    pos_semi = np.all(eigvals >= 0) and not pos
    neg = np.all(eigvals < 0)
    neg_semi = np.all(eigvals <= 0) and not neg

    if pos:
        return "Positive definite"
    elif pos_semi:
        return "Positive semi-definite"
    elif neg:
        return "Negative definite"
    elif neg_semi:
        return "Negative semi-definite"
    elif np.any(eigvals > 0) and np.any(eigvals < 0):
        return "Indefinite"
    else:
        return None
