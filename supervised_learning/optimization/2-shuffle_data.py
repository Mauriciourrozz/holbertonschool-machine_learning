#!/usr/bin/env python3
"""
2-shuffle_data.py
"""
import numpy as np


def shuffle_data(X, Y):
    """
    Randomly shuffles two datasets independently.

    Parameters:
    X (numpy.ndarray): Input data of shape (m, ...)
    Y (numpy.ndarray): Labels or corresponding outputs of shape (m, ...)

    Returns:
    tuple: Shuffled versions of X and Y (not necessarily aligned)
    """
    m = X.shape[0]
    permutation = np.random.permutation(m)
    return X[permutation], Y[permutation]
