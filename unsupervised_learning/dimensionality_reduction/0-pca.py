#!/usr/bin/env python3
"""
0-pca.py
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs Principal Component Analysis (PCA) on a dataset.

    Parameters:
    X (numpy.ndarray): A matrix of shape (n, d) where
                       n is the number of data points,
                       d is the number of dimensions.
                       The data must be centered (mean of 0).
    var (float): The fraction of the variance that PCA should maintain
    (default is 0.95).

    Returns:
    numpy.ndarray: The weight matrix W of shape (d, nd) that maintains
                   the desired fraction of the variance.
                   nd is the number of dimensions selected.
    """
    # Matriz de covarianza
    cov = np.cov(X, rowvar=False)

    #  descomposición en valores propios (eigendecomposition)
    eig_vals, eig_vecs = np.linalg.eigh(cov)

    idx = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    var_acum = np.cumsum(eig_vals) / np.sum(eig_vals)
    nd = np.searchsorted(var_acum, var) + 1

    W = eig_vecs[:, :nd]

    return W
