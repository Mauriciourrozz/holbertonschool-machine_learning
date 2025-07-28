#!/usr/bin/env python3
"""
1-pca.py
"""
import numpy as np


def pca(X, ndim):
    """
    Performs PCA on a dataset X and reduces its dimensionality to ndim.

    Returns the transformed dataset with ndim principal components.
    """
    mean_X = np.mean(X, axis=0)
    X_centered = X - mean_X

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    projected = X_centered @ Vt.T

    reduced = projected[:, :ndim]

    return reduced
