#!/usr/bin/env python3
"""
0-pca.py
"""
import numpy as np


def pca(X, var=0.95):
    """
    Performs PCA on a dataset X, keeping enough components to preserve
    the desired variance ratio (var).
    
    Returns the projection matrix W.
    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    total = np.sum(S)

    explained = 0
    num_components = 0

    for i in range(len(S)):
        explained += S[i]
        if explained / total >= var:
            num_components = i + 1
            break

    W = Vt[:num_components].T
    return W
