#!/usr/bin/env python3
import numpy as np


def one_hot_encode(Y, classes):
    """
    One-hot encode a numeric label vector.
    """
    if (not isinstance(Y, np.ndarray)
            or Y.ndim != 1
            or not isinstance(classes, int)
            or classes <= 0
            or Y.min() < 0
            or Y.max() >= classes):
        return None

    m = Y.shape[0]
    matriz = np.zeros((classes, m))
    for i in range(m):
        y = Y[i]
        matriz[y, i] = 1
    return matriz
