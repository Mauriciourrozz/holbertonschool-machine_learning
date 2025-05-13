#!/usr/bin/env python3
"""
2-precision.py
"""
import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix.

    Arguments:
    confusion(numpy.ndarray): Confusion matrix of shape (n_classes, n_classes),
    where confusion[i, j] represents the number of elements from class i
    predicted as class j.

    Returns:
    numpy.ndarray: An array of shape (n_classes,) containing the precision
    for each class.
    """
    precisiones = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        # asi hayo el verdadero positivo
        vp = confusion[i, i]
        # y asi el falso positivo
        fp = np.sum(confusion[:, i]) - confusion[i, i]
        precision = vp / (vp + fp)
        precisiones[i] = precision
    return precisiones
