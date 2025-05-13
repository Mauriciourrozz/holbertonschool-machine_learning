#!/usr/bin/env python3
"""
3-speciity.py
"""
import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class from a confusion matrix.

    Arguments:
    confusion (numpy.ndarray): A square confusion matrix of shape
    (n_classes, n_classes), where confusion[i, j] represents the
    number of samples from true class i predicted as class j.

    Returns:
    numpy.ndarray: An array of shape (n_classes,) containing the specificity
    for each class.
    """
    especifidad = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        # Verdadero positivo
        vp = confusion[i, i]
        # Falso positivo
        fp = np.sum(confusion[:, i]) - vp
        # Falso negativo
        fn = np.sum(confusion[i, :]) - vp
        # Verdadero negativo
        vn = np.sum(confusion) - (vp + fp + fn)
        esp = vn / (vn + fp)
        especifidad[i] = esp

    return especifidad
