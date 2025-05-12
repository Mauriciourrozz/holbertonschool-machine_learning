#!/usr/bin/env python3
"""
1-sensitivity.py
"""
import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity (recall) for each class in a confusion matrix.

    Args:
    confusion (numpy.ndarray): A 2D square confusion matrix of shape
    (classes, classes), where rows represent true labels and columns
    predicted labels.

    Returns:
    numpy.ndarray: 1D array of sensitivities for each class.
    """
    num_classes = confusion.shape[0]
    sensitivities = np.zeros(num_classes)
    for i in range(num_classes):
        true_positives = confusion[i, i]
        total_actual = np.sum(confusion[i, :])

        if total_actual == 0:
            sensitivities[i] = 0
        else:
            sensitivities[i] = true_positives / total_actual

    return sensitivities
