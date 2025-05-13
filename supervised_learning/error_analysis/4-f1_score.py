#!/usr/bin/env python3
"""
4-f1_score.py
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1-score for each class from a confusion matrix.

    The F1-score is the harmonic mean of precision and recall (sensitivity),
    providing a balance between them. It is useful when you need a metric
    that combines both false positives and false negatives.

    Arguments:
    confusion (numpy.ndarray): A confusion matrix of shape
    (n_classes, n_classes),
    where confusion[i, j] represents the number of samples
    from true class i predicted as class j.

    Returns:
    numpy.ndarray: An array of shape (n_classes,) containing the
    F1-score for each class.
    """
    f1score = np.zeros(confusion.shape[0])
    for i in range(confusion.shape[0]):
        precision_ = precision(confusion)
        sensibilidad = sensitivity(confusion)
        f1 = 2 * (
            precision_[i] * sensibilidad[i]) / (
                precision_[i] + sensibilidad[i])

        f1score[i] = f1

    return f1score
