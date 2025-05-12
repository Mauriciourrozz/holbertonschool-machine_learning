#!/usr/bin/env python3
"""
0-create_confusion.py
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix from true labels and predicted logits.

    Args:
    labels (numpy.ndarray): A 2D array of shape (m, classes) containing the
    true labels in one-hot encoded format, where m is the number of samples
    and classes is the number of classes.
    logits (numpy.ndarray): A 2D array of shape (m, classes) containing the
    predicted logits or probabilities, where each row corresponds
    to the predicted class distribution for each sample.

    Returns:
    numpy.ndarray: A 2D confusion matrix of shape (classes, classes), where
    rows represent the true labels and columns represent the predicted labels.
    Each entry (i, j) in the matrix indicates the number of samples
    where the true label was class i and the predicted label was class j.
    """
    num_class = labels.shape[1]
    matrix = np.zeros((num_class, num_class))
    for i in range(labels.shape[0]):
        true_label = np.argmax(labels[i])
        predicted_label = np.argmax(logits[i])
        matrix[true_label, predicted_label] += 1

    return matrix
