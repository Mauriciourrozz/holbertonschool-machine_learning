#!/usr/bin/env python3
"""
3-one_hot.py
"""


import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a list of integer labels into a one-hot encoded matrix.

    Args:
        labels (array-like): A list or array of integer labels to be converted.
        classes (int, optional): The total number of classes
        (columns in the one-hot matrix).
        If None, the number of classes will be inferred from the labels.

    Returns:
        np.ndarray: A one-hot encoded matrix where each row corresponds
        to a label and each column corresponds to a class.
        The position of the 1  in each row corresponds to the
        label's class.
    """
    return K.utils.to_categorical(labels, num_classes=classes)
