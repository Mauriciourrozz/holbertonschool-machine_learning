#!/usr/bin/env python3
"""
This module contains a function one_hot_decode
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot encoded matrix into a vector of labels.

    Args:
        one_hot (numpy.ndarray): A one-hot encoded matrix of shape
        (classes, m),
        where 'classes' is the number of classes and
        'm' is the number of examples. Each column represents
        an example, and each row represents a class.

    Returns:
        numpy.ndarray: A vector of labels of shape (m,) where each value is the
        index of the class with a '1' in the corresponding column.
        If there is any issue with the input, None is returned.
    """
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None
    try:
        return np.argmax(one_hot, axis=0)
    except Exception:
        return None
