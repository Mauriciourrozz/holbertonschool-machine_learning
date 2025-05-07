#!/usr/bin/env python3
"""
1-normalize.py
"""
import numpy as np


def normalize(X, m, s):
    """
    Normalizes the dataset X using the given mean and standard deviation.

    Parameters:
    X (numpy.ndarray): The data to normalize, of shape (m, nx)
    m (numpy.ndarray): The mean of each feature, of shape (nx,)
    s (numpy.ndarray): The standard deviation of each feature, of shape (nx,)

    Returns:
    numpy.ndarray: The normalized data
    """
    return (X - m) / s
