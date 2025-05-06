#!/usr/bin/env python3
"""
0-norm_constants.py
"""
import numpy as np


def normalization_constants(X):
    """
    Calculate the normalization constants (mean and standard deviation)
    of each characteristic in the matrix

    Parameters:
    X (numpy.ndarray): Shape data array (m, nx)

    Returns:
    tuple: mean and standard deviation of each feature (nx,)
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    return mean, std
    