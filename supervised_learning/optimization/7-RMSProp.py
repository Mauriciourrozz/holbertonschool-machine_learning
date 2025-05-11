#!/usr/bin/env python3
"""
7-RMSProp.py
"""
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Parameters:
    alpha (float): Learning rate.
    beta2 (float): RMSProp decay factor (between 0 and 1).
    epsilon (float): Small value to avoid division by zero.
    var (numpy.ndarray): Variable to be updated.
    grad (numpy.ndarray): Gradient of the variable.
    s (numpy.ndarray): Previous second moment of the variable.

    Returns:
    tuple: The updated variable and the new second moment.
    """

    s = beta2 * s + (1 - beta2) * np.square(grad)
    var = var - alpha * grad / (np.sqrt(s) + epsilon)
    return var, s
