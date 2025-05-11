#!/usr/bin/env python3
"""
9-Adam.py
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable using the Adam optimization algorithm.

    Parameters:
    alpha (float): Learning rate.
    beta1 (float): Weight for the first moment (momentum).
    beta2 (float): Weight for the second moment (RMSProp).
    epsilon (float): Small value to prevent division by zero.
    var (np.ndarray): Variable to be updated.
    grad (np.ndarray): Gradient of the variable.
    v (np.ndarray): Previous first moment.
    s (np.ndarray): Previous second moment.
    t (int): Time step for bias correction.

    Returns:
    tuple: Updated variable, updated first moment, updated second moment.
    """
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * np.square(grad)

    v_corr = v / (1 - beta1 ** t)
    s_corr = s / (1 - beta2 ** t)

    var = var - alpha * v_corr / (np.sqrt(s_corr) + epsilon)

    return var, v, s