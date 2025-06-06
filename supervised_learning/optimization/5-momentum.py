#!/usr/bin/env python3
"""
5-momentum.py
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum optimization
    algorithm.

    Parameters:
    alpha (float): The learning rate.
    beta1 (float): The momentum weight (between 0 and 1).
    var (numpy.ndarray): The variable to be updated.
    grad (numpy.ndarray): The gradient of the cost with respect to var.
    v (numpy.ndarray): The previous first moment of var (velocity).

    Returns:
    tuple: The new moment and the updated variable, respectively.
    """
    new_moment = beta1 * v + (1 - beta1) * grad
    var = var - alpha * new_moment
    return var, new_moment
