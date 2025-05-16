#!/usr/bin/env python3
"""
1-l2_reg_gradient_descent.py
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using gradient descent
    with L2 regularization.

    Parameters:
    -----------
    Y : numpy.ndarray
        Array of shape (classes, m) containing the true labels
        (one-hot encoded).
    weights : dict
        Dictionary containing the weights and biases of each layer
        (W1, b1, ..., WL, bL).
    cache : dict
        Dictionary containing the activations of each layer (A0, A1, ..., AL).
    alpha : float
        Learning rate.
    lambtha : float
        L2 regularization parameter.
    L : int
        Total number of layers in the neural network.

    Returns:
    --------
    None
        This function updates the weights dictionary **in-place**.
    """
    m = Y.shape[1]
    dZ = 0

    for l in reversed(range(1, L + 1)):
        A_curr = cache['A' + str(l)]
        A_prev = cache['A' + str(l - 1)]
        W_curr = weights['W' + str(l)]
        b_curr = weights['b' + str(l)]

        if l == L:
            dZ = A_curr - Y
            dW = (1 / m) * np.dot(dZ, A_prev.T)
        else:
            dZ = np.dot(weights['W' + str(l + 1)].T, dZ) * (1 - A_curr ** 2)
            dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambtha / m) * W_curr

        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights['W' + str(l)] -= alpha * dW
        weights['b' + str(l)] -= alpha * db
