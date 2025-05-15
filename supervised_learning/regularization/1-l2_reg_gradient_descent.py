#!/usr/bin/env python3
"""
1-l2_reg_gradient_descent.py
"""


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

    dZ = cache['A' + str(L)] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        dW = (1 / m) * (dZ @ A_prev.T) + (lambtha / m) * W

        db = (1 / m) * dZ.sum(axis=1, keepdims=True)

        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] = b - alpha * db

        if i > 1:
            A_prev = cache['A' + str(i - 1)]
            W = weights['W' + str(i)]
            dZ = (W.T @ dZ) * (1 - A_prev ** 2)
