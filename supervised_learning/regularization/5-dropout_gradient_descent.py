#!/usr/bin/env python3
"""
5-dropout_gradient_descent.py
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a dropout-regularized neural
    network using gradient descent.

    Args:
    Y: np.ndarray of shape (classes, m) with the true labels.
    weights: dict with the keys 'W{i}' and 'b{i}' for each layer i.
    cache: dict with the keys 'A{i}' (activations) and 'D{i}'
    (dropout masks) for each layer i.
    alpha: learning rate.
    keep_prob: probability of keeping a node active.
    L: total number of layers.
    """
    m = Y.shape[1]

    A_final = cache['A' + str(L)]

    dZ = A_final - Y

    for layer in range(L, 0, -1):
        A_prev = cache['A' + str(layer - 1)]

        dW = np.dot(dZ, A_prev.T) / m

        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights['W' + str(layer)] = weights['W' + str(layer)] - alpha * dW
        weights['b' + str(layer)] = weights['b' + str(layer)] - alpha * db

        if layer > 1:
            dA = np.dot(weights['W' + str(layer)].T, dZ)

            D = cache['D' + str(layer - 1)]
            dA = dA * D
            dA = dA / keep_prob

            dZ = dA * (1 - np.power(A_prev, 2))
