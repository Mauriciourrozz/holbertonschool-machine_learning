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
    A_output = cache['A' + str(L)]
    dZ = A_output - Y
    grads = {}

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i-1)]
        grads['dW' + str(i)] = np.dot(dZ, A_prev.T) / m
        grads['db' + str(i)] = np.sum(dZ, axis=1, keepdims=True) / m

        if i > 1:
            dA = np.matmul(weights['W' + str(i)].T, dZ)
            dA *= cache['D' + str(i-1)]
            dA /= keep_prob
            dZ = dA * (1 - np.power(A_prev, 2))

        weights['W' + str(i)] -= alpha * grads['dW' + str(i)]
        weights['b' + str(i)] -= alpha * grads['db' + str(i)]
