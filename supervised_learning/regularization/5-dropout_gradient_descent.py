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
    dz = cache[f"A{L}"] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache[f"A{i - 1}"] if i > 1 else cache["A0"]
        W = weights[f"W{i}"]
        b = weights[f"b{i}"]

        dW = (1 / m) * np.dot(dz, A_prev.T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        weights[f"W{i}"] -= alpha * dW
        weights[f"b{i}"] -= alpha * db

        if i > 1:
            da = np.dot(W.T, dz)
            da *= cache[f"D{i - 1}"]
            da /= keep_prob

            A_prev = cache[f"A{i - 1}"]
            dz = da * (1 - A_prev ** 2)
