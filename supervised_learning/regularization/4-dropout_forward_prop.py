#!/usr/bin/env python3
"""
4-dropout_forward_prop.py
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Performs forward propagation in a neural network with dropout.

    Args:
        X (numpy.ndarray): Input data of shape (nx, m), where
            nx is the number of features and
            m is the number of examples.
        weights (dict): Dictionary containing the weights and biases of
        the network, with keys 'W1', 'b1', ..., 'WL', 'bL'.
        L (int): Number of layers in the neural network.
        keep_prob (float): Probability of keeping a node active
        (between 0 and 1).

    Returns:
        dict: Dictionary with the outputs of each layer and the dropout masks,
            where:
            - 'A0' is the input X,
            - 'A{i}' is the activation of layer i,
            - 'D{i}' is the dropout mask used in layer i
              (only for hidden layers, not for the last layer).
    """
    cache = {'A0': X}
    for i in range(1, L + 1):
        Wl = weights['W' + str(i)]
        bl = weights['b' + str(i)]
        Al_prev = cache['A' + str(i - 1)]
        Zl = np.matmul(Wl, Al_prev) + bl
        if i != L:
            A = np.tanh(Zl)
            D = (np.random.rand(A.shape[0],
                                A.shape[1]) < keep_prob).astype(int)

            A = A * D
            A = A / keep_prob
            cache['D' + str(i)] = D
        else:
            A = np.exp(Zl) / np.sum(np.exp(Zl), axis=0, keepdims=True)
        cache['A' + str(i)] = A
    return cache
