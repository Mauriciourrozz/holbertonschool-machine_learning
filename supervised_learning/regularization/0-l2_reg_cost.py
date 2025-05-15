#!/usr/bin/env python3
"""
0-l2_reg_cost.py
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost using L2 regularization.

    Parameters:
    - cost: float, original cost (e.g., cross-entropy).
    - lambtha: float, regularization coefficient.
    - weights: dict, dictionary with the weights for each layer (W1, b1, W2, b2, ...).
    - L: int, number of layers in the network.
    - m: int, number of examples.

    Returns:
    - float, the total cost using L2 regularization.
    """
    l2_sum = 0

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        l2_sum += np.sum(W ** 2)

    l2_term = (lambtha / (2 * m)) * l2_sum

    return cost + l2_term
