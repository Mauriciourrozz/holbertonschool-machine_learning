#!/usr/bin/env python3
"""
policy_gradient.py
"""
import numpy as np


def policy(matrix, weight):
    """
    Compute a stochastic policy using a weighted matrix and softmax.

    Args:
        matrix (np.ndarray): Input matrix or feature vector.
        weight (np.ndarray or float): Weight parameters.

    Returns:
        np.ndarray: Probability distribution (policy).
    """
    # Multiplicación ponderada
    logits = np.matmul(matrix, weight)

    # Softmax para convertir en probabilidades
    exp_vals = np.exp(logits - np.max(logits))
    probs = exp_vals / np.sum(exp_vals)

    return probs
