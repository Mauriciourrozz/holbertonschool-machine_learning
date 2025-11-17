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


def policy_gradient(state, weight):
    """
    Computes Monte-Carlo policy gradient using state and weight.
    Returns selected action and gradient.
    """

    # Ensure state is 2D: (1, n_features)
    state = np.atleast_2d(state)

    # Get policy probs and ensure (1, n_actions)
    probs = policy(state, weight)
    probs = np.atleast_2d(probs)

    # Sample action
    action = np.random.choice(probs.shape[1], p=probs.flatten())

    # Create one-hot action vector
    one_hot = np.zeros_like(probs)
    one_hot[0, action] = 1

    # Compute gradient: (n_features,1) × (1,n_actions) = (n_features,n_actions)
    gradient = np.matmul(state.T, (one_hot - probs))

    return action, gradient
