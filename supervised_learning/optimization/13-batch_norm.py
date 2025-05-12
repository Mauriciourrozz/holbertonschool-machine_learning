#!/usr/bin/env python3
"""
13-batch_norm.py
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes the unactivated output of a neural network layer using batch
    normalization.

    Parameters:
    Z (np.ndarray): Array of shape (m, n) containing the data to normalize.
    gamma (np.ndarray): Scale parameters of shape (1, n).
    beta (np.ndarray): Offset parameters of shape (1, n).
    epsilon (float): Small constant to prevent division by zero.

    Returns:
    np.ndarray: The normalized and scaled output.
    """
    mean = np.mean(Z, axis=0, keepdims=True)
    variance = np.var(Z, axis=0, keepdims=True)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)
    Z_tilde = gamma * Z_norm + beta
    return Z_tilde
