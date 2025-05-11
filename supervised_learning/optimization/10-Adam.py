#!/usr/bin/env python3
"""
10-Adam.py
"""
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Creates an Adam optimizer using the given hyperparameters.

    Parameters:
    alpha (float): The learning rate.
    beta1 (float): The exponential decay rate for the first moment
    estimates (momentum).
    beta2 (float): The exponential decay rate for the second moment
    estimates (RMSProp).
    epsilon (float): A small constant to prevent division by zero.

    Returns:
    tf.keras.optimizers.Adam: A TensorFlow Adam optimizer instance.
    """
    return tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon)
