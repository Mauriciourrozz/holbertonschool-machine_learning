#!/usr/bin/env python3
"""
8-RMSProp.py
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Creates an RMSProp optimizer in TensorFlow.

    Parameters:
    alpha (float): Learning rate.
    beta2 (float): Decay rate (rho) for the moving average of
    squared gradients.
    epsilon (float): Small constant to prevent division by zero.

    Returns:
    tf.keras.optimizers.Optimizer: A TensorFlow RMSprop optimizer.
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        rho=beta2,
        epsilon=epsilon)
