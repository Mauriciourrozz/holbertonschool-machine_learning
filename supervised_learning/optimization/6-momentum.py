#!/usr/bin/env python3
"""
6-momentum.py
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Creates a Momentum optimization operation using TensorFlow.

    Parameters:
    alpha (float): The learning rate.
    beta1 (float): The momentum weight (usually between 0 and 1).

    Returns:
    tf.keras.optimizers.Optimizer: A TensorFlow SGD optimizer with momentum.
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
