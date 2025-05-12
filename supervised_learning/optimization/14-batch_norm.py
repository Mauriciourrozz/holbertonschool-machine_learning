#!/usr/bin/env python3
"""
14-batch_norm.py
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    Parameters:
    prev (tf.Tensor): Activated output of the previous layer.
    n (int): Number of nodes in the new layer.
    activation (function): Activation function to apply after batch normalization.

    Returns:
    tf.Tensor: Activated output of the batch-normalized layer.
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    
    dense = tf.keras.layers.Dense(units=n, kernel_initializer=init)(prev)
    
    batch_norm = tf.keras.layers.BatchNormalization(
        axis=-1,
        momentum=0.99,
        epsilon=1e-7,
        center=True,
        scale=True
    )(dense)
    return activation(batch_norm)
