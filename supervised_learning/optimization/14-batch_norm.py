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
    activation (function): Activation function to apply after batch
    normalization.

    Returns:
    tf.Tensor: Activated output of the batch-normalized layer.
    """
    layer = tf.keras.layers.Dense(
            units=n,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                mode='fan_avg'),
            name="layer")(prev)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]))
    beta = tf.Variable(tf.constant(0.0, shape=[n]))
    mean, variance = tf.nn.moments(layer, axes=[0])
    bn = tf.nn.batch_normalization(layer,
                                   mean=mean,
                                   variance=variance,
                                   offset=beta,
                                   scale=gamma,
                                   variance_epsilon=1e-7)
    return activation(bn)
