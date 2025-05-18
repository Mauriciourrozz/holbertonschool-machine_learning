#!/usr/bin/env python3
"""
3-l2_reg_create_layer.py
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Create a neural network layer with L2 regularization.

    prev: previous layer (input)
    n: number of neurons
    activation: activation function
    lambtha: L2 regularization lambda
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_avg')
    regularizer = tf.keras.regularizers.L2(lambtha)
    layer = tf.keras.layers.Dense(units=n,
                                  activation=activation,
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer)(prev)
    return layer
