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
    l2 = tf.keras.regularizers.L2(lambtha)
    layer = tf.keras.layers.Dense(units=n, activation=activation,
                                  kernel_regularizer=l2)(prev)
    return layer
