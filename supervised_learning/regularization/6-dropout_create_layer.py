#!/usr/bin/env python3
"""
6-dropout_create_layer.py
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Create a dense layer with conditional dropout (only during training).

    Args:
    prev: Tensor, output of the previous layer.
    n: int, number of nodes in the new layer.
    activation: Activation function (e.g., tf.nn.relu, tf.nn.tanh).
    keep_prob: float, probability of keeping a node active.
    training: bool, indicates whether the layer is in training mode
    (applies dropout only in that case).

    Returns:
    Tensor with the layer's output after activation and dropout.
    """
    dense_layer = tf.keras.layers.Dense(units=n, activation=activation)(prev)

    if training:
        dropout_layer = tf.keras.layers.Dropout(rate=1 - keep_prob)
        return dropout_layer(dense_layer, training=True)
    else:
        return dense_layer
