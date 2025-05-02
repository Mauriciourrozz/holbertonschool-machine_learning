#!/usr/bin/env python3
"""
2-optimize.py
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Optimizes the Keras model using the Adam optimizer with categorical
    crossentropy loss.

    Args:
    network (keras.Model): The Keras model to be optimized.
    alpha (float): The learning rate for the Adam optimizer.
    beta_1 (float): The exponential decay rate for the first
    moment estimates in the Adam optimizer.
    beta_2 (float): The exponential decay rate for the second
    moment estimates in the Adam optimizer.

    Returns:
    None: The function does not return any value, it modifies the
    `network` in place.
    """
    optimizer = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2)

    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    return None
