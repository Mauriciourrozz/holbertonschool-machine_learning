#!/usr/bin/env python3
"""
2-l2_reg_cost.py
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters
    ----------
    cost : tf.Tensor
        Tensor containing the cost of the network without L2 regularization.
    model : keras.Model
        Keras model which may include layers with L2 regularization.

    Returns
    -------
    List[tf.Tensor]
        A list of tensors where each element is the L2 regularization cost
        for a layer that has it, and the last element is the total cost
        including all L2 penalties added to the original cost.
    """
    l2_costos = []

    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.InputLayer):
            layer_l2 = tf.reduce_sum(layer.losses) + cost
            l2_costos.append(layer_l2)

    return tf.convert_to_tensor(l2_costos)
