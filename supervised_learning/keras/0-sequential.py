#!/usr/bin/env python3
"""
0-sequential.py
"""
import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network model using Keras Sequential API.

    Parameters:
    nx (int): The number of input features for the model.
    layers (list): A list containing the number of
    nodes in each layer.
    activations (list): A list containing the activation
    functions for each layer.
    lambtha (float): The L2 regularization parameter
    to reduce overfitting.
    keep_prob (float): The probability of keeping a node during
    dropout to prevent overfitting.

    Returns:
    model: A Keras Sequential model with the specified layers,
    activation functions, and regularization.
    """
    model = k.Sequential()

    # Primera capa
    model.add(
        k.layers.Dense(
            units=layers[0],
            activation=activations[0],
            input_shape=(nx,),
            kernel_regularizer=k.regularizers.l2(lambtha)
        )
    )
    if len(layers) > 1:
        model.add(k.layers.Dropout(1 - keep_prob))

    # se agrega el resto de las capas, de la 2da en adelante
    for i in range(1, len(layers)):
        model.add(
            k.layers.Dense(
                units=layers[i],
                activation=activations[i],
                kernel_regularizer=k.regularizers.l2(lambtha)
            )
        )

        if i != len(layers) - 1:
            model.add(k.layers.Dropout(1 - keep_prob))

    return model
