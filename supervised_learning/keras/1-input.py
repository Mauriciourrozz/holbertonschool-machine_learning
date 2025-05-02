#!/usr/bin/env python3
"""
1-input.py
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network model using Keras Functional API.

    Args:
    nx (int): The number of input features.
    layers (list): A list containing the number of nodes in each layer of the network.
    activations (list): A list containing the activation functions to be used for each layer.
    lambtha (float): The L2 regularization parameter.
    keep_prob (float): The probability of keeping a node during dropout.

    Returns:
    keras.Model: A compiled Keras model.
    """

    # se cargan los datos de entrada al modelo y se guardan en inputs
    inputs = K.layers.Input(shape=(nx,))

    # se crea una capa densa y al poner (inputs) se conecta con la entrada
    x = K.layers.Dense(
        units=layers[0],
        activation=activations[0],
        kernel_regularizer=K.regularizers.l2(lambtha))(inputs)

    # se recorren las capaz intermedias, la primera ya fue creada antes
    # y la ultima se hace despues
    for i in range(1, len(layers)):
        # dropout apaga neuronas al azar antes de seguir para que
        # la red no se apoye demasiado en ciertas neuronas
        x = K.layers.Dropout(1 - keep_prob)(x)

        # se crea una nueva capa en cada iteracion y se conecta con la anterior
        # que fue x
        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha))(x)

    # ultima capa, procesa los ultimos valores y se conecta a la
    # ultima capa intermedia
    output = x
    # se esta  creando el modelo completo, conectando todas las capas
    model = K.Model(inputs=inputs, outputs=output)

    return model
