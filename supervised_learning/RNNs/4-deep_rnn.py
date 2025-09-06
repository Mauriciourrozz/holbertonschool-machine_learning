#!/usr/bin/env python3
"""
4-deep_rnn.py
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN.

    Args:
    rnn_cells (list): A list of RNNCell instances to be used for
        forward propagation. Length of the list is the number of layers.
    X (numpy.ndarray): The input data of shape (t, m, i), where:
                       - t is the number of time steps,
                       - m is the batch size,
                       - i is the input dimensionality.
    h_0 (numpy.ndarray): The initial hidden state of shape (L, m, h), where:
                         - L is the number of layers,
                         - m is the batch size,
                         - h is the dimensionality of the hidden state.

    Returns:
    tuple: A tuple containing:
        - H (numpy.ndarray): Hidden states for all layers at all time steps
        - Y (numpy.ndarray): The outputs of the final layer at all time steps
    """
    # Obtener las dimensiones de la entrada
    t, m, i = X.shape
    # Número de capas
    L = len(rnn_cells)
    # Dimensionalidad del estado oculto
    h = h_0.shape[-1]

    # Inicializar matrices para los estados ocultos (H) y las salidas (Y)
    H = np.zeros((t, m, L, h))
    Y = np.zeros((t, m, rnn_cells[-1].Wy.shape[-1]))

    # Establecer el estado oculto inicial
    h_prev = h_0

    # Iterar sobre los pasos de tiempo
    for i in range(t):
        # Iterar sobre las capas de la red
        for j in range(L):
            # Si estamos en el primer paso de tiempo, usar la entrada
            if i == 0:
                x = X[i]
            else:
                # Para pasos posteriores, usar la salida de la capa anterior
                x = h_prev[j-1]

            # Calcular el nuevo estado oculto para la capa j
            h_prev[j], _, _ = rnn_cells[j].forward(h_prev[j], x)

            # Guardar el estado oculto de la capa j en H
            H[i, :, j, :] = h_prev[j]

        # Almacenar la salida de la última capa (capa de salida) en Y
        Y[i] = h_prev[-1]

    # Retornar los estados ocultos y las salidas
    return H, Y
