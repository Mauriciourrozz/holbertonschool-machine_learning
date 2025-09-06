#!/usr/bin/env python3
"""
2-gru_cell.py
"""
import numpy as np


class GRUCell:
    def __init__(self, i, h, o):
        """
        Constructor of the GRUCell class.

        i: Dimensionality of the input data.
        h: Dimensionality of the hidden state.
        o: Dimensionality of the output.
        """
        # Inicialización de los pesos de las puertas
        self.Wz = np.random.randn(i, h)
        self.Wr = np.random.randn(i, h)
        self.Wh = np.random.randn(i, h)
        self.Wy = np.random.randn(h, o)

        # Inicialización de los sesgos
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        GRUCell forward propagation.

        h_prev: Previous hidden state, of size (m, h), where m is the size of
            the batch.
        x_t: Current input, of size (m, i).

        Returns:
        h_next: The next hidden state of the cell.
        y: The output of the cell.
        """
        # Calcular la puerta de actualización
        z_t = sigmoid(np.dot(x_t, self.Wz) + np.dot(
            h_prev, self.Wz.T) + self.bz)

        # Calcular la puerta de reinicio
        r_t = sigmoid(np.dot(x_t, self.Wr) + np.dot(
            h_prev, self.Wr.T) + self.br)

        # Calcular el estado oculto intermedio
        h_tilde = np.tanh(np.dot(x_t, self.Wh) + np.dot(
            r_t * h_prev, self.Wh.T) + self.bh)

        # Calcular el siguiente estado oculto
        h_next = (1 - z_t) * h_prev + z_t * h_tilde

        # Calcular la salida
        y = softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y


def sigmoid(x):
    """
    sigmoid function
    """
    return 1 / (1 + np.exp(-x))


def softmax(x):
    """
    Softmax function
    """
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)
