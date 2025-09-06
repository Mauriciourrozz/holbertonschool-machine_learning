#!/usr/bin/env python3
"""
0-rnn_cell.py
"""
import numpy as np


class RNNCell:
    """
    A class representing a simple Recurrent Neural Network (RNN) cell.

    This cell performs forward propagation for a single time step of an RNN,
    including calculating the hidden state and the output using weights and
    biases.

    Attributes:
    Wh -- Weight matrix for the hidden state to hidden state and input
        concatenation.
    Wy -- Weight matrix for the hidden state to output.
    bh -- Bias vector for the hidden state.
    by -- Bias vector for the output.
    """
    def __init__(self, i, h, o):
        """
        Initializes the RNNCell with input, hidden, and output dimensions,
        and initializes weights and biases using random normal distribution
        and zeros, respectively.

        Parameters:
        i -- int, the dimensionality of the input data (input size).
        h -- int, the dimensionality of the hidden state.
        o -- int, the dimensionality of the output (output size).
        """
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """
        Applies the softmax function to the input array to convert it into
            probabilities.

        Softmax ensures the outputs are probabilities that sum to 1.

        Parameters:
        x -- numpy.ndarray, the input array to apply softmax to.

        Returns:
        numpy.ndarray, the softmax probabilities.
        """
        # Función softmax para convertir las salidas en probabilidades
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step of the RNN.

        Parameters:
        h_prev -- numpy.ndarray of shape (m, h), the previous hidden state.
        x_t -- numpy.ndarray of shape (m, i), the input data at the current
        time step.
        m is the batch size.

        Returns:
        h_next -- numpy.ndarray of shape (m, h), the next hidden state after
            the propagation.
        y -- numpy.ndarray of shape (m, o), the output of the RNN cell after
            applying softmax.
        """
        # Concatenamos h_prev y x_t a lo largo de las columnas
        concatenation = np.concatenate([h_prev, x_t], axis=1)

        # Cálculo del nuevo estado oculto
        h_next = np.tanh(np.matmul(concatenation, self.Wh) + self.bh)

        # Cálculo de la salida y, aplicando softmax
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
