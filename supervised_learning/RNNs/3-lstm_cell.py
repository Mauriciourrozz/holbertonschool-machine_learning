#!/usr/bin/env python3
"""
3-lstm_cell.py
"""
import numpy as np


class LSTMCell:
    """
    LSTMCell class that implements a single Long Short-Term Memory
        (LSTM) unit.

    Attributes:
    Wf (numpy.ndarray): Weight matrix for the forget gate.
    Wu (numpy.ndarray): Weight matrix for the update gate (input gate).
    Wc (numpy.ndarray): Weight matrix for the candidate cell state.
    Wo (numpy.ndarray): Weight matrix for the output gate.
    Wy (numpy.ndarray): Weight matrix for the output layer.
    bf (numpy.ndarray): Bias for the forget gate.
    bu (numpy.ndarray): Bias for the update gate (input gate).
    bc (numpy.ndarray): Bias for the candidate cell state.
    bo (numpy.ndarray): Bias for the output gate.
    by (numpy.ndarray): Bias for the output layer.
    """
    def __init__(self, i, h, o):
        """
        Initializes the LSTMCell with random weights and zero biases.

        Args:
        i (int): Dimensionality of the input data.
        h (int): Dimensionality of the hidden state.
        o (int): Dimensionality of the output.

        Initializes the weight matrices (Wf, Wu, Wc, Wo, Wy) and the bias
            vectors (bf, bu, bc, bo, by)
        using random normal distribution and zero initialization for the
            biases.
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))

        self.bf = np.zeros((h,))
        self.bu = np.zeros((h,))
        self.bc = np.zeros((h,))
        self.bo = np.zeros((h,))
        self.by = np.zeros((o,))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step in the LSTM.

        Args:
        h_prev (numpy.ndarray): The previous hidden state (shape: (m, h)).
        c_prev (numpy.ndarray): The previous cell state (shape: (m, h)).
        x_t (numpy.ndarray): The input data at time step t (shape: (m, i)).

        Returns:
        tuple: A tuple containing:
            - h_next (numpy.ndarray): The next hidden state (shape: (m, h)).
            - c_next (numpy.ndarray): The next cell state (shape: (m, h)).
            - y (numpy.ndarray): The output of the LSTM cell at time step t
                (shape: (m, o)).

        The forward propagation computes the forget gate, update gate,
            candidate cell state,
        output gate, next cell state, next hidden state, and final output
            using the input and
        previous states.
        """
        # Concatenar el estado oculto previo con la entrada
        combined = np.concatenate((h_prev, x_t), axis=1)

        # Calcular las puertas
        f_t = self.sigmoid(np.dot(combined, self.Wf) + self.bf)
        u_t = self.sigmoid(np.dot(combined, self.Wu) + self.bu)
        c_t = np.tanh(np.dot(combined, self.Wc) + self.bc)
        o_t = self.sigmoid(np.dot(combined, self.Wo) + self.bo)

        # Actualizar el estado de la celda
        c_next = f_t * c_prev + u_t * c_t

        # Calcular el siguiente estado oculto
        h_next = o_t * np.tanh(c_next)

        # Calcular la salida de la celda
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, c_next, y

    def sigmoid(self, x):
        """
        Sigmoid activation function.

        Args:
        x (numpy.ndarray): The input array.

        Returns:
        numpy.ndarray: The output after applying the sigmoid function
            element-wise.
        """
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """
        Softmax activation function.

        Args:
        x (numpy.ndarray): The input array.

        Returns:
        numpy.ndarray: The output after applying the softmax function
            row-wise (over the last axis).
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
