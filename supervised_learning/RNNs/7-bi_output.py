#!/usr/bin/env python3
"""
7-bi_output.py
"""
import numpy as np


class BidirectionalCell:
    """
    Implements a Bidirectional RNN Cell that performs both forward and
    backward passes
    """
    def __init__(self, i, h, o):
        """
        Initializes a Bidirectional RNN cell.

        Args:
            i: Dimensionality of the input data.
            h: Dimensionality of the hidden states.
            o: Dimensionality of the output data.
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)

        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Performs the forward pass for a single time step.

        Args:
            h_prev: numpy.ndarray of shape (m, h) containing the previous
            hidden state.
            x_t: numpy.ndarray of shape (m, i) containing the input data
            at the current time step.

        Returns:
            h_next: numpy.ndarray of shape (m, h) containing the next
            hidden state in the forward direction.
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the previous hidden state by performing a
        propagation step.

        Args:
            h_next: numpy.ndarray of shape (m, h) containing the
            next hidden state
                m: batch size
                h: dimensionality of the hidden state
            x_t: numpy.ndarray of shape (m, i) containing the data
            input for the cell
                i: dimensionality of the data

        Returns:
            h_prev: previous hidden state (numpy.ndarray of shape (m, h))
        """
        concat = np.concatenate((h_next, x_t), axis=1)

        h_prev = np.tanh(np.dot(concat, self.Whb) + self.bhb)

        return h_prev

    def output(self, H):
        """
        Calculates the output of the RNN

        Args:
            H: numpy.ndarray of shape (t, m, 2h) containing the
            concatenated hidden states from both directions,
            excluding their initialized states
                t: maximum number of time steps
                m: batch size
                h: dimensionality of the hidden state
        """
        t, m, _ = H.shape
        o = self.by.shape[1]

        Y = np.empty((t, m, o))

        for time_step in range(t):
            linear_output = np.dot(H[time_step], self.Wy) + self.by

            y_exp = np.exp(linear_output - np.max(
                linear_output, axis=1, keepdims=True))
            Y[time_step] = y_exp / np.sum(y_exp, axis=1, keepdims=True)

        return Y
