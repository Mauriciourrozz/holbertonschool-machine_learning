#!/usr/bin/env python3
"""
5-bi_forward.py
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
