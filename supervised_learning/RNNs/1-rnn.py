#!/usr/bin/env python3
"""
1-rnn.py
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Perform forward propagation for a simple RNN.

    Parameters:
    rnn_cell -- an instance of RNNCell used for forward propagation
    X -- numpy.ndarray of shape (t, m, i), input data
    h_0 -- numpy.ndarray of shape (m, h), the initial hidden state

    Returns:
    H -- numpy.ndarray of shape (t, m, h), all hidden states
    Y -- numpy.ndarray of shape (t, m, o), all outputs
    """
    t, m, i = X.shape
    h = np.zeros((t, m, rnn_cell.Wh.shape[1]))
    y = np.zeros((t, m, rnn_cell.Wy.shape[1]))
    h_prev = h_0

    for time_step in range(t):
        h[time_step], y[time_step] = rnn_cell.forward(h_prev, X[time_step])
        h_prev = h[time_step]

    return h, y
