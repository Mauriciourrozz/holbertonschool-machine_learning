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
    # Inicialización de h
    h = np.zeros((t + 1, m, rnn_cell.Wh.shape[1]))
    # Inicialización de y
    y = np.zeros((t, m, rnn_cell.Wy.shape[1]))
    h_prev = h_0

    for step in range(t):
        h[step] = h_prev
        h_prev, y_step = rnn_cell.forward(h_prev, X[step])
        y[step] = y_step
        
    # Almacenamos el último estado oculto
    h[t] = h_prev

    return h, y