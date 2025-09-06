#!/usr/bin/env python3
"""
8-bi_rnn.py
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN.

    Parameters:
    bi_cell -- BidirectionalCell instance for propagation
    X -- Input data, numpy.ndarray of form (t, m, i)
          t: maximum number of time steps
          m: batch size
          i: dimensionality of data
    h_0 -- Initial hidden state in forward direction, numpy.ndarray of
        form (m, h)
    h_t -- Initial hidden state in backward direction, numpy.ndarray of
        form (m, h)
            h: dimensionality of the hidden state

    Returns:
    H -- All hidden states concatenated, numpy.ndarray
    AND -- All outputs, numpy.ndarray
    """

    # Obtener dimensiones
    t, m, i = X.shape
    h = h_0.shape[1]

    # Inicializar arrays para almacenar estados y salidas
    H_forward = np.zeros((t, m, h))
    H_backward = np.zeros((t, m, h))
    H = np.zeros((t, m, 2 * h))  # Concatenado
    Y = np.zeros((t, m, bi_cell.output_size))

    # Propagación forward (de 0 a t-1)
    h_prev_forward = h_0
    for step in range(t):
        h_prev_forward = bi_cell.forward(h_prev_forward, X[step])
        H_forward[step] = h_prev_forward

    # Propagación backward (de t-1 a 0)
    h_prev_backward = h_t
    for step in range(t-1, -1, -1):
        h_prev_backward = bi_cell.backward(h_prev_backward, X[step])
        H_backward[step] = h_prev_backward

    # Concatenar estados forward y backward
    for step in range(t):
        H[step] = np.concatenate((H_forward[step], H_backward[step]), axis=1)
        Y[step] = bi_cell.output(H[step])

    return H, Y
