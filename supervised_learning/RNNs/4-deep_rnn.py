#!/usr/bin/env python3
"""Deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, H_0):
    """
    Performs forward propagation for a deep RNN.

    Args:
        rnn_cells: List of RNNCell instances for each layer.
        X: numpy.ndarray with shape (t, m, i) containing the input data
            t: maximum number of time steps
            m: lot size
            i: dimensionality of data
        H_0: numpy.ndarray with shape (lc, m, h) containing the initial hidden
        state
            lc: number of layers
            h: dimensionality of the hidden state

    Returns:
        H: numpy.ndarray with all states hidden for each time step and layer
        And: numpy.ndarray with all results for each time step
    """
    t, m, i = X.shape
    lc = len(rnn_cells)
    h = H_0.shape[-1]

    H = np.zeros((t + 1, lc, m, h))
    Y = np.zeros((t, m, rnn_cells[-1].Wy.shape[1]))

    H[0] = H_0

    for step in range(t):
        for layer_idx in range(lc):
            if layer_idx == 0:
                input_data = X[step]
            else:
                input_data = H[step + 1, layer_idx - 1]

            prev_hidden_state = H[step, layer_idx]

            next_hidden_state, output = rnn_cells[layer_idx].forward(
                prev_hidden_state, input_data)

            H[step + 1, layer_idx] = next_hidden_state

            if layer_idx == lc - 1:
                Y[step] = output

    return H, Y
