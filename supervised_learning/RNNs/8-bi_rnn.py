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
    
    # Inicializar arrays para almacenar estados
    H_forward = np.zeros((t, m, h))
    H_backward = np.zeros((t, m, h))
    H = np.zeros((t, m, 2 * h))  # Concatenado
    
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
    
    # Calcular las salidas - necesitamos agregar una dimensión temporal
    Y = np.zeros((t, m, bi_cell.by.shape[1]))  # Usamos by para obtener la dimensión de salida
    
    for step in range(t):
        # Agregar dimensión temporal para que sea (1, m, 2*h)
        h_step = H[step][np.newaxis, :, :]
        output_step = bi_cell.output(h_step)
        Y[step] = output_step[0]  # Remover la dimensión temporal
    
    return H, Y
