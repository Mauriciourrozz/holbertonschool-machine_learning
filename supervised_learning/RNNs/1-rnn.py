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
    # Número de pasos de tiempo (t), tamaño del lote (m), y dimensión de entrada (i)
    t, m, i = X.shape
    h = h_0.shape[1]
    
    # para guardar todos los estados ocultos
    H = np.zeros((t, m, h))
    # para guardar todas las salidas
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))
    
    # Establecer el estado oculto inicial
    h_prev = h_0
    
    # Iteramos sobre los pasos de tiempo
    for step in range(t):
        # Obtención de la entrada en el tiempo t
        x_t = X[step]
        
        # Propagación hacia adelante usando la celda RNN
        h_prev, y_t = rnn_cell.forward(h_prev, x_t)
        
        # Guardamos el estado oculto y la salida en sus respectivos arrays
        H[step] = h_prev
        Y[step] = y_t
    
    return H, Y
