#!/usr/bin/env python3
"""
5-backward.py
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the backward algorithm for a Hidden Markov Model.

    Parameters:
    Observation (numpy.ndarray): shape (T,), indices of observations
    Emission (numpy.ndarray): shape (N, M), emission probabilities
    Transition (numpy.ndarray): shape (N, N), transition probabilities
    Initial (numpy.ndarray): shape (N, 1), initial state probabilities

    Returns:
    P (float): likelihood of the observations given the model
    B (numpy.ndarray): shape (N, T), backward path probabilities
    """
    # Validar que todos los parámetros sean arreglos de numpy
    if (not isinstance(Observation, np.ndarray) or
        not isinstance(Emission, np.ndarray) or
        not isinstance(Transition, np.ndarray) or
            not isinstance(Initial, np.ndarray)):
        return None, None

    N = Emission.shape[0]  # número de estados ocultos
    T = Observation.shape[0]  # número de observaciones

    # Matriz para probabilidades backward
    B = np.zeros((N, T))

    # Paso de inicialización: en el último tiempo, todas son 1
    B[:, -1] = 1

    # Paso backward (de T-2 hacia atrás)
    for t in range(T - 2, -1, -1):
        for i in range(N):
            B[i, t] = np.sum(
                Transition[i, :] * Emission[
                    :, Observation[t + 1]] * B[:, t + 1])

    # Calcular la probabilidad total P
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B
