#!/usr/bin/env python3
"""
4-viterbi.py
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Calculates the most likely sequence of hidden states for
    a Hidden Markov Model.

    Parameters:
    Observation (numpy.ndarray): shape (T,), indices of observations
    Emission (numpy.ndarray): shape (N, M), emission probabilities
    Transition (numpy.ndarray): shape (N, N), transition probabilities
    Initial (numpy.ndarray): shape (N, 1), initial state probabilities

    Returns:
    path (list): length T, most likely sequence of hidden states
    P (float): probability of obtaining the path sequence
    """
   # Validar que todos los parámetros sean arreglos de numpy
    if (not isinstance(Observation, np.ndarray) or
        not isinstance(Emission, np.ndarray) or
        not isinstance(Transition, np.ndarray) or
        not isinstance(Initial, np.ndarray)):
        return None, None

    N = Emission.shape[0]  # número de estados ocultos
    T = Observation.shape[0]  # número de observaciones

    # Matriz para probabilidades Viterbi
    V = np.zeros((N, T))
    # Matriz para guardar el mejor estado previo
    backpointer = np.zeros((N, T), dtype=int)

    # Paso de inicialización (t = 0)
    V[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    # Rellenar V y backpointer (programación dinámica)
    for t in range(1, T):  # para cada tiempo
        for j in range(N):  # para cada estado
            # Calcula probabilidad de venir de cada estado previo → llegar a j
            probs = V[:, t-1] * Transition[:, j] * Emission[j, Observation[t]]
            # Guarda el índice del estado previo con mayor probabilidad
            backpointer[j, t] = np.argmax(probs)
            # Guarda la mayor probabilidad en V
            V[j, t] = np.max(probs)

    # Backtracking para reconstruir el camino más probable
    path = [np.argmax(V[:, -1])]  # comenzamos desde el último tiempo
    for t in range(T - 1, 0, -1):
        path.insert(0, backpointer[path[0], t])

    # Probabilidad final de la secuencia
    P = np.max(V[:, -1])
    return path, P
