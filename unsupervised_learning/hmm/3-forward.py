#!/usr/bin/env python3
"""
3-forward.py
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a Hidden Markov Model.

    Parameters:
    Observation (numpy.ndarray): shape (T,), indices of observations
    Emission (numpy.ndarray): shape (N, M), emission probabilities
    Transition (numpy.ndarray): shape (N, N), transition probabilities
    Initial (numpy.ndarray): shape (N, 1), initial state probabilities

    Returns:
    P (float): likelihood of the observations
    F (numpy.ndarray): shape (N, T), forward probabilities
    """
    if (not isinstance(Observation, np.ndarray) or
        not isinstance(Emission, np.ndarray) or
        not isinstance(Transition, np.ndarray) or
            not isinstance(Initial, np.ndarray)):
        return None, None

    N = Emission.shape[0]  # Numero de estados
    T = Observation.shape[0]  # Numero de observacioens
    F = np.zeros((N, T))

    # Initializacion
    F[:, 0] = Initial[:, 0] * Emission[:, Observation[0]]

    # probabilidad de estar en el estado anterior × probabilidad de saltar a j
    for t in range(1, T):
        for j in range(N):
            F[j, t] = np.sum(F[:, t-1] * Transition[:, j]) * Emission[
                j, Observation[t]]

    # probabilidad total
    P = np.sum(F[:, -1])
    return P, F
