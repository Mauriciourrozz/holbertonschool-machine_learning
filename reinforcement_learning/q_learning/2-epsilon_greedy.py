#!/usr/bin/env python3
"""
2-epsilon_greedy.py
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Selects the next action using the epsilon-greedy strategy.

    Parameters:
    - Q: Q table (numpy.ndarray)
    - state: current state
    - epsilon: probability of exploring

    Returns:
    - Index of the next action
    """
    p = np.random.uniform(0, 1)  # número aleatorio entre 0 y 1

    if p < epsilon:
        # elegir acción aleatoria
        action = np.random.randint(Q.shape[1])
    else:
        # elegir la mejor acción conocida
        action = np.argmax(Q[state])
    
    return action
