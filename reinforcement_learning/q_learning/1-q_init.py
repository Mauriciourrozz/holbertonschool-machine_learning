#!/usr/bin/env python3
"""
1-q_init.py
"""
import numpy as np


def q_init(env):
    """
    Initializes the Q table with zeros.

    Parameters:
    - env: instance of the FrozenLakeEnv environment

    Returns:
    - Q-table: zero-based array of shape (n_states, n_actions)
    """
    n_states = env.observation_space.n  # cantidad de estados
    n_actions = env.action_space.n  # cantidad de acciones
    Q = np.zeros((n_states, n_actions))  # matriz de ceros
    return Q
