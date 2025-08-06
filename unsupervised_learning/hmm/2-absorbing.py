#!/usr/bin/env python3
"""
2-absorbing.py
"""
import numpy as np


def absorbing(P):
    """
    Determine if a Markov chain is absorbing.

    Parameters:
    P(numpy.ndarray): Square transition matrix (n x n) where
                       P[i, j] is the probability of going from state i to j.

    Returns:
    bool: True if the string is absorbing, False otherwise or error.
    """
    if not isinstance(P, np.ndarray):
        return False

    n = P.shape[0]

    absorbing_states = np.where(np.isclose(np.diag(P), 1))[0]
    if len(absorbing_states) == 0:
        return False

    reachability = np.linalg.matrix_power(P, n**2)

    for i in range(n):
        if i not in absorbing_states:
            if not reachability[i, absorbing_states].any():
                return False

    return True
