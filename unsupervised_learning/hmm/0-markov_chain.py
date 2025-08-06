#!/usr/bin/env python3
"""
0-markov_chain.py
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    Calculates the probability of being in each state after t steps
    in a Markov chain.

    Parameters:
    P (numpy.ndarray): A square 2D array of shape (n, n) representing
                       the transition matrix. P[i, j] is the probability
                       of transitioning from state i to state j.
    s (numpy.ndarray): A 2D row vector of shape (1, n) representing
                       the initial state probabilities.
    t (int): The number of steps to simulate. Default is 1.

    Returns:
    numpy.ndarray: A 2D row vector of shape (1, n) representing the
                   probabilities of being in each state after t steps,
                   or None if input is invalid.
    """
    try:
        pot = np.linalg.matrix_power(P, t)
        return s @ pot
    except Exception:
        return None
