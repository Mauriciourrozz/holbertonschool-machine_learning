#!/usr/bin/env python3
"""
8-EM.py
"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the EM algorithm for a GMM.

    Parameters:
    - X: (n, d) dataset
    - k: number of clusters
    - iterations: max number of iterations
    - tol: tolerance for early stopping
    - verbose: print log likelihood every 10 steps and last step

    Returns:
    - pi: (k,) priors
    - m: (k, d) means
    - S: (k, d, d) covariances
    - g: (k, n) posterior probabilities
    - l: log likelihood
    """
    if (not isinstance(X, np.ndarray) or not isinstance(k, int) or
        not isinstance(iterations, int) or not isinstance(tol, float) or
        not isinstance(verbose, bool) or
            X.ndim != 2 or k <= 0 or iterations <= 0 or tol < 0):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    g, l_old = expectation(X, pi, m, S)

    if g is None:
        return None, None, None, None, None

    for i in range(iterations):
        pi, m, S = maximization(X, g)
        if pi is None:
            return None, None, None, None, None

        g, l_new = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None

        if verbose and (i % 10 == 0 or i == iterations - 1):
            print(f"Log Likelihood after {i} iterations: {l_new:.5f}")

        if abs(l_new - l_old) <= tol:
            if verbose and i % 10 != 0:
                print(f"Log Likelihood after {i} iterations: {l_new:.5f}")
            break

        l_old = l_new

    return pi, m, S, g, l_new
