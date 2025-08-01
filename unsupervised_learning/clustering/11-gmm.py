#!/usr/bin/env python3
"""
11-gmm.py
"""
import sklearn.mixture


def gmm(X, k):
    """
    Performs Gaussian Mixture Model clustering on a dataset.

    Parameters:
    -----------
    X : numpy.ndarray of shape (n, d)
        The dataset where n is the number of data points and d is the
        number of features.
    k : int
        The number of mixture components (clusters).

    Returns:
    --------
    pi : numpy.ndarray of shape (k,)
        The weights (prior probabilities) of each mixture component.
    m : numpy.ndarray of shape (k, d)
        The mean vectors of each mixture component.
    S : numpy.ndarray of shape (k, d, d)
        The covariance matrices of each mixture component.
    clss : numpy.ndarray of shape (n,)
        The cluster index assigned to each data point.
    bic : float
        The Bayesian Information Criterion (BIC) value for the fitted model.
    """
    gmm = sklearn.mixture.GaussianMixture(n_components=k)

    gmm.fit(X)

    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)

    return pi, m, S, clss, bic
