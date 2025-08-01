#!/usr/bin/env python3
"""
12-agglomerative.py
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering using Ward linkage and 
    plots a colored dendrogram.

    Parameters:
    -----------
    X : numpy.ndarray of shape (n, d)
        Dataset with n samples and d features.
    dist : float
        Maximum cophenetic distance to cut the dendrogram.

    Returns:
    --------
    clss : numpy.ndarray of shape (n,)
        Cluster labels for each data point.
    """
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')

    clss = scipy.cluster.hierarchy.fcluster(Z, t=dist, criterion='distance')

    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.show()

    return clss
