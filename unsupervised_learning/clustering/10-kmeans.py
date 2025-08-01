#!/usr/bin/env python3
"""
10-kmeans.py
"""
import sklearn.cluster


def kmeans(X, k):
    """
    Performs the K-means algorithm to cluster a data set into k clusters.

    Parameters:
    ----------
    X : numpy.ndarray of shape (n, d)
    The data set where n is the number of points and d is the number of
    dimensions.
    k : int
    Number of clusters to form.

    Returns:
    -------
    c : numpy.ndarray of shape (k, d)
    The centroids (means) of each cluster.
    clss : numpy.ndarray of shape (n,)
    An array with the cluster label assigned to each point in X.
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=k)

    kmeans.fit(X)

    c = kmeans.cluster_centers_
    clss = kmeans.labels_

    return c, clss
