#!/usr/bin/env python3
"""
3-mini_batch.py
"""
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches from the input data and labels.

    The function first shuffles the data randomly, then splits it into
    mini-batches of the specified size. If the total number of examples
    is not divisible by the batch size, the last mini-batch will contain
    the remaining examples.

    Parameters:
    X (numpy.ndarray): Input data of shape (m, nx), where m is the number
                       of examples and nx is the number of features.
    Y (numpy.ndarray): Labels corresponding to X, of shape (m, ny), where
                       ny is the number of classes or targets.
    batch_size (int): The number of examples in each mini-batch.

    Returns:
    list: A list of tuples, where each tuple contains a mini-batch:
          (X_batch, Y_batch)
    """
    x_shuffle, y_shuffle = shuffle_data(X, Y)
    mini_batches = []
    m = X.shape[0]

    for i in range(0, m, batch_size):
        X_batch = x_shuffle[i:i+batch_size]
        Y_batch = y_shuffle[i:i+batch_size]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
