#!/usr/bin/env python3
"""
5-train.py
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, verbose=True, shuffle=False):
    """
    Trains a Keras model using mini-batch gradient descent.

    Args:
        network: The Keras model to be trained.
        data (numpy.ndarray): The input data, with shape (m, nx), where
        'm' is the number of examples and 'nx' is the number of features.
        labels (numpy.ndarray): The labels corresponding to the input data,
        with shape (m, classes), where 'classes' is the number of possible
        output classes.
        batch_size (int): The size of each mini-batch for gradient descent.
        epochs (int): The number of passes through the entire dataset.
        verbose (bool, optional): If True, displays training progress.
        Default is True.
        shuffle (bool, optional): If True, shuffles the data at the beginning
        of each epoch. Default is False.

    Returns:
        history: A Keras History object containing the training
        loss and metrics for each epoch.
    """
    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
