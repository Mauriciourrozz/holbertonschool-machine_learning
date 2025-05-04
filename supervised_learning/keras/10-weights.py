#!/usr/bin/env python3
"""
10-weights.py
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    Saves the weights of a neural network model to a file.

    Parameters:
    network (keras.Model): The neural network model whose weights are to
    be saved.
    filename (str): The path and filename where the weights will be stored.
    save_format (str, optional): The format for saving the weights.
    Defaults to 'keras'.
    If you want to use a different format, you can specify it (e.g., 'h5').

    Returns:
    None: The model's weights are saved to the file specified.
    """
    network.save(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Loads the weights of a previously saved model from the given file.

    Parameters:
    network (keras.Model): The neural network model to which the
    weights will be loaded.
    filename (str): The path and filename from which the weights will
    be loaded.

    Returns:
    None: The model's weights are loaded into the provided network model.
    """
    network.load_weights(filename)
