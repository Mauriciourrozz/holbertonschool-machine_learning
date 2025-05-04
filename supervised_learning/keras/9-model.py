#!/usr/bin/env python3
"""
9-model.py
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves a neural network model to a file.

    Parameters:
    network (keras.Model): The neural network model to be saved.
    filename (str): The path and filename where the model will be stored.

    Returns:
    None
    """
    network.save(filename, save_format='keras')


def load_model(filename):
    """
    Loads a previously saved neural network model from a file.

    Parameters:
    filename (str): The path and filename from which to load the model.

    Returns:
    keras.Model: The loaded neural network model.
    """
    modelo = K.models.load_model(filename)
    return modelo
