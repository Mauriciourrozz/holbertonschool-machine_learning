#!/usr/bin/env python3
"""
11-config.py
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves the architecture of a Keras model to a file in JSON format.

    Parameters:
    network (keras.Model): The Keras model whose configuration will be saved.
    filename (str): The path to the file where the JSON configuration
    will be stored.

    Returns:
    None
    """
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)


def load_config(filename):
    """
    Loads the architecture of a Keras model from a JSON configuration file.

    Parameters:
    filename (str): The path to the JSON file containing the model
    configuration.

    Returns:
    keras.Model: The reconstructed Keras model based on the loaded
    configuration.
    """
    with open(filename, 'r') as f:
        model_json = f.read()

    model = K.models.model_from_json(model_json)
    return model
