#!/usr/bin/env python3
"""
13-predict.py
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Uses a trained Keras model to make predictions on new data.

    Parameters:
    network (keras.Model): The trained Keras model used for prediction.
    data (numpy.ndarray): The input data to make predictions on.
    verbose (bool): Whether to display the prediction progress
    (default is False).

    Returns:
    numpy.ndarray: The predicted outputs from the model.
    """
    prediction = network.predict(data, verbose=verbose)
    return prediction
