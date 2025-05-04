#!/usr/bin/env python3
"""
12-test.py
"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Evaluates a trained Keras model using test data and labels.

    Parameters:
    network (keras.Model): The trained Keras model to evaluate.
    data (numpy.ndarray): Input data to test the model.
    labels (numpy.ndarray): True labels corresponding to the input data.
    verbose (bool): Whether to display the evaluation progress
    (default is True).

    Returns:
    tuple: A tuple containing the loss and accuracy of the model on
    the test data.
    """
    loss, accuracy = network.evaluate(data, labels, verbose)
    return [loss, accuracy]
