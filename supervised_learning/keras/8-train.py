#!/usr/bin/env python3
"""
7-train.py
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
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
        validation_data (tuple, optional): Data on which to evaluate
        the loss and anymodel metrics at the end of each epoch. Should be a
        tuple (val_data, val_labels).
        early_stopping (bool, optional): If True, enables early
        stopping based on validation loss.
        Only used if validation_data is provided. Default is False.
        patience (int, optional): Number of epochs with no improvemen
          after which training
            will be stopped if early stopping is enabled. Default is 0.
        verbose (bool, optional): If True, displays training progress.
        Default is True.
        shuffle (bool, optional): If True, shuffles the data at the beginning
        of each epoch.
            Default is False.

    Returns:
        history: A Keras History object containing the training
        loss and metrics for each epoch.
    """
    callbacks = []
    if validation_data:
        paradaTemprana = K.callbacks.EarlyStopping(
            monitor='val_loss', patience=patience)
        callbacks.append(paradaTemprana)

    if learning_rate_decay and validation_data:
        decaimiento = K.callbacks.LearningRateScheduler(
            lambda epochs: alpha / (1 + decay_rate * epochs),
            verbose=1)
        callbacks.append(decaimiento)

    if save_best and validation_data is not None:
        guardado = K.callbacks.ModelCheckpoint(
            filepath,
            save_best_only=True
        )
        callbacks.append(guardado)

    history = network.fit(data,
                          labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          callbacks=callbacks,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
