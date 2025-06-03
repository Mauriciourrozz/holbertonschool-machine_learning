#!/usr/bin/env python3
"""
0-inception_block.py
"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Builds an Inception block as described in the paper:
    "Going Deeper with Convolutions" (Szegedy et al., 2014)

    Parameters:
    - A_prev: keras tensor
        Output from the previous layer.
    - filters: tuple or list of 6 integers
        Contains the number of filters for each convolutional layer:
        * F1: number of filters for the 1x1 convolution.

        * F3R: number of filters for the 1x1 convolution before the 3x3
        convolution.

        * F3: number of filters for the 3x3 convolution.

        * F5R: number of filters for the 1x1 convolution before the 5x5
        convolution.

        * F5: number of filters for the 5x5 convolution.

        * FPP: number of filters for the 1x1 convolution after max pooling.

    Returns:
    - The concatenated output of the Inception block (keras tensor).
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    rama1 = K.layers.Conv2D(
        F1, (1, 1), activation="relu", padding="same")(A_prev)

    rama2 = K.layers.Conv2D(
        F3R, (1, 1), activation='relu', padding='same')(A_prev)
    rama2 = K.layers.Conv2D(
        F3, (3, 3), activation='relu', padding='same')(rama2)

    rama3 = K.layers.Conv2D(
        F5R, (1, 1), activation="relu", padding="same")(A_prev)
    rama3 = K.layers.Conv2D(
        F5, (5, 5), activation="relu", padding="same")(rama3)

    rama4 = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(1, 1), padding='same')(A_prev)
    rama4 = K.layers.Conv2D(
        FPP, (1, 1), activation='relu', padding='same')(rama4)

    output = K.layers.Concatenate(axis=-1)([rama1, rama2, rama3, rama4])

    return output
