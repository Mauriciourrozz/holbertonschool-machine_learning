#!/usr/bin/env python3

"""
2-identity_block.py
"""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in the Deep Residual Learning paper

    Parameters:
    - A_prev: tensor, output from the previous layer
    - filters: tuple/list of three integers (F11, F3, F12)
        * F11: number of filters in the first 1x1 convolution
        * F3: number of filters in the 3x3 convolution
        * F12: number of filters in the second 1x1 convolution

    Returns:
    - Output tensor of the identity block after adding the shortcut and
    applying ReLU activation

    Description:
    The identity block consists of three convolutional layers:
    1- A 1x1 convolution reducing dimensionality.
    2- A 3x3 convolution processing spatial features.
    3- A 1x1 convolution restoring dimensionality.

    Batch normalization and ReLU activation are applied after the first two
    convolutions, and batch normalization is applied after the third
    convolution.

    The input tensor is added (skip connection) to the output of the third
    convolutional layer before applying a final ReLU activation. This skip
    connection helps mitigate the vanishing gradient problem and allows
    training of very deep networks.
    """
    F11, F3, F12 = filters

    rama1 = K.layers.Conv2D(F11, (1, 1), padding="same")(A_prev)
    rama1 = K.layers.BatchNormalization()(rama1)
    rama1 = K.layers.Activation("relu")(rama1)

    rama2 = K.layers.Conv2D(F3, (3, 3), padding="same")(rama1)
    rama2 = K.layers.BatchNormalization()(rama2)
    rama2 = K.layers.Activation("relu")(rama2)

    rama3 = K.layers.Conv2D(F12, (1, 1), padding="same")(rama2)
    rama3 = K.layers.BatchNormalization()(rama3)

    output = K.layers.add([A_prev, rama3])
    output = K.layers.Activation("relu")(output)

    return output
