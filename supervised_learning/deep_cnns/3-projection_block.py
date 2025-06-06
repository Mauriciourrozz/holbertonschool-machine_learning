#!/usr/bin/env python3
"""
3-projection_block.py
"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Build a projection block as used in ResNet networks.

    This block applies a series of convolutions with batch normalization and
    ReLU activation, plus a projection path (shortcut) to match the dimensions
    between the input and the output of the block, allowing the addition.

    Parameters:
    -----------
    A_prev : tensor
        Output of the previous layer (block input).

    filters : tuple or list of 3 integers
        Numbers of filters for each convolutional layer of the block:
        - F11: number of filters for the first 1x1 convolution
        - F3: number of filters for 3x3 convolution
        - F12: number of filters for the second 1x1 convolution
        (and also for the projection)

    s : int, optional (default=2)
        Stride for the first convolution and for the projection path.
        This value controls the reduction of spatial dimensions.

    Returns:
    --------
    tensor
        Output of the block after applying addition with projection and ReLU
        activation.
    """
    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)

    branch1 = K.layers.Conv2D(
        F11, (1, 1), strides=s, padding="same",
        kernel_initializer=initializer)(A_prev)
    branch1 = K.layers.BatchNormalization(axis=3)(branch1)
    branch1 = K.layers.Activation("relu")(branch1)

    branch2 = K.layers.Conv2D(
        F3, (3, 3), padding="same",
        kernel_initializer=initializer)(branch1)
    branch2 = K.layers.BatchNormalization(axis=3)(branch2)
    branch2 = K.layers.Activation("relu")(branch2)

    branch3 = K.layers.Conv2D(
        F12, (1, 1), padding="same",
        kernel_initializer=initializer)(branch2)
    branch3 = K.layers.BatchNormalization(axis=3)(branch3)

    x = K.layers.Conv2D(
        F12, (1, 1), strides=s, padding="same",
        kernel_initializer=initializer)
    x = K.layers.BatchNormalization(axis=3)(x)

    output = K.layers.add([branch3, x])
    output = K.layers.Activation("relu")(output)

    return output
