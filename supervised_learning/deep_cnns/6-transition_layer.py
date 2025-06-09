#!/usr/bin/env python3
"""
6-transition_layer.py
"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer as described in Dense Convolutional Networks
    (DenseNet-C).

    Parameters:
    - X: Input tensor from the previous layer.
    - nb_filters: Integer, the current number of filters in the input tensor.
    - compression: Float between 0 and 1, the compression factor to reduce
    the number of filters.

    Returns:
    - X: Output tensor after applying the transition layer.
    - nb_filters: Integer, the updated number of filters after compression.

    Details:
    The transition layer consists of:
      1. Batch Normalization
      2. ReLU activation
      3. 1x1 Convolution with filters reduced by the compression factor
      4. 2x2 Average Pooling with stride 2 to reduce spatial dimensions
      by half.
    The convolution weights are initialized using He normal initialization
    with seed zero.
    """
    filters = int(nb_filters * compression)
    initializer = K.initializers.he_normal(seed=0)

    batch = K.layers.BatchNormalization(axis=3)(X)
    act = K.layers.Activation("relu")(batch)
    conv = K.layers.Conv2D(filters=filters, kernel_size=(1, 1), padding="same",
                           kernel_initializer=initializer)(act)
    apool = K.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(conv)

    return apool, filters
