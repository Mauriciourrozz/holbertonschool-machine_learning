#!/usr/bin/env python3
"""
5-dense_block.py
"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block as described in Dense Convolutional Networks
    (DenseNet-B).

    Parameters:
    - X: Input tensor from the previous layer.
    - nb_filters: Integer, the current number of filters in the input tensor.
    - growth_rate: Integer, the growth rate (number of filters to add per
    dense layer).
    - layers: Integer, the number of convolutional layers within the dense
    block.

    Returns:
    - X: Output tensor after concatenating the outputs of all layers in
    the dense block.
    - nb_filters: Integer, the updated total number of filters after
    the dense block.

    Details:
    Each layer within the block consists of:
      1. Batch Normalization
      2. ReLU activation
      3. 1x1 Convolution (bottleneck layer with 4 * growth_rate filters)
      4. Batch Normalization
      5. ReLU activation
      6. 3x3 Convolution (producing growth_rate filters)
    The output of each layer is concatenated with the block’s input to be
    passed into the next layer.
    """
    initializer = K.initializers.he_normal(seed=0)

    for i in range(layers):

        batch = K.layers.BatchNormalization(axis=3)(X)
        activation = K.layers.Activation("relu")(batch)
        conv1 = K.layers.Conv2D(filters=4 * growth_rate,
                                kernel_size=(1, 1), padding="same",
                                kernel_initializer=initializer)(activation)
        batch2 = K.layers.BatchNormalization(axis=3)(conv1)
        activation2 = K.layers.Activation("relu")(batch2)
        conv2 = K.layers.Conv2D(filters=growth_rate,
                                kernel_size=(3, 3), padding="same",
                                kernel_initializer=initializer)(activation2)
        X = K.layers.Concatenate(axis=3)([X, conv2])

        nb_filters += growth_rate

    return X, nb_filters
