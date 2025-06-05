#!/usr/bin/env python3
"""
1-inception_network.py
"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the Inception network as described in the paper
    "Going Deeper with Convolutions" (2014).

    The input shape is assumed to be (224, 224, 3), corresponding to
    a standard RGB image.

    The network consists of an initial convolution and max-pooling layer,
    followed by intermediate convolutional layers and multiple inception blocks
    with different filter configurations. It ends with average pooling,
    dropout, and a fully connected softmax layer for classification.

    Returns:
        keras.Model: the complete Keras model representing the Inception
        network
    """
    input = K.Input(shape=(224, 224, 3))

    block1 = K.layers.Conv2D(
        filters=64, kernel_size=(7, 7), strides=(2, 2),
        padding="same", activation="relu")(input)
    block1 = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2),
        padding="same")(block1)

    block2 = K.layers.Conv2D(
        filters=64, kernel_size=(1, 1), activation="relu")(block1)
    block2 = K.layers.Conv2D(
        filters=192, kernel_size=(3, 3), padding="same",
        activation="relu")(block2)
    block2 = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                   padding="same")(block2)

    block3a = inception_block(block2, [64, 96, 128, 16, 32, 32])
    block3b = inception_block(block3a, [128, 128, 192, 32, 96, 64])
    block3 = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                   padding="same")(block3b)

    block4a = inception_block(block3, [192, 96, 208, 16, 48, 64])
    block4b = inception_block(block4a, [160, 112, 224, 24, 64, 64])
    block4c = inception_block(block4b, [128, 128, 256, 24, 64, 64])
    block4d = inception_block(block4c, [112, 144, 288, 32, 64, 64])
    block4e = inception_block(block4d, [256, 160, 320, 32, 128, 128])
    block4 = K.layers.MaxPooling2D((3, 3), strides=(2, 2),
                                   padding="same")(block4e)

    block5a = inception_block(block4, [256, 160, 320, 32, 128, 128])
    block5b = inception_block(block5a, [384, 192, 384, 48, 128, 128])

    blockf = K.layers.AveragePooling2D((7, 7), strides=(1, 1),
                                       padding="valid")(block5b)
    blockf = K.layers.Dropout(0.4)(blockf)
    blockf = K.layers.Dense(1000, activation="softmax")(blockf)

    model = K.Model(inputs=input, outputs=blockf)
    return model
