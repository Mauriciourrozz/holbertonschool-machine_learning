#!/usr/bin/env python3
"""
4-resnet50.py
"""
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block
from tensorflow import keras as K


def resnet50():
    """
    
    """
    input = K.Input(shape=(224, 224, 3))
    initializer = K.initializers.he_normal(seed=0)

    # capa1
    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same", kernel_initializer=initializer)(input)
    batch1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation("relu")(batch1)
    maxpool1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(act1)

    # Capa 1
    proy1 = projection_block(maxpool1, [64, 64, 256], s=1)
    identity1 = identity_block(proy1, [64, 64, 256])
    identity2 = identity_block(identity1, [64, 64, 256])

    # Capa 2
    proy2_1 = projection_block(identity2, [128, 128, 512])
    identity2_1 = identity_block(proy2_1, [128, 128, 512])
    identity2_2 = identity_block(identity2_1, [128, 128, 512])
    identity2_3 = identity_block(identity2_2, [128, 128, 512])

    # capa 3
    proy3_1 = projection_block(identity2_3, [256, 256, 1024])
    identity3_1 = identity_block(proy3_1, [256, 256, 1024])
    identity3_2 = identity_block(identity3_1, [256, 256, 1024])
    identity3_3 = identity_block(identity3_2, [256, 256, 1024])
    identity3_4 = identity_block(identity3_3, [256, 256, 1024])
    identity3_5 = identity_block(identity3_4, [256, 256, 1024])

    # capa 4
    proy4_1 = projection_block(identity3_5, [512, 512, 2048])
    identity4_1 = identity_block(proy4_1, [512, 512, 2048])
    identity4_2 = identity_block(identity4_1, [512, 512, 2048])

    output = K.layers.AveragePooling2D((7, 7), strides=None)(identity4_2)
    output = K.layers.Dense(1000, activation="softmax", kernel_initializer=initializer)(output)

    model = K.models.Model(input, output)
    return model
