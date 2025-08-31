#!/usr/bin/env python3
"""
Builds a convolutional Generator and Discriminator
"""

import tensorflow as tf
from tensorflow import keras


def convolutional_GenDiscr():
    """
    Builds a convolutional Generator and Discriminator

    Returns:
        generator (keras.Model): Generator model
        discriminator (keras.Model): Discriminator model
    """

    def generator():
        """
        Creates the Generator model

        Input shape: (16,)
        Output shape: (16, 16, 1)
        """
        inputs = keras.Input(shape=(16,))

        # Expandir dimensiones con capas Dense
        x = keras.layers.Dense(128, activation='tanh')(inputs)
        x = keras.layers.Dense(16 * 16, activation='tanh')(x)

        # Convertir a formato tipo imagen
        x = keras.layers.Reshape((16, 16, 1))(x)

        # Agregar capas convolucionales
        x = keras.layers.Conv2D(
            32, (3, 3),
            activation='tanh',
            padding='same')(x)
        x = keras.layers.Conv2D(
            1, (3, 3),
            activation='tanh',
            padding='same')(x)

        return keras.Model(inputs, x, name="generator")

    def discriminator():
        """
        Creates the Discriminator model

        Input shape: (16, 16, 1)
        Output shape: scalar probability
        """
        inputs = keras.Input(shape=(16, 16, 1))

        # Extracción de características con convoluciones
        x = keras.layers.Conv2D(
            32, (3, 3),
            activation='tanh',
            padding='same')(inputs)
        x = keras.layers.Conv2D(
            64, (3, 3),
            activation='tanh',
            padding='same')(x)

        # Aplanar para clasificación
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(128, activation='tanh')(x)

        # Capa final con salida escalar
        outputs = keras.layers.Dense(1, activation='tanh')(x)

        return keras.Model(inputs, outputs, name="discriminator")

    return generator(), discriminator()
