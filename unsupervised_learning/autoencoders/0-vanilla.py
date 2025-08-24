#!/usr/bin/env python3
"""
0-vanilla.py
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Build an autoencoder (encoder, decoder, and full model).

    Args:
        input_dims (int): Input dimensionality of the model.
        hidden_layers (list[int]): Number of units for each
        encoder hidden layer (in order).
        latent_dims (int): Dimensionality of the latent space.

    Returns:
        (encoder: Model, decoder: Model, auto: Model)
    """
    # Encoder
    input_layer = keras.Input(shape=(input_dims,))
    x = input_layer
    for layer in hidden_layers:
        x = keras.layers.Dense(layer, activation='relu')(x)
    bottleneck = keras.layers.Dense(latent_dims, activation='relu')(x)
    encoder = keras.Model(inputs=input_layer, outputs=bottleneck)

    # Decoder
    input_layer_decoder = keras.Input(shape=(latent_dims,))
    x = input_layer_decoder
    for layer in reversed(hidden_layers):
        x = keras.layers.Dense(layer, activation='relu')(x)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=input_layer_decoder, outputs=decoder_output)

    # Autoencoder completo
    auto_output = decoder(encoder(input_layer))
    auto = keras.Model(inputs=input_layer, outputs=auto_output)

    # Compilación
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
