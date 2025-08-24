#!/usr/bin/env python3
"""
2-convolutional.py
"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Build a convolutional autoencoder (encoder, decoder, and full model).

    Args:
        input_dims (tuple[int]): Dimensions of the input (height, width,
        channels).
        filters (list[int]): Number of filters for each convolutional
        layer in the encoder.
        latent_dims (tuple[int]): Dimensions of the latent space
        (height, width, channels).

    Returns:
        encoder: the encoder model
        decoder: the decoder model
        auto: the full convolutional autoencoder model
    """

    # Encoder
    enc_input = keras.Input(shape=input_dims)
    h = enc_input

    for f in filters:
        h = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                padding='same')(h)
        h = keras.layers.MaxPooling2D((2, 2), padding='same')(h)

    encoder = keras.Model(inputs=enc_input, outputs=h)

    # Decoder
    dec_input = keras.Input(shape=latent_dims)
    h_dec = dec_input

    # Reverse filters except last one
    rev_filters = filters[::-1]
    for f in rev_filters[:-1]:
        h_dec = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                    padding='same')(h_dec)
        h_dec = keras.layers.UpSampling2D((2, 2))(h_dec)

    # Second to last layer with valid padding
    h_dec = keras.layers.Conv2D(rev_filters[-1], (3, 3), activation='relu',
                                padding='valid')(h_dec)
    h_dec = keras.layers.UpSampling2D((2, 2))(h_dec)

    # Final reconstruction layer
    dec_output = keras.layers.Conv2D(input_dims[-1], (3, 3),
                                     activation='sigmoid',
                                     padding='same')(h_dec)

    decoder = keras.Model(inputs=dec_input, outputs=dec_output)

    # Full autoencoder
    auto_input = keras.Input(shape=input_dims)
    auto_output = decoder(encoder(auto_input))
    auto = keras.Model(inputs=auto_input, outputs=auto_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
