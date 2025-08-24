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
    enc_in = keras.Input(shape=input_dims, name="encoder_input")
    x = enc_in
    for i, f in enumerate(filters, start=1):
        x = keras.layers.Conv2D(filters=f, kernel_size=(3,3), padding="same", activation="relu",
                                name=f"enc_conv_{i}")(x)
        x = keras.layers.MaxPooling2D(pool_size=(2,2), padding="same", name=f"enc_pool_{i}")(x)

    encoder = keras.Model(inputs=enc_in, outputs=x, name="encoder")

    # Decoder
    dec_in = keras.Input(shape=x.shape[1:], name="decoder_input")
    y = dec_in

    # Primeras tres capas convolucionales con UpSampling
    y = keras.layers.Conv2D(filters=filters[-1], kernel_size=(3,3), padding="same", activation="relu",
                            name="dec_conv_1")(y)
    y = keras.layers.UpSampling2D(size=(2,2), name="dec_upsample_1")(y)

    y = keras.layers.Conv2D(filters=filters[-1], kernel_size=(3,3), padding="same", activation="relu",
                            name="dec_conv_2")(y)
    y = keras.layers.UpSampling2D(size=(2,2), name="dec_upsample_2")(y)

    y = keras.layers.Conv2D(filters=filters[-1], kernel_size=(3,3), padding="same", activation="relu",
                            name="dec_conv_3")(y)
    y = keras.layers.UpSampling2D(size=(2,2), name="dec_upsample_3")(y)

    # Segunda a última convolución con padding "valid"
    y = keras.layers.Conv2D(filters=filters[0], kernel_size=(3,3), padding="valid", activation="relu",
                            name="dec_conv_second_last")(y)
    y = keras.layers.UpSampling2D(size=(2,2), name="dec_upsample_second_last")(y)

    # Última convolución con filtros igual a canales de entrada
    dec_out = keras.layers.Conv2D(filters=input_dims[-1], kernel_size=(3,3), padding="same",
                                  activation="sigmoid", name="dec_conv_last")(y)

    decoder = keras.Model(inputs=dec_in, outputs=dec_out, name="decoder")

    # Autoencoder completo
    auto_out = decoder(encoder(enc_in))
    auto = keras.Model(inputs=enc_in, outputs=auto_out, name="conv_autoencoder")

    # Compilación
    auto.compile(optimizer=keras.optimizers.Adam(), loss="binary_crossentropy")

    return encoder, decoder, auto