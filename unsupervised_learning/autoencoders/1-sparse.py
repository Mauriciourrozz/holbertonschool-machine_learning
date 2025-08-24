#!/usr/bin/env python3
"""
1-sparse.py
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Build a sparse autoencoder (encoder, decoder, and full model).

    Args:
        input_dims (int): Input dimensionality of the model.
        hidden_layers (list[int]): Number of units for each encoder hidden
        layer (in order).
        latent_dims (int): Dimensionality of the latent space.
        lambtha (float): L1 regularization parameter on the latent layer.

    Returns:
        (encoder: Model, decoder: Model, auto: Model)
    """
    # Encoder
    enc_in = keras.Input(shape=(input_dims,), name="encoder_input")
    x = enc_in
    for i, units in enumerate(hidden_layers, start=1):
        x = keras.layers.Dense(units, activation="relu",
                               name=f"enc_dense_{i}")(x)

    # Capa latente con regularización L1
    latent = keras.layers.Dense(latent_dims, activation="relu",
                                activity_regularizer=keras.regularizers.l1(
                                    lambtha), name="latent")(x)
    encoder = keras.Model(inputs=enc_in, outputs=latent, name="encoder")

    # --- Decoder ---
    dec_in = keras.Input(shape=(latent_dims,), name="decoder_input")
    y = dec_in
    for j, units in enumerate(reversed(hidden_layers), start=1):
        y = keras.layers.Dense(units, activation="relu",
                               name=f"dec_dense_{j}")(y)
    dec_out = keras.layers.Dense(input_dims, activation="sigmoid",
                                 name="reconstruction")(y)
    decoder = keras.Model(inputs=dec_in, outputs=dec_out, name="decoder")

    # Autoencoder completo
    auto_out = decoder(encoder(enc_in))
    auto = keras.Model(inputs=enc_in, outputs=auto_out,
                       name="sparse_autoencoder")

    # Compilación
    auto.compile(optimizer=keras.optimizers.Adam(), loss="binary_crossentropy")

    return encoder, decoder, auto
