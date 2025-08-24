#!/usr/bin/env python3
"""
3-variational.py
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Build a Variational Autoencoder (VAE) with encoder, decoder, and full
    model.

    Args:
        input_dims (int): Dimensionality of the input data
            (number of features).
        hidden_layers (list[int]): List of integers representing the number of
            units in each hidden layer of the encoder. The decoder will reverse
            this order.
        latent_dims (int): Dimensionality of the latent space representation
        (size of z).

    Returns:
        encoder (keras.Model): Encoder model that takes input and outputs
        three tensors:
            - z: the sampled latent vector
            - z_mean: the mean of the latent distribution
            - z_log_var: the log variance of the latent distribution
        decoder (keras.Model): Decoder model that takes a latent vector z as
        input and reconstructs the input data.
        vae (keras.Model): Full VAE model connecting the encoder and decoder,
        compiled with the Adam optimizer. Includes the VAE loss
        (reconstruction + KL divergence).
    """

    # Sampling function
    def sample_latent(args):
        mean, log_var = args
        batch_size = keras.backend.shape(mean)[0]
        dim = keras.backend.int_shape(mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch_size, dim))
        return mean + keras.backend.exp(0.5 * log_var) * epsilon

    # Encoder
    enc_in = keras.Input(shape=(input_dims,), name='encoder_input')
    h = enc_in
    for units in hidden_layers:
        h = keras.layers.Dense(units, activation='relu')(h)

    z_mean = keras.layers.Dense(latent_dims, name='z_mean')(h)
    z_log_var = keras.layers.Dense(latent_dims, name='z_log_var')(h)
    z = keras.layers.Lambda(sample_latent, name='z')([z_mean, z_log_var])

    encoder = keras.Model(enc_in, [z, z_mean, z_log_var], name='encoder')

    # Decoder
    dec_in = keras.Input(shape=(latent_dims,), name='decoder_input')
    h_dec = dec_in
    for units in reversed(hidden_layers):
        h_dec = keras.layers.Dense(units, activation='relu')(h_dec)

    dec_out = keras.layers.Dense(input_dims, activation='sigmoid',
                                 name='decoder_output')(h_dec)
    decoder = keras.Model(dec_in, dec_out, name='decoder')

    # VAE full model
    z_enc, z_mean_enc, z_log_var_enc = encoder(enc_in)
    vae_out = decoder(z_enc)
    vae = keras.Model(enc_in, vae_out, name='vae')

    # VAE loss
    recon_loss = keras.losses.binary_crossentropy(enc_in, vae_out) * input_dims
    kl = 1 + z_log_var_enc - keras.backend.square(
        z_mean_enc) - keras.backend.exp(z_log_var_enc)
    kl_loss = keras.backend.sum(kl, axis=-1) * -0.5
    vae_loss = keras.backend.mean(recon_loss + kl_loss)
    vae.add_loss(vae_loss)

    vae.compile(optimizer='adam')

    return encoder, decoder, vae
