#!/usr/bin/env python3
"""
4-create_masks.py
"""
import tensorflow as tf


def create_padding_mask(seq):
    """
    Creates a padding mask for a given sequence.
    Args:
        seq: tf.Tensor of shape (batch_size, seq_len).
    Returns:
        mask: tf.Tensor of shape (batch_size, 1, 1, seq_len).
    """
    # Crear máscara, 1 en los paddings (tokens == 0), 0 en los tokens válidos
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # Expandir dimensiones para que se pueda aplicar en la atención
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(size):
    """
    Creates a look-ahead mask to hide future tokens.
    Args:
        size: length of the target sequence.
    Returns:
        mask: tf.Tensor of shape (1, size, size).
    """
    # Matriz triangular superior (1 en posiciones futuras)
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def create_masks(inputs, target):
    """
    Creates encoder, decoder, and combined masks for training.
    Args:
        inputs: tf.Tensor of shape (batch_size, seq_len_in) - input sequence.
        target: tf.Tensor of shape (batch_size, seq_len_out) - target sequence.
    Returns:
        encoder_mask: padding mask for encoder (batch_size, 1,1, seq_len_in).
        combined_mask: look-ahead + decoder target padding mask
                       (batch_size,1, seq_len_out, seq_len_out).
        decoder_mask: padding mask for encoder-decoder attention
                      (batch_size, 1, 1, seq_len_in).
    """

    # Máscara del encoder para inputs
    encoder_mask = create_padding_mask(inputs)

    # Máscara del decoder (2da atención, encoder-decoder)
    decoder_mask = create_padding_mask(inputs)

    # Máscara de padding en target
    decoder_target_padding_mask = create_padding_mask(target)

    # Look-ahead mask (no ver posiciones futuras)
    seq_len = tf.shape(target)[1]
    look_ahead_mask = create_look_ahead_mask(seq_len)

    # Expandir batch_size en look-ahead mask
    look_ahead_mask = tf.maximum(
        decoder_target_padding_mask,
        look_ahead_mask[tf.newaxis, tf.newaxis, :, :]
    )

    combined_mask = look_ahead_mask

    return encoder_mask, combined_mask, decoder_mask
