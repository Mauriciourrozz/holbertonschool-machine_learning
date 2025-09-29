#!/usr/bin/env python3
"""
9-transformer_encoder.py
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """
    Transformer Encoder composed of N EncoderBlocks.

    Args:
        N (int): Number of encoder blocks.
        dm (int): Dimensionality of the model.
        h (int): Number of attention heads.
        hidden (int): Number of units in the hidden fully connected layer.
        input_vocab (int): Size of the input vocabulary.
        max_seq_len (int): Maximum possible sequence length.
        drop_rate (float): Dropout rate.

    Public Attributes:
        N (int): Number of encoder blocks.
        dm (int): Model dimensionality.
        embedding (tf.keras.layers.Embedding): Input embedding layer.
        positional encoding: Positional encodings tensor of shape (max_seq_len, dm).
        blocks (list): List of EncoderBlock instances.
        dropout (tf.keras.layers.Dropout): Dropout layer for positional encodings.
    """
    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        # Capa de embedding de entrada
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        # Positional encoding directamente usando la función importada
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        # Lista de EncoderBlocks
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        # Dropout aplicado a los embeddings con posicion
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """
        Forward pass of the Transformer encoder.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, input_seq_len)
            training (bool): Boolean indicating if the model is training
            mask (tf.Tensor or None): Mask for multi-head attention

        Returns:
            tf.Tensor: Output tensor of shape (batch, input_seq_len, dm)
        """
        seq_len = tf.shape(x)[1]

        # Convertir indices a embeddings
        x = self.embedding(x)
        # Aplicar codificación posicional
        x += self.positional_encoding[:seq_len, :]
        x = self.dropout(x, training=training)

        # Pasar por cada EncoderBlock
        for block in self.blocks:
            x = block(x, training, mask)

        return x
