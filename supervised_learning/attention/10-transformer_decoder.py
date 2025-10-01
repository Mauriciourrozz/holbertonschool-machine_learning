#!/usr/bin/env python3
"""
10-transformer_decoder.py
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """
    Transformer Decoder composed of N DecoderBlocks.
    """

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm

        # Embedding de entrada (tokens del target)
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)

        # Positional encoding
        self.positional_encoding = positional_encoding(max_seq_len, dm)

        # Lista de bloques Decoder
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)
        ]

        # Dropout aplicado tras la suma embedding + posición
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training,
             look_ahead_mask, padding_mask):
        """
        Forward pass del Decoder.
        """
        seq_len = tf.shape(x)[1]

        # Convertir índices a embeddings
        x = self.embedding(x)

        # Ajustar codificación posicional
        positions = self.positional_encoding[:seq_len, :]
        positions = tf.expand_dims(positions, 0)

        # Sumar embeddings + posición
        x += positions

        # Dropout
        x = self.dropout(x, training=training)

        # Pasar por cada DecoderBlock
        for block in self.blocks:
            x = block(x, encoder_output, training,
                      look_ahead_mask, padding_mask)

        return x
