#!/usr/bin/env python3
"""
11-transformer.py
"""
import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """
    Transformer model that combines Encoder and Decoder.
    """

    def __init__(self, N, dm, h, hidden,
                 input_vocab, target_vocab,
                 max_seq_input, max_seq_target,
                 drop_rate=0.1):
        """
        Initialize the Transformer.
        """
        super(Transformer, self).__init__()

        # Encoder
        self.encoder = Encoder(N, dm, h, hidden,
                               input_vocab, max_seq_input,
                               drop_rate)

        # Decoder
        self.decoder = Decoder(N, dm, h, hidden,
                               target_vocab, max_seq_target,
                               drop_rate)

        # Capa final para predecir vocabulario de salida
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(self, inputs, target, training,
             encoder_mask, look_ahead_mask, decoder_mask):
        """
        Forward pass of the Transformer.
        """
        # pasar entradas por el Encoder
        encoder_output = self.encoder(inputs, training, encoder_mask)

        # pasar targets por el Decoder con salida del Encoder
        decoder_output = self.decoder(target, encoder_output,
                                      training,
                                      look_ahead_mask,
                                      decoder_mask)

        # capa final lineal (predicciones sobre vocabulario)
        final_output = self.linear(decoder_output)

        return final_output
