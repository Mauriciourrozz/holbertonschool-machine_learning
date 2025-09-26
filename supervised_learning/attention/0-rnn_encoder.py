#!/usr/bin/env python3
"""
0-rnn_encoder.py
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    RNNEncoder is a custom Keras Layer that encodes input sequences
    into hidden representations for machine translation.
    It uses an Embedding layer followed by a GRU.
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor.

        Args:
            vocab (int): Size of the input vocabulary.
            embedding (int): Dimensionality of the embedding vectors.
            units (int): Number of hidden units in the GRU cell.
            batch (int): Batch size.

        Public Attributes:
            batch (int): Batch size.
            units (int): Number of hidden units in the GRU cell.
            embedding (tf.keras.layers.Embedding): Embedding layer to map
                words from the vocabulary into dense vectors.
            gru (tf.keras.layers.GRU): GRU layer with `units` hidden units,
                returning both sequences and the final hidden state.
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """
        Initializes the hidden state of the GRU to a tensor of zeros.

        Returns:
            tf.Tensor: Tensor of shape (batch, units) filled with zeros,
            representing the initial hidden state.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Executes the forward pass of the encoder.

        Args:
            x (tf.Tensor): Tensor of shape (batch, input_seq_len),
                containing input word indices from the vocabulary.
            initial (tf.Tensor): Tensor of shape (batch, units),
                containing the initial hidden state.

        Returns:
            outputs (tf.Tensor): Tensor of shape (batch, input_seq_len, units),
                containing the sequence of encoder outputs.
            hidden (tf.Tensor): Tensor of shape (batch, units),
                containing the last hidden state of the encoder.
        """
        # Convertimos los índices de palabras en vectores de embedding
        x = self.embedding(x)

        # Pasamos los embeddings por la GRU
        outputs, hidden = self.gru(x, initial_state=initial)

        return outputs, hidden
