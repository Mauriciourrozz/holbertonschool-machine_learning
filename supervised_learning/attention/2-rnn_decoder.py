#!/usr/bin/env python3
"""
2-rnn_decoder.py
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    RNNDecoder is a custom Keras Layer that decodes sequences for
    machine translation. It uses an embedding layer, GRU, and
    Bahdanau-style self-attention to produce the output words.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor.

        Args:
            vocab (int): Size of the output vocabulary.
            embedding (int): Dimensionality of the embedding vectors.
            units (int): Number of hidden units in the GRU cell.
            batch (int): Batch size.

        Public Attributes:
            embedding (tf.keras.layers.Embedding): Embedding layer that
                converts words into embedding vectors.
            gru (tf.keras.layers.GRU): GRU layer with `units` hidden units,
                returning both sequences and the last hidden state.
            F (tf.keras.layers.Dense): Dense layer with `vocab` units
                to predict the next word.
            attention (SelfAttention): SelfAttention layer to compute
                context vectors from encoder outputs.
        """
        super(RNNDecoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        Performs the forward pass of the decoder.

        Args:
            x (tf.Tensor): Tensor of shape (batch, 1), containing
                the previous word as an index in the target vocabulary.
            s_prev (tf.Tensor): Tensor of shape (batch, units),
                containing the previous decoder hidden state.
            hidden_states (tf.Tensor): Tensor of shape (batch,
            input_seq_len, units), containing the outputs of the encoder.

        Returns:
            y (tf.Tensor): Tensor of shape (batch, vocab), containing
                the next word probabilities in the target vocabulary.
            s (tf.Tensor): Tensor of shape (batch, units), containing
                the new decoder hidden state.
        """
        # Calcular el vector de contexto usando la atención
        context, _ = self.attention(s_prev, hidden_states)

        # Convertir el índice de palabras anterior a embedding
        x = self.embedding(x)

        # Concatenar vector de contexto e incrustar a lo largo del último eje
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        x = tf.keras.layers.Dense(self.units)(x)

        # Pasar por GRU
        output, s = self.gru(x, initial_state=s_prev)

        # Aplana la salida y pasa a través de la capa para obtener predicciones
        output = tf.reshape(output, (-1, output.shape[2]))
        y = self.F(output)

        return y, s
