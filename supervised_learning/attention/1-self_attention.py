#!/usr/bin/env python3
"""
1-self_attention.py
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    SelfAttention is a custom Keras Layer that implements Bahdanau-style
    additive attention for machine translation.
    It computes attention weights over the encoder hidden states given
    the previous decoder hidden state, and produces a context vector
    for the decoder.
    """
    def __init__(self, units):
        """
        Class constructor.

        Args:
            units (int): Number of hidden units in the alignment model.

        Public Attributes:
            W (tf.keras.layers.Dense): Dense layer with `units` units,
                applied to the previous decoder hidden state.
            U (tf.keras.layers.Dense): Dense layer with `units` units,
                applied to the encoder hidden states.
            V (tf.keras.layers.Dense): Dense layer with 1 unit,
                applied to the tanh of the sum of the outputs of W and U.
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Performs the forward pass of the attention mechanism.

        Args:
            s_prev (tf.Tensor): Tensor of shape (batch, units),
                containing the previous decoder hidden state.
            hidden_states (tf.Tensor): Tensor of shape (batch, input_seq_len,
                units), containing the outputs of the encoder.

        Returns:
            context (tf.Tensor): Tensor of shape (batch, units),
                representing the context vector for the decoder.
            weights (tf.Tensor): Tensor of shape (batch, input_seq_len, 1),
                representing the attention weights.
        """
        # Expandimos s_prev para que tenga la misma forma que hidden_states
        s_prev_expanded = tf.expand_dims(s_prev, axis=1)

        # Calculamos las energías (score)
        score = self.V(
            tf.nn.tanh(
                self.W(s_prev_expanded) + self.U(hidden_states)
            )
        )

        # Calculamos los pesos de atención (softmax a lo largo de la secuencia)
        weights = tf.nn.softmax(score, axis=1)

        # Calculamos el contexto como suma ponderada de hidden_states
        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
