#!/usr/bin/env python3
"""
6-multihead_attention.py
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    MultiHeadAttention layer for performing multi-head attention.

    Args:
        dm (int): Dimensionality of the model.
        h (int): Number of attention heads.

    Public Attributes:
        h (int): Number of heads.
        dm (int): Model dimensionality.
        depth (int): Depth of each attention head.
        Wq (tf.keras.layers.Dense): Dense layer to generate queries.
        Wk (tf.keras.layers.Dense): Dense layer to generate keys.
        Wv (tf.keras.layers.Dense): Dense layer to generate values.
        linear (tf.keras.layers.Dense): Dense layer for output of attention.
    """
    def __init__(self, dm, h):
        super(MultiHeadAttention, self).__init__()
        self.dm = dm
        self.h = h
        # profundidad de cada cabeza
        self.depth = dm // h

        # Capas Dense para Q, K y V
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        # Capa Dense final para la salida de la atención
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Divide el embedding en múltiples cabezas de atención.

        Args:
            x (tf.Tensor): Tensor de forma (batch, seq_len, dm)
            batch_size (int): Tamaño del batch

        Returns:
            tf.Tensor: Tensor de forma (batch, h, seq_len, depth)
        """
        # Reshape a (batch, seq_len, h, depth)
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        # Transpose a (batch, h, seq_len, depth)
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """
        Forward pass de la multi-head attention.

        Args:
            Q (tf.Tensor): Query tensor de forma (batch, seq_len_q, dm)
            K (tf.Tensor): Key tensor de forma (batch, seq_len_v, dm)
            V (tf.Tensor): Value tensor de forma (batch, seq_len_v, dm)
            mask (None): Siempre None

        Returns:
            output (tf.Tensor): Tensor de forma (batch, seq_len_q, dm)
            weights (tf.Tensor): Tensor de forma (batch, h, seq_len_q,
            seq_len_v)
        """
        batch_size = tf.shape(Q)[0]

        # Pasar Q, K, V por sus capas Dense
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        # Dividir en h cabezas
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Calcular la atención usando sdp_attention
        output, weights = sdp_attention(Q, K, V, mask)

        # Transponer y reshaping de vuelta a (batch, seq_len_q, dm)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_output = tf.reshape(output, (batch_size, -1, self.dm))

        # Pasar por la capa Dense final
        output = self.linear(concat_output)

        return output, weights
