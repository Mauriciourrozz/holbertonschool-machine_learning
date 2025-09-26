#!/usr/bin/env python3
"""
7-transformer_encoder_block.py
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    Encoder block for a Transformer.

    Args:
        dm (int): Dimensionality of the model.
        h (int): Number of attention heads.
        hidden (int): Number of units in the hidden fully connected layer.
        drop_rate (float): Dropout rate.

    Public Attributes:
        mha (MultiHeadAttention): Multi-head attention layer.
        dense_hidden (tf.keras.layers.Dense): Hidden dense layer with ReLU
        activation.
        dense_output (tf.keras.layers.Dense): Output dense layer with dm units.
        layernorm1 (tf.keras.layers.LayerNormalization): First layer
        normalization layer.
        layernorm2 (tf.keras.layers.LayerNormalization): Second layer
        normalization layer.
        dropout1 (tf.keras.layers.Dropout): First dropout layer.
        dropout2 (tf.keras.layers.Dropout): Second dropout layer.
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Forward pass of the encoder block.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, input_seq_len, dm)
            training (bool): Boolean indicating if the model is training
            mask (tf.Tensor or None): Mask for multi-head attention

        Returns:
            tf.Tensor: Output tensor of shape (batch, input_seq_len, dm)
        """
        # Aplicar multi-head attention con residual connection y normalización
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # Feed-forward
        ff_output = self.dense_hidden(out1)
        ff_output = self.dense_output(ff_output)
        ff_output = self.dropout2(ff_output, training=training)
        out2 = self.layernorm2(out1 + ff_output)

        return out2
