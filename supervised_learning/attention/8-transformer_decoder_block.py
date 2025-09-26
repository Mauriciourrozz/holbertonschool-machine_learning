#!/usr/bin/env python3
"""
8-transformer_decoder_block.py
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    Decoder block for a Transformer.

    Args:
        dm (int): Dimensionality of the model.
        h (int): Number of attention heads.
        hidden (int): Number of units in the hidden fully connected layer.
        drop_rate (float): Dropout rate.

    Public Attributes:
        mha1 (MultiHeadAttention): First multi-head attention layer
        (self-attention).
        mha2 (MultiHeadAttention): Second multi-head attention layer
        (encoder-decoder attention).
        dense_hidden (tf.keras.layers.Dense): Hidden dense layer with
        ReLU activation.
        dense_output (tf.keras.layers.Dense): Output dense layer with dm units.
        layernorm1 (tf.keras.layers.LayerNormalization): First layer
        normalization.
        layernorm2 (tf.keras.layers.LayerNormalization): Second layer
        normalization.
        layernorm3 (tf.keras.layers.LayerNormalization): Third layer
        normalization.
        dropout1 (tf.keras.layers.Dropout): First dropout layer.
        dropout2 (tf.keras.layers.Dropout): Second dropout layer.
        dropout3 (tf.keras.layers.Dropout): Third dropout layer.
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        super(DecoderBlock, self).__init__()
        # Atención multi-cabeza: self-attention
        self.mha1 = MultiHeadAttention(dm, h)
        # Atención multi-cabeza: encoder-decoder attention
        self.mha2 = MultiHeadAttention(dm, h)
        # Feed-forward
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        # Layer norms
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        # Dropouts
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass of the decoder block.

        Args:
            x (tf.Tensor): Input tensor of shape (batch, target_seq_len, dm)
            encoder_output (tf.Tensor): Output of the encoder of shape (batch,
            input_seq_len, dm)
            training (bool): Boolean indicating if the model is training
            look_ahead_mask (tf.Tensor or None): Mask for first attention layer
            padding_mask (tf.Tensor or None): Mask for second attention layer

        Returns:
            tf.Tensor: Output tensor of shape (batch, target_seq_len, dm)
        """
        # Self-attention con máscara look-ahead
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        # Conexión residual + LayerNorm
        out1 = self.layernorm1(x + attn1)

        # Encoder-decoder attention con máscara de padding
        attn2, _ = self.mha2(out1,
                             encoder_output, encoder_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        # Conexión residual + LayerNorm
        out2 = self.layernorm2(out1 + attn2)

        # Feed-forward
        ff_output = self.dense_hidden(out2)
        ff_output = self.dense_output(ff_output)
        ff_output = self.dropout3(ff_output, training=training)
        # Conexión residual + LayerNorm
        out3 = self.layernorm3(out2 + ff_output)

        return out3
