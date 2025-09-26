#!/usr/bin/env python3
"""
5-sdp_attention.py
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention.

    Args:
        Q (tf.Tensor): Query tensor of shape (..., seq_len_q, dk)
        K (tf.Tensor): Key tensor of shape (..., seq_len_v, dk)
        V (tf.Tensor): Value tensor of shape (..., seq_len_v, dv)
        mask (tf.Tensor or None): Optional mask tensor broadcastable to
                                  (..., seq_len_q, seq_len_v)

    Returns:
        output (tf.Tensor): Tensor of shape (..., seq_len_q, dv)
                            containing the attention output
        weights (tf.Tensor): Tensor of shape (..., seq_len_q, seq_len_v)
                             containing the attention weights
    """
    # Obtenemos la dimensión de las keys para el escalado
    dk = tf.cast(tf.shape(K)[-1], tf.float32)

    # Calculamos el producto punto QK^T
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    # Escalamos por sqrt(dk)
    scaled_qk = matmul_qk / tf.math.sqrt(dk)

    # Aplicamos la máscara si existe
    if mask is not None:
        # Sumamos -1e9 a las posiciones enmascaradas
        scaled_qk += (mask * -1e9)

    # Calculamos los pesos de atención con softmax
    weights = tf.nn.softmax(scaled_qk, axis=-1)

    # Multiplicamos los pesos por V para obtener la salida final
    output = tf.matmul(weights, V)

    return output, weights
