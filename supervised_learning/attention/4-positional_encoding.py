#!/usr/bin/env python3
"""
4-positional_encoding.py
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a Transformer.

    Args:
        max_seq_len (int): Maximum length of the input sequences.
        dm (int): Depth of the model (embedding dimension).

    Returns:
        np.ndarray: Array of shape (max_seq_len, dm) containing
                    the positional encoding vectors.
    """
    # Inicializar la matriz de codificación posicional
    PE = np.zeros((max_seq_len, dm))

    # crea un vector de pocisiones 0, 1, 2, hasta max_seq_len-1
    positions = np.arange(max_seq_len)[:, np.newaxis]

    # Crea un vector de dimensiones 0, 1, 2, hasta dm-1
    dims = np.arange(dm)[np.newaxis, :]

    # Calcular los ángulos para seno y coseno.
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(dm))
    angle_rads = positions * angle_rates

    # Aplicar seno a índices pares y coseno a índices impares
    PE[:, 0::2] = np.sin(angle_rads[:, 0::2])
    PE[:, 1::2] = np.cos(angle_rads[:, 1::2])

    return PE
