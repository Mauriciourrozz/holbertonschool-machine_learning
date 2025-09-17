#!/usr/bin/env python3
"""
3-gensim_to_keras.py
"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a Keras Embedding layer.

    Args:
        model: trained gensim word2vec model.

    Returns:
        Keras Embedding layer (trainable).
    """
    vocab_size = len(model.wv)

    # Dimensión de los vectores de palabras
    vector_size = model.vector_size

    # Matriz de embeddings ya entrenada por gensim
    embedding_matrix = model.wv.vectors

    # Crear la capa Embedding en Keras
    embedding_layer = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[embedding_matrix],
        trainable=True
    )

    return embedding_layer
