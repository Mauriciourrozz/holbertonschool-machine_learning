#!/usr/bin/env python3
"""
0-bag_of words.py
"""
import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """
    Create a Bag of Words embedding matrix.

    Parameters:
    sentences (list of str): List of sentences to analyze.
    vocab (list of str, optional): List of vocabulary words to use.
                                   If None, all words in sentences are used.

    Returns:
    embeddings (numpy.ndarray): Array of shape (s, f) containing the embeddings
                                s = number of sentences, f = number of features
    features (list of str): List of vocabulary words used as features.
    """
    # Tokenizar y limpiar cada oración usando una comprensión de listas
    tokenized_sentences = [
        # palabras de 2 o más letras
        re.findall(r'\b[a-zA-Z]{2,}\b', s.lower())
        for s in sentences
    ]

    # Construir vocabulario si no se proporciona
    if vocab is None:
        vocab = sorted(
            {word for words in tokenized_sentences for word in words})

    word_to_index = {word: idx for idx, word in enumerate(vocab)}

    # Inicializar matriz de embeddings
    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)

    # Rellenar la matriz de embeddings
    for i, words in enumerate(tokenized_sentences):
        for word in words:
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += 1

    return embeddings, np.array(vocab)
