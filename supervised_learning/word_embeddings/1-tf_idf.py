#!/usr/bin/env python3
"""
1-tf_idf.py
"""
import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding array.

    phrases: list of sentences to analyze
    vocab: list of vocabulary words to use, or None to use all words
    Returns: embeddings, features.
    """
    # Tokenizar y limpiar cada oración
    tokenized_sentences = [
        re.findall(r'\b[a-zA-Z]{2,}\b', s.lower())
        for s in sentences
    ]

    # Construir vocabulario si no se proporciona
    if vocab is None:
        vocab = sorted({word for words in tokenized_sentences for word in words})

    N = len(sentences)

    # Calcular document frequency
    df = {}
    for word in vocab:
        df[word] = sum(1 for sentence in tokenized_sentences if word in sentence)

    # Inicializar matriz
    embeddings = np.zeros((N, len(vocab)), dtype=float)

    # Rellenar matriz TF-IDF
    word_to_index = {word: i for i, word in enumerate(vocab)}

    for i, words in enumerate(tokenized_sentences):
        counts = {}
        for w in words:
            # frecuencia absoluta
            counts[w] = counts.get(w, 0) + 1
        for w, c in counts.items():
            if w in word_to_index:
                j = word_to_index[w]
                idf = np.log((N + 1) / (1 + df[w]))
                embeddings[i, j] = c * idf

    return embeddings, np.array(vocab)