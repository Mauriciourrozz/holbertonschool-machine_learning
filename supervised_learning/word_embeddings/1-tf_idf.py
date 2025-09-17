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
        vocab = sorted(
            {word for words in tokenized_sentences for word in words})

    # Número total de documentos
    N = len(sentences)

    # Calcular cuántos documentos contienen cada palabra
    df = {word: 0 for word in vocab}
    for word in vocab:
        for sentence in tokenized_sentences:
            if word in sentence:
                df[word] += 1

    # Inicializar matriz TF-IDF
    embeddings = np.zeros((N, len(vocab)), dtype=float)

    for i, words in enumerate(tokenized_sentences):
        # frecuencia de palabras
        word_counts = {w: words.count(w) for w in words}
        total_words = len(words)
        for j, word in enumerate(vocab):
            # frecuencia normalizada
            tf = word_counts.get(word, 0) / total_words
            # IDF
            idf = np.log(N / (1 + df[word]))
            embeddings[i, j] = tf * idf

    return embeddings, np.array(vocab)
