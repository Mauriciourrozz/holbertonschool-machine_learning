#!/usr/bin/env python3
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
    clean_sentences = []

    for sentence in sentences:
        # pasar a minúsculas
        sentence = sentence.lower()
        # eliminar puntuación
        sentence = re.sub(r"'s\b", "", sentence)
        sentence = re.sub(r"[^a-z0-9\s]", "", sentence)
        clean_sentences.append(sentence)

    # Crear vocabulario si no se pasa
    if vocab is None:
        all_words = set()
        for sentence in clean_sentences:
            words = sentence.split()
            all_words.update(words)
        vocab = sorted(all_words)

    features = vocab
    s = len(clean_sentences)
    f = len(features)

    embeddings = np.zeros((s, f), dtype=int)

    # Construir la matriz BoW con frecuencia
    for i in range(s):
        words = clean_sentences[i].split()
        for j in range(f):
            embeddings[i, j] = words.count(features[j])

    return embeddings, features