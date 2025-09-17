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

    # pasar a minúsculas y eliminar signos de puntuación
    clean_sentences = []
    for sentence in sentences:
        sentence = sentence.lower()
        # solo letras y números
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

    # Inicializar matriz de embeddings
    embeddings = np.zeros((s, f), dtype=int)

    # Construir la matriz BoW
    for i in range(s):
        words_in_sentence = set(clean_sentences[i].split())
        for j in range(f):
            if features[j] in words_in_sentence:
                embeddings[i, j] = 1

    return embeddings, features
