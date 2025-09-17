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
    tokenized_sentences = []
    for sentence in sentences:
        # Convertir a minúsculas y eliminar puntuación y apóstrofes
        words = re.findall(r'\b\w+\b', sentence.lower())
        tokenized_sentences.append(words)

    # Construir vocabulario si no se proporciona
    if vocab is None:
        vocab = sorted(
            {word for sentence in tokenized_sentences for word in sentence})

    embeddings = np.zeros((len(sentences), len(vocab)), dtype=int)
    
    for i, sentence in enumerate(tokenized_sentences):
        for j, word in enumerate(vocab):
            embeddings[i, j] = sentence.count(word)
    
    # Convertir vocab a array de numpy para pasar algunos tests automáticos
    return embeddings, np.array(vocab)
