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
    # Preprocesamiento: convertir a minúsculas y tokenizar
    def preprocess(sentence):
        # Convertir a minúsculas y eliminar caracteres no alfabéticos
        sentence = re.sub(r'[^a-zA-Z\s]', '', sentence.lower())
        return sentence.split()

    # Tokenizar todas las oraciones
    tokenized_sentences = [preprocess(sentence) for sentence in sentences]

    # Crear vocabulario si no se proporciona
    if vocab is None:
        # Usar conjunto para palabras únicas y ordenar alfabéticamente
        vocab = sorted(set(
            word for sentence in tokenized_sentences for word in sentence))

    # Crear mapeo de palabra a índice
    word_to_index = {word: idx for idx, word in enumerate(vocab)}

    # Inicializar matriz de embeddings
    # Número de oraciones
    s = len(sentences)
    # Número de características
    f = len(vocab)
    embeddings = np.zeros((s, f), dtype=int)

    # Llenar la matriz de embeddings
    for i, sentence_tokens in enumerate(tokenized_sentences):
        for word in sentence_tokens:
            if word in word_to_index:
                embeddings[i, word_to_index[word]] += 1

    return embeddings, vocab
