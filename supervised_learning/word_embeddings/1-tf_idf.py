#!/usr/bin/env python3
"""
1-tf_idf.py
"""
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding array.

    phrases: list of sentences to analyze
    vocab: list of vocabulary words to use, or None to use all words
    Returns: embeddings, features.
    """
    # Tokenizar y limpiar cada oración
    clean_sentences = list(map(
        lambda s: " ".join(re.findall(r'\b[a-zA-Z]{2,}\b', s.lower())),
        sentences
    ))

    # Crear vocabulario si no se proporciona
    if vocab is None:
        vocab = sorted(
            {word for sent in clean_sentences for word in sent.split()})

    # Crear vectorizador TF-IDF
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    # Generar matriz TF-IDF
    embeddings = vectorizer.fit_transform(clean_sentences).toarray()

    # Extraer las features usadas
    features = vectorizer.get_feature_names_out()

    return embeddings, features
