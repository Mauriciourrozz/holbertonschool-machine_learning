#!/usr/bin/env python3
"""
2-word2vec.py
"""
from gensim.models import Word2Vec
import re


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a Word2Vec model using gensim.

    Args:
        sentences (list of str): List of sentences to train the model on.
        vector_size (int): Dimensionality of the embedding vectors.
        min_count (int): Minimum number of occurrences for a word to be
        included in training.
        window (int): Maximum distance between the current word and predicted
        word within a sentence.
        negative (int): Size of negative sampling.
        cbow (bool): True for CBOW training, False for Skip-gram.
        epochs (int): Number of iterations to train over the dataset.
        seed (int): Seed for the random number generator.
        workers (int): Number of worker threads to train the model.

    Returns:
        model (gensim.models.Word2Vec): The trained Word2Vec model.
    """
    # Tokenizar cada oración en palabras
    tokenized_sentences = [sentence.lower().split() for sentence in sentences]

    # Determinar tipo de entrenamiento
    sg = 0 if cbow else 1

    # Crear el modelo Word2Vec
    model = Word2Vec(sentences=tokenized_sentences,
                     vector_size=vector_size,
                     window=window,
                     min_count=min_count,
                     negative=negative,
                     sg=sg,
                     seed=seed,
                     workers=workers)

    # Entrenar el modelo sobre las oraciones tokenizadas
    model.train(tokenized_sentences, total_examples=len(
        tokenized_sentences), epochs=epochs)

    return model
