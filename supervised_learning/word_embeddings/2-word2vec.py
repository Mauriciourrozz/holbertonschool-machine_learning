#!/usr/bin/env python3
"""
2-word2vec.py
"""
import gensim


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
    # Determinar tipo de entrenamiento: 0 = CBOW, 1 = Skip-gram
    sg = 0 if cbow else 1

    # Crear el modelo Word2Vec usando la ruta completa
    model = gensim.models.Word2Vec(sentences=sentences,
                                   vector_size=vector_size,
                                   window=window,
                                   min_count=min_count,
                                   negative=negative,
                                   sg=sg,
                                   seed=seed,
                                   workers=workers)
    
    model.build_vocab(sentences)

    model.train(sentences, total_examples=model.corpus_count, epochs=epochs)
    
    return model