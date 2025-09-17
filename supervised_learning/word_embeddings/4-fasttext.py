#!/usr/bin/env python3
"""
4-fasttext.py
"""
import gensim


def fasttext_model(sentences, vector_size=100,
                   min_count=5, negative=5, window=5,
                   cbow=True, epochs=5, seed=0, workers=1):
    """
    Creates, builds, and trains a Gensim FastText model.

    Args:
        sentences (list): List of sentences used for training.
        vector_size (int): Dimensionality of the embedding layer.
        Default is 100.
        min_count (int): Minimum number of occurrences of a word to be
        included in training. Default is 5.
        negative (int): Size of negative sampling. Default is 5.
        window (int): Maximum distance between the current and predicted
        word within a sentence. Default is 5.
        cbow (bool): Training type; True for CBOW, False for Skip-gram.
        Default is True.
        epochs (int): Number of training iterations. Default is 5.
        seed (int): Random seed for reproducibility. Default is 0.
        workers (int): Number of worker threads to use for training.
        Default is 1.

    Returns:
        gensim.models.FastText: The trained FastText model.
    """
    if cbow:
        sg = 0
    else:
        sg = 1

    model = gensim.models.FastText(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=sg,
        workers=workers,
        seed=seed,
        epochs=epochs
    )

    model.build_vocab(sentences)

    model.train(
    sentences,
    total_examples=model.corpus_count,
    epochs=epochs
    )

    return model
