#!/usr/bin/env python3
"""
1-ngram_bleu.py
"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    Calculate the n-gram BLEU score for a given candidate sentence.

    Args:
        references (list of list of str): List of reference translations,
            each reference is a list of words.
        sentence (list of str): Candidate sentence as a list of words.
        n (int): The size of the n-gram.

    Returns:
        float: The n-gram BLEU score.
    """

    # Función auxiliar para obtener n-gramas a partir de una lista de palabras
    def get_ngrams(words, n):
        ngrams = []
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i:i+n])
            ngrams.append(ngram)
        return np.array(ngrams, dtype=object)

    # Obtener los n-gramas de la oración candidata y sus conteos
    cand_ngrams = get_ngrams(sentence, n)
    if len(cand_ngrams) == 0:
        return 0.0  # no hay n-gramas si la oración es más corta que n

    # Obtener los n-gramas únicos y sus frecuencias en la oración candidata
    cand_words, cand_counts = np.unique(cand_ngrams, return_counts=True)

    # Para cada referencia, obtener los n-gramas y sus conteos máximos
    max_ref_counts = {}
    for ref in references:
        ref_ngrams = get_ngrams(ref, n)
        ref_words, ref_counts = np.unique(ref_ngrams, return_counts=True)
        ref_dict = dict(zip(ref_words, ref_counts))

        for w, c in zip(cand_words, cand_counts):
            max_ref_counts[w] = max(
                max_ref_counts.get(w, 0), ref_dict.get(w, 0))

    # sumar el mínimo entre el conteo candidato y el máximo en referencias
    clipped_count = 0
    for w, c in zip(cand_words, cand_counts):
        clipped_count += min(c, max_ref_counts.get(w, 0))

    total_count = len(cand_ngrams)
    precision = clipped_count / total_count if total_count > 0 else 0

    # Penalización por brevedad
    c_len = len(sentence)
    ref_lens = np.array([len(ref) for ref in references])
    r_len = ref_lens[np.argmin(np.abs(ref_lens - c_len))]

    if c_len > r_len:
        bp = 1.0
    else:
        bp = np.exp(1 - r_len / c_len) if c_len > 0 else 0

    return bp * precision
