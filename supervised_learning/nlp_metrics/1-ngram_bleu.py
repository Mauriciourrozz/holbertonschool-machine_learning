#!/usr/bin/env python3
"""
1-ngram_bleu.py
"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    calculates the n-gram BLEU score for a sentence

    Args:
    references (list of list of str): List of reference translations,
        each reference is a list of words.
    sentence (list of str): Candidate sentence as a list of words.
    n (int): The size of the n-gram.

    Returns:
        float: The n-gram BLEU score.
    """

    # Función interna para generar n-gramas de tamaño n
    def generate_ngrams(words, n):
        length = len(words)
        if n <= 0 or n > length:
            return []
        return [tuple(words[i:i+n]) for i in range(length - n + 1)]

    # Función interna para contar ocurrencias de n-gramas
    def count_ngrams(ngrams):
        counts = {}
        for gram in ngrams:
            counts[gram] = counts.get(gram, 0) + 1
        return counts

    # Generar n-gramas y contar en la oración candidata
    cand_ngrams = generate_ngrams(sentence, n)
    cand_counts = count_ngrams(cand_ngrams)

    # Generar y contar n-gramas para cada referencia
    max_ref_counts = {}
    for ref in references:
        ref_ngrams = generate_ngrams(ref, n)
        ref_counts = count_ngrams(ref_ngrams)
        for gram, count in ref_counts.items():
            max_ref_counts[gram] = max(max_ref_counts.get(gram, 0), count)

    # Calcular conteos con clipping
    clipped_count = 0
    for gram, count in cand_counts.items():
        clipped_count += min(count, max_ref_counts.get(gram, 0))

    # Precisión: proporción de n-gramas coincidentes
    total_cand_ngrams = len(cand_ngrams)
    if total_cand_ngrams == 0:
        precision = 0.0
    else:
        precision = clipped_count / total_cand_ngrams

    # Penalización por brevedad
    c_len = len(sentence)
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda ref_len: abs(ref_len - c_len))

    if c_len > closest_ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - closest_ref_len / c_len) if c_len > 0 else 0.0

    # BLEU score final
    bleu = bp * precision
    return bleu
