#!/usr/bin/env python3
"""
2-cumulative_bleu.py
"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """
    Calculate the cumulative n-gram BLEU score for a sentence
    using uniform weights for all n-gram precisions up to n.

    Args:
        references (list of list of str): List of reference translations,
            each reference is a list of words.
        sentence (list of str): Candidate sentence as a list of words.
        n (int): The largest n-gram size to consider.

    Returns:
        float: The cumulative n-gram BLEU score.
    """

    # Función interna para generar n-gramas de tamaño k
    def generate_ngrams(words, k):
        length = len(words)
        if k <= 0 or k > length:
            return []
        return [tuple(words[i:i+k]) for i in range(length - k + 1)]

    # Función interna para contar ocurrencias de n-gramas
    def count_ngrams(ngrams):
        counts = {}
        for gram in ngrams:
            counts[gram] = counts.get(gram, 0) + 1
        return counts

    precisions = []
    for i in range(1, n+1):
        # Obtener n-gramas de candidato
        cand_ngrams = generate_ngrams(sentence, i)
        cand_counts = count_ngrams(cand_ngrams)

        # Obtener max conteos de n-gramas en referencias
        max_ref_counts = {}
        for ref in references:
            ref_ngrams = generate_ngrams(ref, i)
            ref_counts = count_ngrams(ref_ngrams)
            for gram, count in ref_counts.items():
                max_ref_counts[gram] = max(max_ref_counts.get(gram, 0), count)

        # Conteo con clipping
        clipped_count = 0
        total_count = len(cand_ngrams)
        for gram, count in cand_counts.items():
            clipped_count += min(count, max_ref_counts.get(gram, 0))

        # Precisión para n-grama i
        if total_count == 0:
            precision = 0.0
        else:
            precision = clipped_count / total_count

        precisions.append(precision)

    # Si alguna precisión es cero, BLEU acumulativo es cero
    if min(precisions) == 0:
        return 0.0

    # Promedio geométrico de las precisiones (log)
    log_precisions = [np.log(p) for p in precisions]
    avg_log_precision = sum(log_precisions) / n
    geo_mean = np.exp(avg_log_precision)

    # Penalización por brevedad
    c_len = len(sentence)
    ref_lens = [len(ref) for ref in references]
    closest_ref_len = min(ref_lens, key=lambda ref_len: abs(ref_len - c_len))

    if c_len > closest_ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - closest_ref_len / c_len) if c_len > 0 else 0.0

    # BLEU acumulativo final
    bleu = bp * geo_mean
    return bleu
