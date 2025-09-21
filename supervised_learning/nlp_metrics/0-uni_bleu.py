#!/usr/bin/env python3
"""
0-uni_bleu.py
"""
import numpy as np


def uni_bleu(references, sentence):
    """
    Calculate the unigram BLEU score for a given candidate sentence.

    Args:
        references (list of list of str): A list of reference translations,
            where each reference is a list of words.
        sentence (list of str): A list of words representing the candidate
        sentence.

    Returns:
        float: The unigram BLEU score, a value between 0 and 1 indicating the
        similarity of the candidate sentence to the reference translations.
    """
    sentence = np.array(sentence, dtype=object)

    # Contar ocurrencias en la oración generada
    words_sen, counts_sen = np.unique(sentence, return_counts=True)

    # Preparar un diccionario para guardar máximo conteo por palabra
    max_ref_counts = {}

    for ref in references:
        ref = np.array(ref, dtype=object)
        words_ref, counts_ref = np.unique(ref, return_counts=True)

        # Crear diccionario para esta referencia
        ref_dict = dict(zip(words_ref, counts_ref))

        for w, c in zip(words_sen, counts_sen):
            max_ref_counts[w] = max(
                max_ref_counts.get(w, 0), ref_dict.get(w, 0))

    # Calcular clipped counts sumando mínimo entre counts_sen y max_ref_counts
    clipped_count = 0
    for w, c in zip(words_sen, counts_sen):
        clipped_count += min(c, max_ref_counts.get(w, 0))

    total_count = len(sentence)

    # Precisión
    precision = clipped_count / total_count if total_count > 0 else 0

    # Encontrar referencia con longitud más cercana
    ref_lens = np.array([len(ref) for ref in references])
    c = total_count
    r = ref_lens[np.argmin(np.abs(ref_lens - c))]

    # Brevity Penalty
    if c > r:
        bp = 1.0
    else:
        bp = np.exp(1 - r / c) if c > 0 else 0

    return bp * precision
