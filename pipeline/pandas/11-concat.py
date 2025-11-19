#!/usr/bin/env python3
"""
11-concat.py
"""
import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """
    Indexes both dataframes on 'Timestamp', filters df2 up to timestamp
    1417411920, concatenates df2 (bitstamp) on top of df1 (coinbase),
    and adds keys to distinguish the sources.

    Returns the concatenated DataFrame.
    """

    # Junta ambos dataframes usando index
    df1 = index(df1)
    df2 = index(df2)

    # Filtrar df2 la marca de tiempo
    df2_filtered = df2.loc[:1417411920]

    # Concatenar df2 encima de df1 con claves
    result = pd.concat(
        [df2_filtered, df1],
        keys=['bitstamp', 'coinbase']
    )

    return result
