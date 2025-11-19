#!/usr/bin/env python3
"""
12-hierarchy.py
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


def hierarchy(df1, df2):
    """
    Reorganizes the MultiIndex so Timestamp is first,
    filters both exchanges between timestamps 1417411980 and 1417417980,
    concatenates with keys (bitstamp, coinbase),
    and ensures chronological order.
    """

    df1 = index(df1)
    df2 = index(df2)

    start = 1417411980
    end = 1417417980

    df1_slice = df1.loc[start:end]
    df2_slice = df2.loc[start:end]

    combined = pd.concat(
        [df2_slice, df1_slice],
        keys=['bitstamp', 'coinbase']
    )

    combined = combined.reorder_levels([1, 0]).sort_index()

    return combined
