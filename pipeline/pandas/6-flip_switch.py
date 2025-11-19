#!/usr/bin/env python3
"""
6-flip_switch.py
"""


def flip_switch(df):
    """
    Sorts the DataFrame in reverse chronological order,
    transposes it, and returns the transformed DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The sorted and transposed DataFrame.
    """
    # Ordenar al revés
    df_sorted = df.sort_index(ascending=False)

    # Transponer
    return df_sorted.transpose()
