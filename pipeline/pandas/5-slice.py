#!/usr/bin/env python3
"""
5-slice.py
"""


def slice(df):
    """
    Extracts the High, Low, Close, and Volume_BTC columns,
    selects every 60th row, and returns the sliced DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing the required columns.

    Returns:
        pd.DataFrame: The sliced DataFrame.
    """
    # Quedarse solo con las columnas necesarias
    sub = df[['High', 'Low', 'Close', 'Volume_(BTC)']]

    # Seleccionar cada fila número 60
    return sub.iloc[::60]
