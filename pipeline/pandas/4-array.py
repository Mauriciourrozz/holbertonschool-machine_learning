#!/usr/bin/env python3
"""
4-array.py
"""
import numpy as np


def array(df):
    """
    Selects the last 10 rows of the 'High' and 'Close' columns
    and converts them into a numpy.ndarray.

    Args:
        df (pd.DataFrame): DataFrame containing 'High' and 'Close' columns.

    Returns:
        numpy.ndarray: Array with the last 10 rows of High and Close.
    """
    # Seleccionar últimas 10 filas
    last = df[['High', 'Close']].tail(10)

    # Convertir a ndarray
    lastest = last.to_numpy()

    return lastest
