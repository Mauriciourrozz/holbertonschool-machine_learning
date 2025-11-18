#!/usr/bin/env python3
"""
0-from_numpy.py
"""
import pandas as pd


def from_numpy(array):
    """
    Create a pandas DataFrame from a NumPy ndarray.

    Args:
        array (np.ndarray): Input NumPy array used to build the DataFrame.

    Returns:
        pandas.DataFrame: New DataFrame with alphabetically labeled columns
    """
    # Cantidad de columnas
    num_cols = array.shape[1]

    # Generar A, B, C..
    column_labels = [chr(ord('A') + i) for i in range(num_cols)]

    # Crear DataFrame
    return pd.DataFrame(array, columns=column_labels)
