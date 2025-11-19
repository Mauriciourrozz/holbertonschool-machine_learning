#!/usr/bin/env python3
"""
3-rename.py
"""
import pandas as pd


def rename(df):
    """
    Renames the 'Timestamp' column to 'Datetime', converts it to datetime,
    and returns a DataFrame containing only the 'Datetime' and 'Close' columns.

    Args:
        df (pd.DataFrame): DataFrame containing a column named 'Timestamp'.

    Returns:
        pd.DataFrame: The modified DataFrame with 'Datetime' and 'Close'
        columns.
    """
    # Cambiar nombre de columna
    df = df.rename(columns={'Timestamp': 'Datetime'})

    # Convertir a datetimedf
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

    # Mostrar solo estas dos columnas
    df = df[['Datetime', 'Close']]

    return df
