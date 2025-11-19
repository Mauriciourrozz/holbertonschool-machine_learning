#!/usr/bin/env python3
"""
9-fill.py
"""


def fill(df):
    """
    Removes the 'Weighted_Price' column and fills missing values:
    - 'Close': forward fill (previous row's value)
    - 'High', 'Low', 'Open': filled with the row's 'Close'
    - 'Volume_(BTC)', 'Volume_(Currency)': filled with 0

    Returns the modified DataFrame.
    """
    # Eliminar la columna Weighted_Price
    if 'Weighted_Price' in df.columns:
        df = df.drop(columns=['Weighted_Price'])

    # Rellenar Close con el valor previo
    df['Close'] = df['Close'].fillna(method='ffill')

    # High, Low y Open se llenan con el Close de la misma fila
    for col in ['High', 'Low', 'Open']:
        if col in df.columns:
            df[col] = df[col].fillna(df['Close'])

    # Volume_(BTC) y Volume_(Currency) se llenan con 0
    for col in ['Volume_(BTC)', 'Volume_(Currency)']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    return df
