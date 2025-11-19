#!/usr/bin/env python3
"""
10-index.py
"""


def index(df):
    """
    Sets the 'Timestamp' column as the DataFrame index.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with Timestamp as index.
    """
    df = df.set_index('Timestamp')
    return df
