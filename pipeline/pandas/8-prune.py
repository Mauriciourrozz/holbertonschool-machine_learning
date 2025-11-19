#!/usr/bin/env python3
"""
8-prune.py
"""


def prune(df):
    """
    Removes rows where the 'Close' column contains NaN values.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame without NaN values in 'Close'.
    """
    return df.dropna(subset=['Close'])
