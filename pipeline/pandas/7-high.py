#!/usr/bin/env python3
"""
7-high.py
"""


def high(df):
    """
    Sorts the DataFrame by the 'High' column in descending order.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame sorted by High (highest first).
    """
    return df.sort_values(by='High', ascending=False)
