#!/usr/bin/env python3
"""
4-array.py
"""


def array(df):
    """
    Selects the last 10 rows of the 'High' and 'Close' columns
    and converts them into a numpy.ndarray.

    Args:
        df (pd.DataFrame): DataFrame containing 'High' and 'Close' columns.

    Returns:
        numpy.ndarray: Array with the last 10 rows of High and Close.
    """
    last = df[['High', 'Close']].tail(10)
    return last.to_numpy()
