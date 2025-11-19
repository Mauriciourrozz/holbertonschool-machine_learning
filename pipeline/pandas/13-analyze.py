#!/usr/bin/env python3
"""
13-analyze.py
"""


def analyze(df):
    """
    Analyze a DataFrame by returning descriptive statistics
    for all columns except Timestamp.
    """
    # Exclude Timestamp if present
    cols = [c for c in df.columns if c != 'Timestamp']

    # Compute descriptive statistics
    stats = df[cols].describe()

    return stats
