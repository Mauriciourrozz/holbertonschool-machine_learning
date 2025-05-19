#!/usr/bin/env python3
"""
7-early_stopping.py
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines whether to stop training early.

    Returns:
    Tuple (bool, int): whether to stop training, and the new count.
    """
    improvement = opt_cost - cost

    if improvement > threshold:
        reset_count = 0
        stop_training = False
    else:
        reset_count = count + 1
        stop_training = reset_count >= patience

    return stop_training, reset_count
