#!/usr/bin/env python3
"""
summation_i_squared(n)
"""
import numpy as np


def summation_i_squared(n):
    """
    Function that calculates sigma
    """
    if not isinstance(n, int) or n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
