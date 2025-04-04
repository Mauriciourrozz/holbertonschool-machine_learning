#!/usr/bin/env python3
"""
This function adds 2 matrixs
"""


def add_arrays(arr1, arr2):
    """
    This function adds 2 matrixs
    """
    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]
