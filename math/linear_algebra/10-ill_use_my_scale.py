#!/usr/bin/env python3
"""
This funcion calculates the shape of a numpy.ndarray
"""


def np_shape(matrix):
    """
    This funcion calculates the shape of a numpy.ndarray
    """
    return (len(matrix),) + (
        isinstance(matrix[0], list) and np_shape(matrix[0]) or ()
    )
