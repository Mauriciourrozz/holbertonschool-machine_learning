#!/usr/bin/env python3
"""
This file contains the function matrix_shape
"""


def matrix_shape(matrix):
    """
    Function that return the shape of a matrix
    """
    forma = []
    while isinstance(matrix, list):
        forma.append(len(matrix))
        matrix = matrix[0]
    return forma
