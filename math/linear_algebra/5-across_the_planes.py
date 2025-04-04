#!/usr/bin/env python3
"""
Function that add 2D matrix
"""


def add_matrices2D(mat1, mat2):
    """
    Function that add 2D matrix
    """
    if len(mat1[0]) != len(mat2[0]):
        return None

    suma = []
    for i in range(len(mat1)):
        fila = []
        for j in range(len(mat1[0])):
            fila.append(mat1[i][j] + mat2[i][j])
        suma.append(fila)

    return suma
