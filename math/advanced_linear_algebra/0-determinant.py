#!/usr/bin/env python3
"""
0-determinant.py
"""


def determinant(matrix):
    """
Calculates the determinant of a square matrix represented as a list of lists.

Args:
matrix (list of list): Square matrix of numbers (int or float).
The special case [[]] represents the 0x0 matrix, whose determinant is 1.

Raises:
TypeError: If `matrix` is not a list of lists.
ValueError: If `matrix` is not a square matrix.
TypeError: If `matrix` is an empty list (no rows).

Returns:
int or float: The calculated determinant of the matrix.
    """
    # si es una lista vacia dentro de una lista el determinante es 1
    if matrix == [[]]:
        return 1

    # compruebo que matrix sea una lista de listas
    if not isinstance(matrix, list) or not all(isinstance(
            i, list) for i in matrix) or matrix == []:
        raise TypeError("matrix must be a list of lists")

    # compruebo que sea una matrix cuadrada
    if not all(len(fila) == len(matrix) for fila in matrix):
        raise ValueError("matrix must be a square matrix")

    filas = len(matrix)

    if filas == 1:
        return matrix[0][0]
    elif filas == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        det = 0
        for j in range(filas):
            submatriz = []
            for fila in range(1, filas):
                sub_fila = []
                for col in range(filas):
                    if col != j:
                        sub_fila.append(matrix[fila][col])
                submatriz.append(sub_fila)

            cofactor = matrix[0][j] * determinant(submatriz)
            if j % 2 == 0:
                det += cofactor
            else:
                det -= cofactor
        return det
