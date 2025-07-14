#!/usr/bin/env python3
"""
1-minor.py
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
    if matrix == [[]]:
        return 1

    if matrix == []:
        return 1

    if not isinstance(matrix, list) or not all(isinstance(i, list)
                                               for i in matrix):
        raise TypeError("matrix must be a list of lists")

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


def minor(matrix):
    """
    Calculates the matrix of minors of a square matrix.

    Args:
    matrix (list of lists): Square matrix of numbers.

    Raises:
    TypeError: If matrix is not a list of lists or is empty.
    ValueError: If matrix is not square.

    Returns:
    list of lists: Matrix of minors.
    """
    if not matrix:
        raise TypeError("matrix must be a list of lists")
    if not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    if not matrix or matrix == [] or matrix == [[]] or len(matrix) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)
    menores = []

    for i in range(n):
        fila_menores = []
        for j in range(n):
            # Construir submatriz excluyendo fila i y columna j
            submatriz = [
                [matrix[x][y] for y in range(n) if y != j]
                for x in range(n) if x != i
            ]
            # Calcular determinante de la submatriz
            menor = determinant(submatriz)
            fila_menores.append(menor)
        menores.append(fila_menores)

    return menores
