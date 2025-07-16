#!/usr/bin/env python3
"""
4-inverse.py
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


def adjugate(matrix):
    """
    Calculates the adjugate (adjoint) of a square matrix.

    The adjugate of a matrix is the transpose of its cofactor matrix.

    Args:
        matrix (list of lists): A non-empty square matrix represented as a
        list of lists of numbers.

    Raises:
        TypeError: If `matrix` is not a list of lists or is empty ([] or [[]]).
        ValueError: If `matrix` is not a non-empty square matrix.

    Returns:
        list of lists: The adjugate matrix of the input matrix.
    """
    if not isinstance(matrix, list) or not all(
            isinstance(fila, list) for fila in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == []:
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    if any(len(fila) != len(matrix) for fila in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    cof = cofactor(matrix)
    n = len(cof)
    adj = [[cof[j][i] for j in range(n)] for i in range(n)]
    return adj


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a given square matrix.

    Args:
        matrix (list of lists): A non-empty square matrix represented as a
        list of lists of numbers.

    Raises:
        TypeError: If `matrix` is not a list of lists or is empty ([] or [[]]).
        ValueError: If `matrix` is not a square matrix.

    Returns:
        list of lists: The cofactor matrix of the input matrix.
    """
    if not isinstance(matrix, list) or not all(
            isinstance(fila, list) for fila in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == []:
        raise TypeError("matrix must be a list of lists")

    if matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    if any(len(fila) != len(matrix) for fila in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    n = len(matrix)

    if n == 1:
        return [[1]]

    cof_matrix = []
    for i in range(n):
        cof_row = []
        for j in range(n):
            submatriz = [
                [matrix[x][y] for y in range(n) if y != j]
                for x in range(n) if x != i
            ]
            menor = determinant(submatriz)
            cofactor_value = ((-1) ** (i + j)) * menor
            cof_row.append(cofactor_value)
        cof_matrix.append(cof_row)

    return cof_matrix



def inverse(matrix):
    """
    Calculates the inverse of a square matrix using its adjugate and determinant.

    Args:
        matrix (list of lists): A non-empty square matrix.

    Raises:
        TypeError: If `matrix` is not a list of lists.
        ValueError: If `matrix` is not a non-empty square matrix.

    Returns:
        list of lists: The inverse of the input matrix, or None if the matrix is singular.
    """
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")

    if matrix == [] or matrix == [[]]:
        raise ValueError("matrix must be a non-empty square matrix")

    if any(len(row) != len(matrix) for row in matrix):
        raise ValueError("matrix must be a non-empty square matrix")

    det = determinant(matrix)
    if det == 0:
        return None

    adj = adjugate(matrix)
    n = len(matrix)
    inv = [[adj[i][j] / det for j in range(n)] for i in range(n)]

    return inv
