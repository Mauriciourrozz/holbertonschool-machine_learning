#!/usr/bin/env python3
"""
This file contains a function matrix transpose
"""


def matrix_transpose(matrix):
    filas = len(matrix)
    columnas = len(matrix[0])
    transpuesta = []

    for j in range(columnas):
        nueva_fila = []
        for i in range(filas):
            nueva_fila.append(matrix[i][j])
        transpuesta.append(nueva_fila)

    return transpuesta
