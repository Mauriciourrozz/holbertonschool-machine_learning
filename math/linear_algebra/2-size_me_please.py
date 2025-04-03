#!/usr/bin/env python3
def matrix_shape(matrix):
    forma = []
    while isinstance(matrix, list):
        forma.append(len(matrix))
        matrix = matrix[0]
    return forma
