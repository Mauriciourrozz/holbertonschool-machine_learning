#!/usr/bin/env python3
"""
Function that performs matrix multiplication
"""


def mat_mul(mat1, mat2):
    """
    Function that performs matrix multiplication
    """
    if len(mat1[0]) != len(mat2):
        return None

    resultado = []
    for i in range(len(mat1)):
        fila_resultado = []
        for j in range(len(mat2[0])):
            suma = 0
            for k in range(len(mat1[0])):
                suma += mat1[i][k] * mat2[k][j]
            fila_resultado.append(suma)
        resultado.append(fila_resultado)
    return resultado
