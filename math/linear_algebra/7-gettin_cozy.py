#!/usr/bin/env python3
"""
This function concatenates two matrices along a specified axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenate two matrices along a specified axis.
    """
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2

    elif axis == 1:
        new_mat = []
        if len(mat1) != len(mat2):
            return None
        for i in range(len(mat1)):
            new_mat.append(mat1[i] + mat2[i])
        return new_mat

    return None