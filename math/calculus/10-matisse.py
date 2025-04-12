#!/usr/bin/env python3
"""
10-matisse.py
"""


def poly_derivative(poly):
    """
    Function that calculates the derivative of a polynomial
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if len(poly) == 1:
        return [0]
    derivada = []
    for i in range(1, len(poly)):
        derivada.append(i * poly[i])
    return derivada
