#!/usr/bin/env python3
def poly_integral(poly, C=0):
    if not isinstance(poly, list) or len(poly) == 0 or not isinstance(C, int):
        return None
    integral = [C]
    for i in range(len(poly)):
        value = poly[i] / (i + 1)
        if value == int(value):
            value = int(value)
        integral.append(value)
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral
