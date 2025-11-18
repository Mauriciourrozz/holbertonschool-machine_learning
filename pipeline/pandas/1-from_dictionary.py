#!/usr/bin/env python3
"""
Write a python script that creates a pd.DataFrame from a dictionary
"""
import pandas as pd

# Crea el diccionario y las columnas
data = {
    "First": [0.0, 0.5, 1.0, 1.5],
    "Second": ["one", "two", "three", "four"]
}

# Crea la tabla con los indices A, B, C y D
df = pd.DataFrame(data, index=["A", "B", "C", "D"])
