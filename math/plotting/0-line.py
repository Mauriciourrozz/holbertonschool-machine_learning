#!/usr/bin/env python3
"""
Line Graph
"""


import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    This function plot y as a line graph
    """
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, color="r")
    plt.xlim(0, len(y) -1)
    plt.show()
