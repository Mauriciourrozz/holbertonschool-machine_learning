#!/usr/bin/env python3
"""
Stacking Bars
"""


import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Function that plot a stacked bar graph
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    personas = ["Farrah", "Fred", "Felicia"]
    frutas = ["apples", "bananas", "oranges", "peaches"]
    colores = ["red", "yellow", "#ff8000", "#ffe5b4"]

    base = np.zeros(3)
    for i in range(4):
        plt.bar(range(3), fruit[i], bottom=base, width=0.5,
                color=colores[i], label=frutas[i])
        base = base + fruit[i]

    plt.xticks(range(3), personas)
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))
    plt.ylabel("Quantity of Fruit")
    plt.title("Number of Fruit per Person")
    plt.legend()

    plt.show()
