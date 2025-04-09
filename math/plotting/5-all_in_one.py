#!/usr/bin/env python3
"""
All in one
"""


import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    Plot all 5 previous graphs in one figure
    """

    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    fig, axes = plt.subplots(3, 2, figsize=(8, 6))

    # Primera grafica
    axes[0, 0].plot(y0, color='r')
    axes[0, 0].set_xlim(0, len(y0) - 1)

    # Segunda grafica
    axes[0, 1].scatter(x1, y1, color="m")
    axes[0, 1].set_title("Men's Height vs Weight", fontsize="x-small")
    axes[0, 1].set_ylabel("Weight (lbs)", fontsize="x-small")
    axes[0, 1].set_xlabel("Height (in)", fontsize="x-small")

    # Tercera grafica
    axes[1, 0].plot(x2, y2)
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlim(0, 28650)
    axes[1, 0].set_title("Exponential Decay of C-14", fontsize="x-small")
    axes[1, 0].set_xlabel("Time (years)", fontsize="x-small")
    axes[1, 0].set_ylabel("Fraction Remaining", fontsize="x-small")

    # Cuarta grafica
    axes[1, 1].set_xlabel("Time (years)", fontsize="x-small")
    axes[1, 1].set_ylabel("Fraction Remaining", fontsize="x-small")
    axes[1, 1].set_title("Exponential Decay of Radioactive Elements",
                         fontsize="x-small")
    axes[1, 1].plot(x3, y31, linestyle='--', color='r', label="C-14")
    axes[1, 1].plot(x3, y32, linestyle='-', color='green', label="Ra-226")
    axes[1, 1].set_xlim(0, 20000)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend(loc="upper right")

    # Quinta grafica
    fig.delaxes(axes[2, 0])
    fig.delaxes(axes[2, 1])
    ax_hist = fig.add_subplot(3, 2, (5, 6))
    ax_hist.set_xlabel("Grades", fontsize="x-small")
    ax_hist.set_ylabel("Number of Students", fontsize="x-small")
    ax_hist.set_title("Project A", fontsize="x-small")
    ax_hist.set_xlim(0, 100)
    ax_hist.set_xticks(np.arange(0, 110, 10))
    ax_hist.set_ylim(0, 30)
    ax_hist.hist(student_grades, bins=range(0, 110, 10), edgecolor='black')

    plt.tight_layout()
    plt.suptitle("All in One")
    plt.show()
