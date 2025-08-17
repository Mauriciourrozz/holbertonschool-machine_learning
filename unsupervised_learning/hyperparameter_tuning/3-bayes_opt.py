#!/usr/bin/env python3
"""
3-bayes_opt.py
"""
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian Optimization on a noiseless 1D Gaussian Process.

    Attributes:
        f (callable): The black-box function to be optimized.
        gp (GaussianProcess): Instance of the GaussianProcess class modeling f.
        X_s (numpy.ndarray): Acquisition sample points of shape (ac_samples, 1)
                             evenly spaced between the bounds (min, max).
        xsi (float): Exploration-exploitation factor for the acquisition
            function.
        minimize (bool): If True, optimization seeks the minimum of f;
                         if False, seeks the maximum.
    """

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """
        Initializes the BayesianOptimization object.

        Args:
            f (callable): The black-box function to optimize.
            X_init (numpy.ndarray): Initial input samples, shape (t, 1).
            Y_init (numpy.ndarray): Initial output values for X_init,
                shape (t, 1).
            bounds (tuple): (min, max) limits of the search space.
            ac_samples (int): Number of acquisition sample points.
            l (float, optional): Length scale parameter for the kernel.
                Default is 1.
            sigma_f (float, optional): Output scale for the kernel.
                Default is 1.
            xsi (float, optional): Exploration-exploitation factor.
                Default is 0.01.
            minimize (bool, optional): True for minimization,
                False for maximization. Default is True.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)

        # X_s: puntos de adquisición distribuidos uniformemente
        # entre bounds[0] y bounds[1].
        # bounds = (min, max) define los límites del espacio de búsqueda.
        # np.linspace genera ac_samples valores espaciados en ese rango.
        # reshape los convierte en una matriz columna de forma (ac_samples, 1).
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(
            ac_samples, 1)
        self.xsi = xsi
        self.minimize = minimize
