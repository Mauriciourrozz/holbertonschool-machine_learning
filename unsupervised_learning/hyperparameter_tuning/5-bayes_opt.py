#!/usr/bin/env python3
"""
5-bayes_opt.py
"""
import numpy as np
from scipy.stats import norm
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

    def acquisition(self):
        """
        Calculates the next best sample point using Expected Improvement (EI).

        Returns:
            X_next (numpy.ndarray): Next point to sample, shape (1,)
            EI (numpy.ndarray): Expected Improvement for each candidate in X_s,
            shape (ac_samples,)
        """
        # Mejor valor hasta ahora
        if self.minimize:
            Y_best = np.min(self.gp.Y)
        else:
            Y_best = np.max(self.gp.Y)

        # Predecir media y desviación estándar en los puntos candidatos
        mu, sigma = self.gp.predict(self.X_s)
        # asegurar que sigma tenga forma (ac_samples,)
        sigma = sigma.reshape(-1)

        # Inicializar EI con ceros
        EI = np.zeros_like(mu)

        # Evitar división por cero en sigma
        nonzero_sigma = sigma > 0

        # Calcular Z y EI solo para sigma > 0
        if self.minimize:
            Z = (Y_best - mu[nonzero_sigma] - self.xsi) / sigma[nonzero_sigma]
            EI[nonzero_sigma] = (Y_best - mu[nonzero_sigma] - self.xsi
                                 ) * norm.cdf(Z) + sigma[
                                     nonzero_sigma] * norm.pdf(Z)
        else:
            Z = (mu[nonzero_sigma] - Y_best - self.xsi) / sigma[nonzero_sigma]
            EI[nonzero_sigma] = (mu[nonzero_sigma] - Y_best - self.xsi
                                 ) * norm.cdf(Z) + sigma[
                                     nonzero_sigma] * norm.pdf(Z)

        # Elegir el siguiente punto: X_s con EI máximo
        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI

    def optimize(self, iterations=100):
        """
        Performs Bayesian Optimization to find the optimum of the black-box
        function.

        Args:
            iterations (int): Maximum number of iterations to perform.

        Returns:
            X_opt (numpy.ndarray): Optimal point found, shape (1,)
            Y_opt (numpy.ndarray): Function value at X_opt, shape (1,)
        """
        for i in range(iterations):
            # Calcular el siguiente punto candidato usando la función de
            # adquisición
            X_next, _ = self.acquisition()

            # Detener si el punto ya fue evaluado
            if np.any(np.all(self.gp.X == X_next, axis=1)):
                break

            # Evaluar la función caja negra en el nuevo punto
            Y_next = self.f(X_next).reshape(1, 1)

            # Actualizar el Gaussian Process con el nuevo dato
            self.gp.update(X_next.reshape(1, 1), Y_next)

        # Obtener el mejor valor observado hasta ahora
        if self.minimize:
            idx_opt = np.argmin(self.gp.Y)
        else:
            idx_opt = np.argmax(self.gp.Y)

        X_opt = self.gp.X[idx_opt]
        Y_opt = self.gp.Y[idx_opt]

        return X_opt, Y_opt
