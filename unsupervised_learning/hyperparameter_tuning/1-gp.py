#!/usr/bin/env python3
"""
1-gp.py
"""
import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian Process.

    Attributes:
        X (numpy.ndarray): Sampled inputs of shape (t, 1).
        Y (numpy.ndarray): Outputs of the black-box function for each
            input in X.
        l (float): Length scale parameter of the RBF kernel.
        sigma_f (float): Standard deviation of the function outputs.
        K (numpy.ndarray): Covariance kernel matrix of shape (t, t).
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initializes the Gaussian Process with initial samples.

        Args:
            X_init (numpy.ndarray): Initial sampled inputs of shape (t, 1).
            Y_init (numpy.ndarray): Corresponding outputs of shape (t, 1).
            l (float, optional): Length scale parameter of the kernel.
                Defaults to 1.
            sigma_f (float, optional): Standard deviation of the outputs.
                Defaults to 1.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices using
            the RBF kernel.

        Args:
            X1 (numpy.ndarray): First set of points, shape (m, 1).
            X2 (numpy.ndarray): Second set of points, shape (n, 1).

        Returns:
            numpy.ndarray: Covariance matrix of shape (m, n) where each element
                           represents the RBF covariance between points from X1
                           and X2.
        """
        # Calcula la diferencia entre cada par de puntos y lo eleva al cuadrado
        distance = (X1 - X2.T) ** 2

        # Aplica la formula RBF
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * distance)

    def predict(self, X_s):
        """
        Predicts the mean and variance of points in a Gaussian process.

        Args:
            X_s (numpy.ndarray): Points to predict, shape (s, 1)

        Returns:
            mu (numpy.ndarray): Mean for each point in X_s, shape (s,)
            sigma (numpy.ndarray): Variance for each point in X_s, shape (s,)
        """
        # Calcular la covarianza entre los puntos conocidos y los nuevos puntos
        K_s = self.kernel(self.X, X_s)

        # Calcular la covarianza de los nuevos puntos consigo mismos
        K_ss = self.kernel(X_s, X_s)

        # Calcular la inversa de la matriz de covarianza de los puntos
        # conocidos
        K_inv = np.linalg.inv(self.K)

        # Calcular la media de las predicciones
        mu = K_s.T @ K_inv @ self.Y

        # Calcular la varianza de las predicciones
        sigma = np.diag(K_ss - K_s.T @ K_inv @ K_s)

        # mu a forma (s,)
        mu = mu.ravel()

        return mu, sigma
