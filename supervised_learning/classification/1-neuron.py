#!/usr/bin/env python3
"""Module that defines a Neuron for binary classification."""


import numpy as np


class Neuron:
    """Represents a single neuron performing binary classification.

    Attributes:
        nx (int): Number of input features.
        W (numpy.ndarray): Weights vector for the neuron (shape: 1 x nx).
        b (float): Bias for the neuron initialized to 0.
        A (float): Activated output of the neuron (prediction),
        initialized to 0.
    """
    def __init__(self, nx):
        # nx es el número de características de entrada a la neurona
        """Initialize a Neuron instance.

        Args:
            nx (int): Number of input features.

        Raises:
            TypeError: If nx is not an integer.
            ValueError: If nx is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.nx = nx  # pesos de la neurona
        self.__W = np.random.randn(1, nx)
        self.__b = 0  # Sesgo (bias)
        self.__A = 0  # salida activada

    @property
    def W(self):
        """
        Getter for the neuron's weight vector
        """
        return self.__W

    @property
    def b(self):
        """
        Getter for the neuron's bias
        """
        return self.__b

    @property
    def A(self):
        """
        Getter for the neuron's activated output
        """
        return self.__A
