#!/usr/bin/env python3
"""
neural_network module

This module defines the NeuralNetwork class for
building and training a neural network
with one hidden layer and one output neuron.
"""
import numpy as np


class NeuralNetwork:
    """
    Represents a neural network with one hidden layer.

    Attributes:
        nx (int): number of input features.
        nodes (int): number of nodes in the hidden layer.
        W1 (np.ndarray): weight matrix for the hidden
        layer of shape (nodes, nx).
        b1 (float): bias for the hidden layer.
        A1 (float): activated output of the hidden layer.
        W2 (np.ndarray): weight matrix for the output
        neuron of shape (nodes, nx).
        b2 (float): bias for the output neuron.
        A2 (float): activated output of the output neuron.
    """
    def __init__(self, nx, nodes):
        """
        Initializes the neural network.

        Args:
            nx (int): number of input features.
            nodes (int): number of nodes in the hidden layer.

        Raises:
            TypeError: if nx or nodes is not an integer.
            ValueError: if nx or nodes is less than 1.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # número de características de entrada
        self.nx = nx

        # número de nodos que se encuentran en la capa oculta
        self.nodes = nodes

        # vector de pesas para la capa oculta
        self.__W1 = np.random.normal(0, 1, (nodes, nx))

        # sesgo para la capa oculta
        self.__b1 = np.zeros((nodes, 1))

        # salida activada para la capa oculta
        self.__A1 = 0

        # vector de pesas para la neurona de salida
        self.__W2 = np.random.normal(0, 1, (1, nodes))

        # sesgo para la neurona de salida
        self.__b2 = 0

        # salida activada para la neurona de salida (predicción)
        self.__A2 = 0

    @property
    def W1(self):
        """Weight matrix for the hidden layer."""
        return self.__W1

    @property
    def b1(self):
        """Bias vector for the hidden layer."""
        return self.__b1

    @property
    def A1(self):
        """Activated output for the hidden layer."""
        return self.__A1

    @property
    def W2(self):
        """Weight matrix for the output neuron."""
        return self.__W2

    @property
    def b2(self):
        """Bias for the output neuron."""
        return self.__b2

    @property
    def A2(self):
        """Activated output for the output neuron."""
        return self.__A2

    def forward_prop(self, X):
        """
        Perform forward propagation through the network.

        Parameters
        ----------
        X : numpy.ndarray of shape (nx, m)
            Input data, where nx is the number of input features and
            m is the number of examples.

        Returns
        -------
        A1 : numpy.ndarray of shape (nodes, m)
            Activations of the hidden layer.
        A2 : numpy.ndarray of shape (1, m)
            Activations of the output layer (predicted probabilities).
        """
        # Calcula la combinación lineal Z1 de la capa oculta:
        # producto de pesos y entradas más sesgo
        z1 = (self.__W1 @ X) + self.__b1

        # Aplica la función sigmoide elemento a elemento a Z1
        sigm1 = 1 / (1 + np.exp(-z1))

        # Guarda las activaciones de la capa oculta en el atributo privado __A1
        self.__A1 = sigm1

        # Calcula la combinación lineal Z2 de la capa de salida:
        # pesos de salida por activación oculta más sesgo
        z2 = (self.__W2 @ self.__A1) + self.__b2

        # Aplica la sigmoide a Z2 para obtener las probabilidades de salida
        sigm2 = 1 / (1 + np.exp(-z2))

        # Guarda las activaciones de la capa de salida en __A2
        self.__A2 = sigm2

        # Devuelve las activaciones de la capa oculta y de salida
        return self.__A1, self.__A2
