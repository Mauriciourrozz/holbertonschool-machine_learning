#!/usr/bin/env python3
"""
deep_neural_network module

This module defines the DeepNeuralNetwork class,
which builds a deep neural network for binary
classification using He et al. initialization.
"""
import numpy as np


class DeepNeuralNetwork:
    """
    DeepNeuralNetwork defines a deep neural network
    for binary classification.

    Attributes:
        L (int): number of layers in the network.
        cache (dict): intermediary values (activations).
        weights (dict): weights and biases of each layers.
    """
    def __init__(self, nx, layers):
        """
        Initialize a DeepNeuralNetwork
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        # número de características de entrada
        self.nx = nx
        # lista que representa el número de nodos en cada capa de la red
        self.layers = layers
        # número de capas en la red neuronal
        self.__L = len(layers)
        # diccionario para mantener todos los valores intermediarios de la red
        self.__cache = {}
        # diccionario para contener todos los pesos y sesgados de la red
        self.__weights = {}

        for i in range(self.L):
            # Comprobacion de cada elemento de layers es entero positivo
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

            if i == 0:
                self.weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])

    @property
    def L(self):
        """
        Retrieve the number of layers in the deep neural network.
        """
        return self.__L

    @property
    def cache(self):
        """
        Access the cache of intermediate values (activations)
        during forward propagation.
        """
        return self.__cache

    @property
    def weights(self):
        """
        Get the dictionary of network parameters (weights and biases).
        """
        return self.__weights
