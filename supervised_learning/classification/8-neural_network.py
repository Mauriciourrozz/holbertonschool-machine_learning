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
        self.W1 = np.random.normal(0, 1, (nodes, nx))

        # sesgo para la capa oculta
        self.b1 = np.zeros((nodes, 1))

        # salida activada para la capa oculta
        self.A1 = 0

        # vector de pesas para la neurona de salida
        self.W2 = np.random.normal(0, 1, (1, nodes))

        # sesgo para la neurona de salida
        self.b2 = 0

        # salida activada para la neurona de salida (predicción)
        self.A2 = 0
