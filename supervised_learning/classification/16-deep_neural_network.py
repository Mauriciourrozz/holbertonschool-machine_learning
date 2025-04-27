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
        weights (dict): weights and biases of each layer.
    """
    def __init__(self, nx, layer):
        """
        Initialize a DeepNeuralNetwork
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        # número de características de entrada
        self.nx = nx
        # lista que representa el número de nodos en cada capa de la red
        self.layer = layer
        # número de capas en la red neuronal
        self.L = len(layer)
        # diccionario para mantener todos los valores intermediarios de la red
        self.cache = {}
        # diccionario para contener todos los pesos y sesgados de la red
        self.weights = {}
        for idx, nodes in enumerate(layer, start=1):
            # comprobacion de que cada elemento de layer sea >0 y no sea vacia
            if not isinstance(nodes, int) or nodes < 1:
                raise TypeError("layer must be a list of positive integers")

            # Si estoy en la primera capa (idx == 1), las entradas son nx
            # Si estoy en la capa 2, la capa anterior es la 1,
            # que en la lista está en layer[0]. Y 0 es 2 - 2
            # de ahi layer[idx-2]:
            # Para idx = 2, idx-2 = 0: layer[0]: nodos de la capa 1
            # Para idx = 3, idx-2 = 1: layer[1]: nodos de la capa 2
            n_prev = nx if idx == 1 else layer[idx - 2]

            # se inicializan los pesos usando He et al
            self.weights[f"W{idx}"] = (
                np.random.randn(nodes, n_prev) * np.sqrt(2 / n_prev)
            )

            # inicializa sesgos en 0 y se guarda como pide la letra
            self.weights[f"b{idx}"] = np.zeros((nodes, 1))
