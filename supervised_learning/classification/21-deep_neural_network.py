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

    def forward_prop(self, X):
        """
        Perform forward propagation through the deep neural network.

        Args:
            X (np.ndarray): Input data of shape (nx, m), where
                nx is the number of input features and m is
                the number of examples.

        Returns:
            tuple:
                - AL (np.ndarray): Activation of the last layer
                (shape matches the output layer size and m).
                - cache (dict): Dictionary containing all activations:
                    keys "A0", "A1", …, "A{L}" with their corresponding arrays.
        """
        # aqui guardo x en el diccionario cache con la clave A0
        self.__cache["A0"] = X
        for i in range(1, self.L + 1):
            # saca de self.__weights la matriz de pesos Wl y
            # el vector de sesgo bl para la capa i
            Wl = self.__weights[f"W{i}"]
            bl = self.__weights[f"b{i}"]

            # Busca en self.__cache la activación de la capa anterior,
            # bajo "A{i-1}"
            a_anterior = self.__cache[f"A{i - 1}"]

            # Calcula la suma ponderada
            zl = Wl @ a_anterior + bl

            # Aplica la función sigmoidea para obtener la nueva activación Al
            Al = 1 / (1 + np.exp(-zl))

            # Mete Al en self.__cache con la clave "A{i}"
            self.__cache[f"A{i}"] = Al

        # Devuelve la última activación y el diccionario cache
        return Al, self.__cache

    def cost(self, Y, A):
        """
        Calculates the cost of the model using binary cross-entropy loss.

        Parameters
        ----------
        Y : numpy.ndarray of shape (1, m)
            True labels vector (0 or 1).
        A : numpy.ndarray of shape (1, m)
            Predicted probabilities vector.
        """
        m = Y.shape[1]

        costo = np.sum(-(Y * np.log(A) + (1-Y) * np.log(1.0000001-A))) / m
        return costo

    def evaluate(self, X, Y):
        """
        Evaluates the predictions of the neural network.

        Args:
        X (numpy.ndarray): A matrix of shape (nx, m) containing the input data,
        where nx is the number of input features and m is
        the number of examples
        Y (numpy.ndarray): A matrix of shape (1, m)
        containing the true labels for the input data.
        """
        a, _ = self.forward_prop(X)

        costo = self.cost(Y, a)

        prediccion = np.where(a >= 0.5, 1, 0)
        return prediccion, costo

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        calculates one pass of gradient descent on the neural network
        Y numpy.ndarray with shape (1, m)
            that contains the correct labels for the input data
        cache is a dictionary containing all the
            intermediary values of the network
        alpha is the learning rate
        """
        # numero de ejemplos
        m = Y.shape[1]

        # activación de la última capa
        A = cache['A' + str(self.L)]

        # gradiente de la capa de salida
        dZ = A - Y

        # recorre de atras hacia adelante
        for i in range(self.L, 0, -1):
            # activación de la capa anterior
            A_prev = cache['A' + str(i - 1)]

            # pesos de la capa actual
            W = self.weights['W' + str(i)]

            # gradiente de los pesos de la capa actual
            dW = np.dot(dZ, A_prev.T) / m

            # gradiente de los sesgos de la capa actual
            db = np.sum(dZ, axis=1, keepdims=True) / m

            # gradiente de la capa anterior
            dZ = np.dot(W.T, dZ) * (A_prev * (1 - A_prev))

            # actualizamos pesos y sesgos
            self.weights['W' + str(i)] -= alpha * dW
            self.weights['b' + str(i)] -= alpha * db
