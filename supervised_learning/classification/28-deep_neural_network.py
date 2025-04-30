#!/usr/bin/env python3
"""
deep_neural_network module

This module defines the DeepNeuralNetwork class,
which builds a deep neural network for binary
classification using He et al. initialization.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    DeepNeuralNetwork defines a deep neural network
    for binary classification.

    Attributes:
        L (int): number of layers in the network.
        cache (dict): intermediary values (activations).
        weights (dict): weights and biases of each layers.
    """
    def __init__(self, nx, layers, activation='sig'):
        """
        Initialize a DeepNeuralNetwork
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")

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
        #  tipo de función de activación utilizada en las capas ocultas
        self.__activation = activation

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

            # softmax en la última capa para obtener probabilidades
            if i == self.L:
                exp_zl = np.exp(zl - np.max(zl, axis=0, keepdims=True))
                Al = exp_zl / np.sum(exp_zl, axis=0, keepdims=True)
            # Sigmoidea o tanh para las capas intermedias
            else:
                if self.__activation == 'sig':
                    Al = 1 / (1 + np.exp(-zl))  # Sigmoidea
                elif self.__activation == 'tanh':
                    Al = np.tanh(zl)  # Tanh

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

        costo = -np.sum(Y * np.log(A)) / m
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
        A, _ = self.forward_prop(X)

        costo = self.cost(Y, A)

        prediccion = np.argmax(A, axis=0)
        one_hot_prediccion = np.zeros_like(A)
        one_hot_prediccion[prediccion, np.arange(A.shape[1])] = 1
        return one_hot_prediccion, costo

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
            # Aquí se usa la derivada según la función de activación seleccionada
            if self.__activation == 'sig':  # Si la activación es sigmoidea
                dZ = np.dot(W.T, dZ) * (A_prev * (1 - A_prev))  # Derivada de la sigmoidea
            elif self.__activation == 'tanh':  # Si la activación es tanh
                dZ = np.dot(W.T, dZ) * (1 - np.tanh(A_prev) ** 2)  # Derivada de tanh

            # actualizamos pesos y sesgos
            self.weights['W' + str(i)] -= alpha * dW
            self.weights['b' + str(i)] -= alpha * db


    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
            Trains the deep neural network using gradient descent,
            with optional console output and plotting of
            the training cost over time.

            Parameters
            ----------
            X : numpy.ndarray of shape (nx, m)
                Input data, where nx is the number of features and m is
                the number of examples.
            Y : numpy.ndarray of shape (1, m)
                Correct labels (0 or 1) for each example.
            iterations : int, optional
                Number of training iterations. Must be a positive integer.
            alpha : float, optional
                Learning rate. Must be a positive float.
            verbose : bool, optional
                If True, prints the cost after every `step` iterations,
                including iteration 0 and the final iteration.
            graph : bool, optional
                If True, plots the cost versus iterations after
                training completes. The plot uses a blue line,
                labels the x-axis "iteration", the y-axis "cost",
                and titles the chart "Training Cost".
            step : int, optional
                Interval (in iterations) at which to print and/or
                plot the cost.
                Only validated if `verbose` or `graph` is True.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        # if not isinstance(step, int):
        #     raise TypeError("step must be an integer")
        # if step <= 0 or step > iterations:
        #     raise ValueError("step must be positive and <= iterations")

        iteraciones = []
        costos = []
        a0, _ = self.forward_prop(X)
        costo0 = self.cost(Y, a0)
        iteraciones.append(0)
        costos.append(costo0)
        if verbose:
            print(f"Cost after 0 iterations: {costo0}")

        for i in range(1, iterations + 1):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if (i % step == 0) or (i == iterations):
                cost_i = self.cost(Y, A)
                iteraciones.append(i)
                costos.append(cost_i)

                if verbose:
                    print(f"Cost after {i} iterations: {cost_i}")

        if graph:
            plt.plot(iteraciones, costos)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """
        Serialize this object to a file using pickle.

        If the given filename does not end with “.pkl”, the extension will
        be appended automatically.
        """
        if not filename.endswith(".pkl"):
            filename = filename + ".pkl"
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """
        Load and return a pickled object from a file.

        If an error occurs (e.g., file not found or unpickling fails),
        this method returns None.
        """
        if not filename.endswith(".pkl"):
            filename = filename + ".pkl"

        try:
            with open(filename, "+rb") as file:
                return pickle.load(file)
        except Exception:
            return None
