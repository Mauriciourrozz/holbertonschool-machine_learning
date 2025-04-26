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
        # forward propagation para obtener activaciones
        _, a = self.forward_prop(X)

        # cálculo del costo con entropía cruzada logística
        costo = self.cost(Y, a)

        # umbral para generar predicciones binarias
        prediccion = np.where(a >= 0.5, 1, 0)
        return prediccion, costo

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Perform an update of the neural network parameters using
        the gradient descent algorithm.

        Arguments:
            X (numpy.ndarray): An input array of shape (nx, m),
            where nx is the number of input features
            and m is the number of training examples.
            Y(numpy.ndarray) – An array of shape (1, m) containing the
            true labelsfor the training examples.
            A1 (numpy.ndarray): Hidden layer activations,
            of shape (n_hidden, m), computed during forward propagation.
            A2 (numpy.ndarray): Output layer activations, of shape (1, m),
            computed during forward propagation.
            alpha (float, optional): The learning rate.
            The default value is 0.05.

        Updates the parameters (weights and biases) of the neural
        network using gradient descent.
        """
        # Número de ejemplos
        m = X.shape[1]

        # Gradientes de la capa de salida
        dZ2 = A2 - Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis=1, keepdims=True) / m

        # Gradientes de la capa oculta
        dZ1 = np.dot(self.__W2.T, dZ2) * A1 * (1 - A1)
        dW1 = np.dot(dZ1, X.T) / m
        db1 = np.sum(dZ1, axis=1, keepdims=True) / m

        # Actualización de los parámetros
        self.__W1 -= alpha * dW1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dW2
        self.__b2 -= alpha * db2
