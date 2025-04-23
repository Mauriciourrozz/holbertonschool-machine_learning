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

    def forward_prop(self, X):  # PROPAGACION HACIA ADELANTE
        """
        Perform forward propagation for the neuron.
        """
        # Z = W·X + b: producto punto entre pesos y
        # entradas para obtener la activación lineal
        z = np.dot(self.__W, X) + self.__b

        # Aplica la función sigmoide a cada elemento de Z.
        # Convierte esos valores en números entre 0 y 1, dando la
        # “activación” de la neurona para cada ejemplo
        activacion_sigmoidea = 1 / (1 + np.exp(-z))

        # Guarda el resultado de la activación en el atributo privado __A
        self.__A = activacion_sigmoidea

        return self.__A

    def cost(self, Y, A):  # Calcula el costo de modelo con regresión logística
        """
        Calculate the cost of the model using logistic regression.

        The cost function measures the distance between the predictions
        of the neuron (A) and the real labels (Y), penalizing
        wrong predictions more strongly.

        Args:
        Y(np.ndarray): Real label row vector, shape(1, m),
                        values ​​0 or 1.
        A (np.ndarray): Row vector of activations (predictions),
                        shape(1, m), values ​​between 0 and 1.
        """
        # se pone Y.shape[1] para en la tupla conseguir el puesto 1, osea m.
        # Recordar que Y es un ndarray de forma (1, m)
        m = Y.shape[1]

        costo = np.sum(-(Y * np.log(A) + (1-Y) * np.log(1.0000001-A))) / m
        return costo

    def evaluate(self, X, Y):  # Evalúa las predicciones de la neurona
        """
        Evaluate the neuron's predictions and calculate its cost.

        Perform forward propagation on data X, compare
        activations with AND labels to generate predictions
        binary, and measures the mean error using logistic cross entropy.

        Args:
        X (np.ndarray): Array of inputs of form (n_x, m), where n_x
        is the number of features and m is the numberof examples.

        Y(np.ndarray): Row vector of true labels of shape (1, m),
        with values ​​0 or 1.
        """
        # forward propagation para obtener activaciones
        a = self.forward_prop(X)

        # cálculo del costo con entropía cruzada logística
        costo = self.cost(Y, a)

        # umbral para generar predicciones binarias
        prediccion = np.where(a > 0.5, 1, 0)
        return prediccion, costo
