#!/usr/bin/env python3
"""Module that defines a Neuron for binary classification."""


import numpy as np


class Neuron:
    """Represents a single neuron performing binary classification.

    Attributes:
        nx (int): Number of input features.
        W (numpy.ndarray): Weights vector for the neuron (shape: 1 x nx).
        b (float): Bias for the neuron initialized to 0.
        a (float): Activated output of the neuron (prediction),
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
    def a(self):
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

    def cost(self, Y, a):  # Calcula el costo de modelo con regresión logística
        """
        Calculate the cost of the model using logistic regression.

        The cost function measures the distance between the predictions
        of the neuron (a) and the real labels (Y), penalizing
        wrong predictions more strongly.

        Args:
        Y(np.ndarray): Real label row vector, shape(1, m),
                        values ​​0 or 1.
        a (np.ndarray): Row vector of activations (predictions),
                        shape(1, m), values ​​between 0 and 1.
        """
        # se pone Y.shape[1] para en la tupla conseguir el puesto 1, osea m.
        # Recordar que Y es un ndarray de forma (1, m)
        m = Y.shape[1]

        costo = np.sum(-(Y * np.log(a) + (1-Y) * np.log(1.0000001-a))) / m
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
        prediccion = np.where(a >= 0.5, 1, 0)
        return prediccion, costo

    def gradient_descent(self, X, Y, a, alpha=0.05):
        """
        Perform a gradient descent pass and update the weights and
        neuron bias.

        Arguments:
        X (np.ndarray): Input data of form (n_x, m), where n_x is the
                        number of features and m the number of examples.
        Y (np.ndarray): True labels of shape (1, m), with values ​​0 or 1.
        a (np.ndarray): Activated outputs of the shape neuron (1, m).
        alpha (float): Learning rate.
        """
        # Número de ejemplo
        m = X.shape[1]

        # Error lineal, diferencia entre etiqueta real y predicción
        dZ = a - Y

        # Gradiente de los pesos
        dW = (1/m) * dZ.dot(X.T)

        # Gradiente del sesgo, promedio de todos los dZ
        db = (1/m) * np.sum(dZ)

        # Actualiza pesos y sesgo usando descenso de gradiente
        self.__W = self.__W - (alpha * dW)
        self.__b = self.__b - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        Train the neuron using gradient descent, with optional logging and plotting.

        Args:
            X (np.ndarray): Input data of shape (n_x, m), where n_x is the
                            number of features and m is the number of examples.
            Y (np.ndarray): True labels of shape (1, m), with binary values {0,1}.
            iterations (int): Number of iterations to run gradient descent.
            alpha (float):   Learning rate (step size) for parameter updates.
            verbose (bool):  If True, prints the training cost every 'step'
                            iterations (including iteration 0 and the final one).
            graph (bool):    If True, plots the training cost versus iterations
                            after training completes.
            step (int):      Interval of iterations at which to print and record cost.
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")
        if not isinstance(graph, bool):
            raise TypeError("graph must be a boolean")
        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step < 1 or step > iterations:
            raise ValueError("step must be positive and <= iterations")

        # Propaga hacia adelante con los pesos y sesgo iniciales
        a0 = self.forward_prop(X)

        # Calcula el costo antes de entrenar
        cost0 = self.cost(Y, a0)

        # Inicia las listas para gráfica
        iters = [0]
        costs = [cost0]

        if verbose and (0 % step == 0):
            print(f"Cost after 0 iterations: {cost0}")

        for i in range(1, iterations + 1):
            # calcula la propagación hacia adelante en cada iteración
            a = self.forward_prop(X)

            # Ajusta __W y __b
            self.gradient_descent(X, Y, a, alpha)

            if verbose and (i % step == 0 or i == iterations):
                # Recalcula costo tras la actualización
                cost_i = self.cost(Y, a)
                print(f"Cost after {i} iterations: {cost_i}")
                iters.append(i)
                costs.append(cost_i)

        if graph:
            import matplotlib.pyplot as plt
            # Aca se utilizan las listas que se crearon mas arriba
            plt.plot(iters, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)