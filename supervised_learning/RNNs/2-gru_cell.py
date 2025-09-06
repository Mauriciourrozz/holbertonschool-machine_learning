import numpy as np


class GRUCell:
    """
    A class that represents a Gated Recurrent Unit (GRU) cell.
    This implementation includes the forward propagation step of the GRU cell.
    """

    def __init__(self, i, h, o):
        """
        Initializes the GRU cell.

        Args:
            i (int): Dimensionality of the input data.
            h (int): Dimensionality of the hidden state.
            o (int): Dimensionality of the output.
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.bz = np.zeros((1, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.br = np.zeros((1, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.bh = np.zeros((1, h))
        self.Wy = np.random.normal(size=(h, o))
        self.by = np.zeros((1, o))
        self.h = h
        self.i = i
        self.o = o

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid activation function.

        Args:
            x (numpy.ndarray): The input array.

        Returns:
            numpy.ndarray: The output of the sigmoid function applied
                element-wise.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """
        Softmax activation function.

        Args:
            x (numpy.ndarray): The input array (m, n) where m is the
                number of samples and n is the number of classes.

        Returns:
            numpy.ndarray: The softmax output applied along the last axis.
        """
        # Resta el valor máximo para estabilidad numérica
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """
        Forward propagation for one time step in the GRU cell.

        Args:
            h_prev (numpy.ndarray): The previous hidden state (m, h).
            x_t (numpy.ndarray): The input data at the current time step (m, i)

        Returns:
            h_next (numpy.ndarray): The next hidden state (m, h).
            y (numpy.ndarray): The output of the GRU cell (m, o).
        """
        # Número de ejemplos
        m = x_t.shape[0]

        # Concatenamos el estado oculto anterior y la entrada actual
        concat_hx = np.concatenate((h_prev, x_t), axis=1)

        # Cálculo de la puerta de actualización
        z_t = self.sigmoid(np.dot(concat_hx, self.Wz) + self.bz)

        # Cálculo de la puerta de reinicio
        r_t = self.sigmoid(np.dot(concat_hx, self.Wr) + self.br)

        # Calculamos el estado oculto candidato usando la puerta de reinicio
        concat_rhx = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_hat = np.tanh(np.dot(concat_rhx, self.Wh) + self.bh)

        # Calculamos el siguiente estado oculto (combinación ponderada del
        # anterior y el candidato)
        h_next = (1 - z_t) * h_prev + z_t * h_hat

        # Calculamos la salida a partir del siguiente estado oculto
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        # Devolvemos el siguiente estado oculto y la salida
        return h_next, y
