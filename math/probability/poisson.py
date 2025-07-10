#!/usr/bin/env python3
"""
poisson.py
"""


class Poisson:
    """
    Represents a Poisson distribution.

    Attributes:
        lambtha (float): The expected number of
        occurrences in a given interval (λ).
    """
    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the Poisson distribution instance.

        Parameters:
            data (list, optional): A list of data points
            to estimate the distribution's λ.
                                   If None, lambtha is used directly.
            lambtha (float, optional): The expected number of occurrences (λ).
                                       Default is 1.0.

        Raises:
            TypeError: If data is not a list.
            ValueError: If data contains fewer than two data points.
            ValueError: If lambtha is not a positive value.
        """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            # lambtha es el promedio esperado de eventos
            # en un intervalo de tiempo
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            # lambtha se calcula como el promedio de los datos
            self.lambtha = float(sum(data) / len(data))


    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of "successes" k.

        Parameters:
            k (int or float): The number of events ("successes").

        Returns:
            float: The probability of exactly k events occurring.

        """
        k = int(k)
        if k < 0:
            return 0

        exponente = -self.lambtha
        resultado = 1.0
        termino = 1.0
        for i in range(1, 30):
            termino *= exponente / i
            resultado += termino

        fact = 1
        for i in range(2, k + 1):
            fact *= i

        pmf_value = (resultado * (self.lambtha ** k)) / fact

        return pmf_value
