#!/usr/bin/env python3
"""
exponential.py
"""


class Exponential:
    """
    Represents an Exponential distribution.

    This distribution models the time between events that happen
    at a constant average rate (like the time between customers
    arriving at a barbershop).

    Attributes:
    - lambtha (float): the expected number of events per time unit.
                       Must be a positive number.
    """
    def __init__(self, data=None, lambtha=1.):
        """
        Initialize an Exponential distribution.

        Parameters:
        - data (list): a list of time values between events.
                       Used to estimate the parameter lambtha.
        - lambtha (float): expected number of events per time unit.
                           Used only if data is not given.
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
            self.lambtha = float(len(data) / sum(data))

    def pdf(self, x):
        """
        Calculate the value of the PDF for a given time x.

        Parameters:
        - x (float): the time to evaluate the PDF at

        Returns:
        - The PDF value for x
        """
        if x < 0:
            return 0

        e = 2.7182818285
        return self.lambtha * (e ** (-self.lambtha * x))

    def cdf(self, x):
        """
        Calculate the value of the CDF for a given time x.

        Parameters:
        - x (float): the time to evaluate the CDF at

        Returns:
        - The CDF value for x
        """
        if x < 0:
            return 0

        e = 2.7182818285
        return 1 - (e ** (-self.lambtha * x))
