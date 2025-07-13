#!/usr/bin/env python3
"""
normal.py
"""


class Normal:
    """
    Represents a Normal (Gaussian) distribution.

    This distribution models data that clusters around
    a mean with a certain standard deviation.

    Attributes:
    - mean (float): the average value of the distribution.
    - stddev (float): the standard deviation (spread) of the distribution.
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize an Normal distribution.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = sum(data) / len(data)
            dif = [(i - self.mean) ** 2 for i in data]
            sum_dif = sum(dif)
            variance = sum_dif / len(data)
            self.stddev = variance ** 0.5

    def z_score(self, x):
        """
        Calculate the z-score of a given x-value.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculate the x-value corresponding to a given z-score.
        """
        return self.mean + z * self.stddev

    def pdf(self, x):
        """
        Calculate the value of the PDF for a given time x.
        """
        e = 2.7182818285
        pi = 3.1415926536

        exp_part = ((x - self.mean) / self.stddev) ** 2
        numerator = e ** (-0.5 * exp_part)
        denominator = self.stddev * (2 * pi) ** 0.5

        return numerator / denominator
