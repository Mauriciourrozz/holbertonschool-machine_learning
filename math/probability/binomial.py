class Binomial:
    """
    Represents a Binomial distribution.

    Attributes:
    - n (int): number of Bernoulli trials
    - p (float): probability of success in each trial
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize the Binomial distribution.

        If data is given, estimate n and p from data.
        Otherwise, use the given n and p.

        Parameters:
        - data (list, optional): list of data points to estimate parameters
        - n (int): number of trials (default 1)
        - p (float): probability of success (default 0.5)

        Raises:
        - TypeError: if data is not a list
        - ValueError: if data has less than two values
        - ValueError: if n is not positive
        - ValueError: if p is not in (0,1)
        """
        if data is None:
            if not isinstance(n, int) or n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")

            self.n = n
            self.p = float(p)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)

            diffs = [(x - mean) ** 2 for x in data]
            variance = sum(diffs) / len(data)

            p = 1 - (variance / mean)
            n = round(mean / p)
            p = mean / n

            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")

            self.n = n
            self.p = float(p)
