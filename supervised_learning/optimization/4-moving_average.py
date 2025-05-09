#!/usr/bin/env python3
"""
4-moving_average.py
"""
def moving_average(data, beta):
    """
    Calculate the exponentially weighted moving average (EMA) of a series of data.
    
    Parameters:
    data (list or numpy.ndarray): A list of numerical values.
    beta (float): Smoothing factor, a value between 0 and 1.
    
    Returns:
    list: The exponentially weighted moving average of the input data.
    """
    v = 0
    corrected_v = 0
    moving_averages = []

    for t, value in enumerate(data):
        v = beta * v + (1 - beta) * value
        corrected_v = v / (1 - beta**(t + 1))
        moving_averages.append(corrected_v)
    
    return moving_averages
