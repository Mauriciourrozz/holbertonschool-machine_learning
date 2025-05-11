#!/usr/bin/env python3
"""
11-learning_rate_decay.py
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay.

    Parameters:
    alpha (float): Initial learning rate.
    decay_rate (float): The rate at which alpha decays.
    global_step (int): Number of gradient descent steps taken.
    decay_step (int): Number of steps before applying further decay.

    Returns:
    float: The updated learning rate.
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))
