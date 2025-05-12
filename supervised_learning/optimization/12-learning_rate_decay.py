#!/usr/bin/env python3
"""
11-learning_rate_decay.py
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation using inverse time decay.

    Parameters:
    alpha (float): Initial learning rate.
    decay_rate (float): The rate at which the learning rate decays.
    decay_step (int): Number of steps before applying more decay.

    Returns:
    tf.keras.optimizers.schedules.LearningRateSchedule: A learning
    rate schedule.
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
