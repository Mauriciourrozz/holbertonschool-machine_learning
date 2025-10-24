#!/usr/bin/env python3
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the Gymnasium FrozenLakeEnv environment.

    Parameters:
    - desc: List of lists with the custom map description or None
    - map_name: Name of the predefined map or None
    - is_slippery: Boolean, True if the ice is slippery

    Returns:
    - env: The loaded environment
    """
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery
    )
    return env