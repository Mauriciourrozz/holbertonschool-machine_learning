#!/usr/bin/env python3
"""
4-play.py
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Makes the trained agent play a full episode.

    Parameters:
    - env: FrozenLake environment (with render_mode="ansi")
    - Q: Trained Q table (numpy.ndarray)
    - max_steps: Maximum steps per episode

    Returns:
    - total_reward: Total reward for the episode
    - outputs: List of rendered board states at each step
    """
    # reiniciamos el entorno
    state, _ = env.reset()
    total_reward = 0
    outputs = []

    for step in range(max_steps):
        # Renderizamos el estado actual y lo guardamos
        rendered = env.render()
        outputs.append(rendered)

        # Elegimos la acción con mayor valor Q
        action = np.argmax(Q[state])

        # Tomamos la acción
        new_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = new_state

        if done:
            # Guardamos el estado final
            outputs.append(env.render())
            break

    return total_reward, outputs
