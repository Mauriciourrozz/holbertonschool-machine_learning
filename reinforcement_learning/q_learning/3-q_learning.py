#!/usr/bin/env python3
"""
3-q_learning.py
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Trains an agent in the FrozenLake environment using Q-learning.

    Parameters:
    - env: FrozenLake environment
    - Q: initial Q-table (numpy.ndarray)
    - episodes: number of episodes
    - max_steps: maximum number of steps per episode
    - alpha: learning rate
    - gamma: discount factor
    - epsilon: initial probability of exploring
    - min_epsilon: minimum value of epsilon
    - epsilon_decay: epsilon decay rate

    Returns:
    - Q: updated Q-table
    - total_rewards: list of rewards per episode
    """
    total_rewards = []

    for episode in range(episodes):
        state, _ = env.reset()  # reiniciamos el entorno
        episode_reward = 0

        for step in range(max_steps):
            # Elegir acción usando epsilon-greedy
            action = epsilon_greedy(Q, state, epsilon)

            # Tomar acción en el entorno
            new_state, reward, done, _, _ = env.step(action)

            # Actualizar recompensa si cae en un agujero
            if reward == 0 and done:
                reward = -1

            # Actualizar tabla Q
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[new_state]) - Q[state, action]
            )

            state = new_state
            episode_reward += reward

            if done:
                break

        # Guardar recompensa del episodio
        total_rewards.append(episode_reward)

        # Decaer epsilon
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q, total_rewards
