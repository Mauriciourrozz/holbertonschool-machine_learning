#!/usr/bin/env python3
"""
train.py
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient
policy = __import__('policy_gradient').policy


def train(env, nb_episodes, alpha=0.000045, gamma=0.98):
    """
    Trains an agent using Monte-Carlo policy gradient.

    Args:
        env: environment instance
        nb_episodes (int): number of episodes
        alpha (float): learning rate
        gamma (float): discount factor

    Returns:
        list: scores for each episode
    """

    weight = np.random.rand(env.observation_space.shape[0],
                            env.action_space.n)

    scores = []

    for episode in range(nb_episodes):

        # Gymnasium returns (observation, info)
        state = env.reset()[0]
        grads = []
        rewards = []
        done = False

        while not done:
            action, grad = policy_gradient(state, weight)
            new_state, reward, terminated, truncated, info = env.step(action)

            grads.append(grad)
            rewards.append(reward)

            state = new_state
            done = terminated or truncated

        score = sum(rewards)
        scores.append(score)

        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + gamma * G
            weight += alpha * grads[t] * G

        print("Episode: {} Score: {}".format(episode, score))

    return scores
