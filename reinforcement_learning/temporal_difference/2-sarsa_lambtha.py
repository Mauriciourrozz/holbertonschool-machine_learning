#!/usr/bin/env python3
"""
SARSA(λ) Value Estimation for Reinforcement Learning.
"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """
    This function performs the SARSA(λ) algorithm to estimate the Q table.
    Args:
        - env: The environment instance.
        - Q: numpy.ndarray of shape (s, a) containing the Q table.
        - lambtha: The eligibility trace factor.
        - episodes: Number of episodes to train over.
        - max_steps: Maximum number of steps per episode.
        - alpha: Learning rate.
        - gamma: Discount rate.
        - epsilon: Initial threshold for epsilon greedy.
        - min_epsilon: Minimum value that epsilon should decay to.
        - epsilon_decay: Decay rate for updating epsilon between episodes.
    Returns:
        - Updated Q, the Q table.
    """
    # Guardar el valor inicial de epsilon
    epsilon_inicial = epsilon

    # Iterar sobre cada episodio
    for episodio in range(episodes):
        # Inicializar las trazas de elegibilidad con ceros
        trazas = np.zeros_like(Q)
        # Reiniciar el entorno y obtener el estado inicial
        estado = env.reset()[0]

        # Elegir la primera acción usando política epsilon-greedy
        if np.random.uniform() < epsilon:
            accion = np.random.randint(Q.shape[1])
        else:
            accion = np.argmax(Q[estado])

        # Iterar sobre los pasos del episodio
        for _ in range(max_steps):
            # Ejecutar la acción y observar el nuevo estado y la recompensa
            nuevo_estado, recompensa, terminado, truncado, _ = env.step(accion)

            # Elegir la siguiente acción usando política epsilon-greedy
            if np.random.uniform() < epsilon:
                nueva_accion = np.random.randint(Q.shape[1])
            else:
                nueva_accion = np.argmax(Q[nuevo_estado])

            # Calcular el error TD (delta)
            delta = (recompensa + gamma * Q[nuevo_estado, nueva_accion] -
                     Q[estado, accion])

            # Actualizar la traza de elegibilidad para el par estado-acción
            trazas[estado, accion] += 1

            # Actualizar la tabla Q usando el error TD y las trazas
            Q += alpha * delta * trazas

            # Reducir las trazas de elegibilidad con el factor gamma y lambda
            trazas *= gamma * lambtha

            # Actualizar estado y acción
            estado, accion = nuevo_estado, nueva_accion

            # Si el episodio terminó, salir del bucle
            if terminado or truncado:
                break

        # Disminuir el valor de epsilon (decaimiento de exploración)
        epsilon = (min_epsilon + (epsilon_inicial - min_epsilon) *
                   np.exp(-epsilon_decay * episodio))

    # Devolver la tabla Q actualizada
    return Q
