#!/usr/bin/env python3
"""
1-td_lambtha.py
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    This function performs the TD(λ) algorithm to estimate the value function.
    Args:
    - env: The environment instance.
    - V: numpy.ndarray of shape (s,) containing the value estimate.
    - policy: Function that takes a state and returns the next action.
    - lambtha: The eligibility trace factor.
    - episodes: Number of episodes to train over.
    - max_steps: Maximum number of steps per episode.
    - alpha: Learning rate.
    - gamma: Discount rate.
    Returns:
    - Updated V, the updated value estimate.
    """
    # Iterar sobre cada episodio
    for _ in range(episodes):
        # Inicializar las trazas de elegibilidad con ceros
        trazas = np.zeros_like(V)
        # Reiniciar entorno y obtener estado inicial
        estado = env.reset()[0]

        # Iterar sobre los pasos dentro del episodio
        for _ in range(max_steps):
            # Elegir una acción según la política
            accion = policy(estado)

            # Ejecutar la acción en el entorno
            nuevo_estado, recompensa, terminado, truncado, _ = env.step(accion)

            # Calcular el error TD (delta)
            delta = recompensa + gamma * V[nuevo_estado] - V[estado]

            # Actualizar la traza de elegibilidad del estado actual
            trazas[estado] += 1

            # Actualizar todos los valores de estado según el error TD
            V += alpha * delta * trazas

            # Reducir las trazas de elegibilidad con el factor de descuento
            trazas *= gamma * lambtha

            # Pasar al siguiente estado
            estado = nuevo_estado

            # Si el episodio termina o se trunca, salir del bucle
            if terminado or truncado:
                break

    # Devolver la función de valor actualizada
    return V
