#!/usr/bin/env python3
"""
0-monte_carlo.py
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
    This function performs the Monte Carlo algorithm to estimate the value
    function. It updates the value estimate V using the incremental mean.
    Args:
    - env: The environment instance.
    - V: numpy.ndarray of shape (s,) containing the value estimate.
    - policy: Function that takes a state and returns the next action.
    - episodes: Number of episodes to train over.
    - max_steps: Maximum number of steps per episode.
    - alpha: Learning rate.
    - gamma: Discount rate.
    Returns:
    - Updated V, the updated value estimate.
    """
    # Iterar sobre el número total de episodios
    for episodio in range(episodes):
        # Reiniciar el entorno
        estado = env.reset()[0]
        # Lista para almacenar (estado, recompensa)
        datos_episodio = []

        # Generar un episodio completo
        for paso in range(max_steps):
            # Elegir acción según la política
            accion = policy(estado)
            nuevo_estado, recompensa, terminado, truncado, _ = env.step(accion)
            # Guardar estado y recompensa
            datos_episodio.append((estado, recompensa))

            if terminado or truncado:
                break

            # Pasar al siguiente estado
            estado = nuevo_estado

        # Calcular los retornos en orden inverso
        retorno = 0
        # Convertir a array
        datos_episodio = np.array(datos_episodio, dtype=int)

        for estado, recompensa in reversed(datos_episodio):
            # Calcular retorno acumulado
            retorno = recompensa + gamma * retorno

            # Actualizar V(s) usando la media incremental (primer visita)
            if estado not in datos_episodio[:episodio, 0]:
                V[estado] += alpha * (retorno - V[estado])

    # Devolver la función de valor actualizada
    return V
