#!/usr/bin/env python3
"""
0-monte_carlo.py
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99):
    """
    Perform first-visit Monte Carlo policy evaluation.

    Assumes:
    - env.reset() returns an initial state (integer index for V).
    - env.step(action) returns (next_state, reward, done, info).
    - policy(state) returns an action for the given state.
    - V is a numpy array of shape (s,) with initial value estimates.

    Returns:
    - V: the updated value estimates (numpy.ndarray).
    """

    # entrenar por la cantidad de episodios indicada
    for episodio in range(episodes):
        # reiniciar el entorno y variables del episodio
        estado, _ = env.reset()
        estados_ep = []
        recompensas_ep = []

        # generar un episodio (hasta max_steps o hasta done)
        for paso in range(max_steps):
            # obtener acción de la política
            accion = policy(estado)

            # ejecutar la acción en el entorno
            siguiente_estado, recompensa, terminado, _, info = env.step(accion)

            # almacenar experiencia
            estados_ep.append(estado)
            recompensas_ep.append(recompensa)

            # avanzar
            estado = siguiente_estado

            if terminado:
                break

        # actualizar V usando first-visit Monte Carlo con learning rate alpha
        T = len(estados_ep)
        # para cada tiempo t en el episodio calculamos el retorno G desde t
        for t in range(T):
            estado_t = estados_ep[t]

            # verificar si es la primera vez que aparece estado_t en el episodio
            if estado_t in estados_ep[:t]:
                continue

            # calcular el retorno G desde t (suma descontada)
            retorno = 0.0
            factor = 1.0
            for k in range(t, T):
                retorno += factor * recompensas_ep[k]
                factor *= gamma

            # actualizar la estimación de valor para el estado (regla incremental con alpha)
            V[estado_t] = V[estado_t] + alpha * (retorno - V[estado_t])

    return V
