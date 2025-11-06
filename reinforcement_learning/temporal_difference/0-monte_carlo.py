#!/usr/bin/env python3
"""
0-monte_carlo.py
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
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

    # diccionario para guardar todos los retornos por estado
    retornos = {}
    for s in range(len(V)):
        retornos[s] = []

    # recorrer todos los episodios
    for episodio in range(episodes):
        # reiniciar entorno y variables
        estado, _ = env.reset()
        episodio_datos = []

        # generar un episodio
        for paso in range(max_steps):
            accion = policy(estado)  # elegir acción con la política
            sig_estado, recompensa, terminado, fallo, _ = env.step(accion)

            # guardar datos
            episodio_datos.append((estado, recompensa))
            # avanzar al siguiente estado
            estado = sig_estado

            # terminar si el episodio finaliza
            if terminado or fallo:
                break

        # calcular retornos hacia atrás
        # retorno acumulado
        G = 0.0
        # para first-visit Monte Carlo
        visitados = set()

        for estado_t, recompensa in reversed(episodio_datos):
            # retorno descontado
            G = recompensa + gamma * G

            # si el estado no fue visitado antes en este episodio
            if estado_t not in visitados:
                visitados.add(estado_t)
                # guardar retorno
                retornos[estado_t].append(G)
                # promedio de retornos
                V[estado_t] = np.mean(retornos[estado_t])

    # devolver la estimación de valores actualizada
    return V
