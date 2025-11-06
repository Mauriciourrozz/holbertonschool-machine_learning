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

    # recorrer el número de episodios
    for episodio in range(episodes):
        # inicializar estado
        estado, _ = env.reset()[0]
        # almacenar secuencia de (estado, recompensa)
        datos_ep = []

        # generar un episodio
        for paso in range(max_steps):
            # elegir acción según la política
            accion = policy(estado)
            sig_estado, recompensa, terminado, fallo, _ = env.step(accion)
            # guardar datos del episodio
            datos_ep.append((estado, recompensa))

            if terminado or fallo:
                break 
            # mover al siguiente estado
            estado = sig_estado

        # calcular los retornos en orden inverso
        # inicializar el retorno
        G = 0
        # convertir a array numpy
        datos_ep = np.array(datos_ep, dtype=int)

        for estado, recompensa in reversed(datos_ep):
            # calcular retorno
            G = recompensa + gamma * G

            # actualizar V(s) usando el promedio incremental (first-visit MC)
            if estado not in datos_ep[:episodio, 0]:
                V[estado] += alpha * (G - V[estado])

    return V  # devolver el valor actualizado