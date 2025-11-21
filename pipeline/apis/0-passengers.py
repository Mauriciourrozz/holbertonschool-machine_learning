#!/usr/bin/env python3
"""
0-passengers.py
"""
import requests


def availableShips(passengerCount):
    """
    Returns a list of ships that can carry at least
    'passengerCount' passengers, using the SWAPI API.
    """

    url = "https://swapi.dev/api/starships/"
    ships = []

    while url:
        # Hacer la solicitud a la API
        response = requests.get(url)
        data = response.json()

        # Recorrer cantidad de naves
        for ship in data.get("results", []):
            pasajeros = ship.get("passengers", "unknown")

            # Ignorar valores desconocidos
            if pasajeros in ["n/a", "unknown"]:
                continue

            # Quitar comas
            pasajeros = pasajeros.replace(",", "")

            # tomar el primer número
            if "-" in pasajeros:
                pasajeros = pasajeros.split("-")[0]

            # Convertir a entero
            try:
                pasajeros = int(pasajeros)
            except ValueError:
                continue

            # Si la nave soporta la cantidad pedida, agregarla
            if pasajeros >= passengerCount:
                ships.append(ship["name"])

        # Pasar a la siguiente página
        url = data.get("next")

    return ships
