#!/usr/bin/env python3
"""
Script that displays the first SpaceX launch
"""

import requests
from datetime import datetime

if __name__ == "__main__":
    # Urls principales de la API no oficial de SpaceX
    launches_url = "https://api.spacexdata.com/v4/launches"
    rockets_url = "https://api.spacexdata.com/v4/rockets/"
    launchpads_url = "https://api.spacexdata.com/v4/launchpads/"

    # Obtener todos los lanzamientos
    launches = requests.get(launches_url).json()

    # Ordenar por date_unix en orden ascendente
    launches = sorted(
        launches,
        key=lambda x: x.get("date_unix", float("inf"))
    )

    # Tomar el primer lanzamiento
    first = launches[0]

    launch_name = first.get("name")
    date_unix = first.get("date_unix")
    rocket_id = first.get("rocket")
    launchpad_id = first.get("launchpad")

    # Convertir fecha a hora local
    date_local = datetime.fromtimestamp(date_unix).strftime("%Y-%m-%d %H:%M:%S")

    # Obtener datos del cohete
    rocket = requests.get(rockets_url + rocket_id).json()
    rocket_name = rocket.get("name")

    # Obtener datos de la plataforma
    launchpad = requests.get(launchpads_url + launchpad_id).json()
    launchpad_name = launchpad.get("name")
    launchpad_locality = launchpad.get("locality")

    # Imprimir
    print(f"{launch_name} ({date_local}) {rocket_name} - {launchpad_name} ({launchpad_locality})")
