#!/usr/bin/env python3
"""
Script that displays the first SpaceX launch
"""


if __name__ == "__main__":
    # Endpoint de lanzamientos
    launches_url = "https://api.spacexdata.com/v4/launches"

    launches = requests.get(launches_url).json()

    # Ordenar por date_unix (segundos)
    launches_sorted = sorted(launches, key=lambda x: x.get("date_unix", 0))

    # Primer lanzamiento
    first = launches_sorted[0]

    # Obtener datos individuales
    launch_name = first["name"]

    # Convertimos fecha a local
    date_local = datetime.fromtimestamp(
        first["date_unix"]).strftime("%Y-%m-%d %H:%M:%S")

    # Obtener detalles del cohete
    rocket_id = first["rocket"]
    rocket = requests.get(
        f"https://api.spacexdata.com/v4/rockets/{rocket_id}").json()
    rocket_name = rocket["name"]

    # Obtener detalles del launchpad
    pad_id = first["launchpad"]
    pad = requests.get(
        f"https://api.spacexdata.com/v4/launchpads/{pad_id}").json()
    pad_name = pad["name"]
    pad_locality = pad["locality"]

    # Formato de salida
    print(f"{launch_name} ({date_local}) {rocket_name} - {pad_name} ({pad_locality})")