#!/usr/bin/env python3
"""
Script that displays the first SpaceX launch
"""
import requests


def get_json(url):
    """
    Get data from a URL and return the result as JSON
    """
    return requests.get(url).json()


if __name__ == "__main__":
    # Obtener todos los lanzamientos próximos
    upcoming = get_json("https://api.spacexdata.com/v4/launches/upcoming")

    # Ordenar por fecha usando date_unix
    first_launch = sorted(upcoming, key=lambda i: i["date_unix"])[0]

    # Obtener datos del cohete
    rocket_id = first_launch["rocket"]
    rocket_info = get_json(
        f"https://api.spacexdata.com/v4/rockets/{rocket_id}")

    # Obtener datos de la plataforma
    pad_id = first_launch["launchpad"]
    pad_info = get_json(f"https://api.spacexdata.com/v4/launchpads/{pad_id}")

    # Fecha en local time ya viene dada por la API
    local_date = first_launch["date_local"]

    # Imprimir
    print(
        f"{first_launch['name']} ({local_date}) "
        f"{rocket_info['name']} - "
        f"{pad_info['name']} ({pad_info['locality']})"
    )
