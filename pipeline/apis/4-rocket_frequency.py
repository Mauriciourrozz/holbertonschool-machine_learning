#!/usr/bin/env python3
"""
Script that displays the number of launches per rocket
"""
import requests
from collections import Counter


def get_json(url):
    """
    Return JSON data from a url
    """
    return requests.get(url).json()


if __name__ == '__main__':
    # Obtener todos los lanzamientos
    launches = get_json("https://api.spacexdata.com/v4/launches")

    # Contar lanzamientos por ID de rocket
    rocket_counts = Counter(i["rocket"] for i in launches)

    # Obtener datos de todos los rockets
    rockets = get_json("https://api.spacexdata.com/v4/rockets")
    rocket_names = {r["id"]: r["name"] for r in rockets}

    # Convertir IDs → nombres
    result = [
        (rocket_names[rid], count) for rid, count in rocket_counts.items()]

    # Ordenar: primero por cantidad DESC, luego nombre ASC
    result.sort(key=lambda x: (-x[1], x[0]))

    # Imprimir
    for name, count in result:
        print(f"{name}: {count}")
