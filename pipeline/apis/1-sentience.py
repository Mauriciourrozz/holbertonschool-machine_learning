#!/usr/bin/env python3
"""
1-sentience.py
"""
import requests


def sentientPlanets():
    """
    Returns a list with the names of the home planets
    of all 'sentient' species according to SWAPI.
    """

    url = "https://swapi.dev/api/species/"
    planet_names = []
    seen = set()

    while url:
        response = requests.get(url)
        data = response.json()

        for species in data.get("results", []):
            classification = (species.get("classification") or "").lower()
            designation = (species.get("designation") or "").lower()

            if "sentient" in classification or "sentient" in designation:
                homeworld_url = species.get("homeworld")
                if not homeworld_url:
                    continue

                planet_data = requests.get(homeworld_url).json()
                name = planet_data.get("name")

                if name and name not in seen:
                    planet_names.append(name)
                    seen.add(name)

        url = data.get("next")

    return planet_names
