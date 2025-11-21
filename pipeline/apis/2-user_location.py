#!/usr/bin/python3
"""
Script that prints the location of a GitHub user
"""
import sys
import requests
import time

if __name__ == "__main__":
    # El argumento debe ser la URL del usuario en la API
    if len(sys.argv) < 2:
        sys.exit(0)

    url = sys.argv[1]

    try:
        response = requests.get(url)
    except Exception:
        print("Not found")
        sys.exit(0)

    # Si el usuario no existe 404
    if response.status_code == 404:
        print("Not found")
        sys.exit(0)

    # Si GitHub bloquea por rate limit 403
    if response.status_code == 403:
        # Tiempo de reset que viene en segundos unix
        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
        now = int(time.time())
        minutes_left = (reset_time - now) // 60
        print(f"Reset in {minutes_left} min")
        sys.exit(0)

    # Si todo esta biense imprime la ubicación
    data = response.json()
    location = data.get("location")

    if location:
        print(location)
    else:
        print("Not found")
