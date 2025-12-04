#!/usr/bin/env python3
"""
Provides stats about Nginx logs stored in MongoDB
"""
from pymongo import MongoClient


if __name__ == "__main__":
    # Conexión a la base logs
    client = MongoClient()
    collection = client.logs.nginx

    # Cantidad total de documentos
    total = collection.count_documents({})
    print("{} logs".format(total))

    print("Methods:")

    # Lista de métodos en orden
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]

    # Conteo de cada método
    for method in methods:
        count = collection.count_documents({"method": method})
        print("\tmethod {}: {}".format(method, count))

    # Conteo de GET /status
    status_count = collection.count_documents(
        {"method": "GET", "path": "/status"}
    )
    print("{} status check".format(status_count))
