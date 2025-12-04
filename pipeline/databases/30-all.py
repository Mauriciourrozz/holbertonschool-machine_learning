#!/usr/bin/env python3
"""
30-all.py
"""


def list_all(mongo_collection):
    """
    Return a list of all documents in the collection.
    If the collection is empty, return an empty list.
    """
    # Devuelve todos los documentos de la colección
    return list(mongo_collection.find())
