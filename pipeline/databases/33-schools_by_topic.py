#!/usr/bin/env python3
"""
33-schools_by_topic.py
"""


def schools_by_topic(mongo_collection, topic):
    """
    Return list of schools having a specific topic
    """
    # Busca todos los documentos que tengan el topic en la lista topics
    return list(mongo_collection.find({"topics": topic}))
