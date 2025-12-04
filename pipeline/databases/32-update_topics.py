#!/usr/bin/env python3
"""
32-update_topics.py
"""


def update_topics(mongo_collection, name, topics):
    """
    Update all topics of a school document based on the school name
    """
    # Actualiza todos los documentos cuyo nombre coincida
    mongo_collection.update_many(
        {"name": name},
        {"$set": {"topics": topics}}
    )
