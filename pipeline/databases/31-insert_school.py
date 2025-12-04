#!/usr/bin/env python3
"""
31-insert_school.py
"""


def insert_school(mongo_collection, **kwargs):
    """
    Insert a new document in the collection using kwargs.
    Returns the inserted document _id.
    """
    # Inserta el documento usando los kwargs
    result = mongo_collection.insert_one(kwargs)
    return result.inserted_id
