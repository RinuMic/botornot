"""
utils.py

This module provides utility functions for preprocessing data, including functions
for encoding visitor recognition types, calculating URL lengths, checking referrer
presence, and determining URL types.

Functions:
- encode_recognition_type(rec_type): Encodes the visitor recognition type into numerical values.
- calculate_url_length(url): Calculates the length of a URL.
- check_referrer_presence(ref): Checks if the referrer is present.
- determine_url_type(url): Determines the type of URL based on its structure.
"""
import pandas as pd

def encode_recognition_type(rec_type):
    """
    Encodes the visitor recognition type into numerical values.
    Args:
        rec_type (str): The visitor recognition type.
    Returns:
        int: Encoded value corresponding to the type, or -1 if type is not found in encoding_map.
    """
    encoding_map = {'': 0, 'ANONYMOUS': 1, 'LOGGEDIN': 2, 'RECOGNIZED': 3}
    return encoding_map.get(rec_type, -1)

def calculate_url_length(url):
    """
    Calculates the length of the URL.

    Args:
        url (str): The URL string.

    Returns:
        int: Length of the URL.
    """
    return len(url)

def check_referrer_presence(ref):
    """
    Check if the referrer is present.

    This function checks if the provided referrer string is null or empty,
    and returns 0 if it is, otherwise returns 1.

    Args:
        ref (str or None): The referrer string to check.

    Returns:
        int: 0 if referrer is null or empty, otherwise 1.
    """
    if pd.isnull(ref) or ref == '':
        return 0
    return 1

def determine_url_type(url):
    """
    Determines the type of URL based on its structure.
    Args:
        url (str): The URL to analyze.
    Returns:
        str: 'product' if '/p/' is in the URL, 'category' if '/l/' is in the URL, otherwise 'other'.
    """
    if '/p/' in url:
        return 'product'
    if '/l/' in url:
        return 'category'
    return 'other'
