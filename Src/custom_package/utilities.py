import hashlib


def compute_unique_primary_key(input_str):
    hashed_string = hashlib.sha256(input_str.encode('utf-8')).hexdigest()
    return hashed_string
