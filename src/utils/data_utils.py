import json


def load_metadata(data_path):
    """Load metadata from a JSON file."""
    with open(f"{data_path}/metadata.json", "rt") as f:
        metadata = json.loads(f.read())
    return metadata
