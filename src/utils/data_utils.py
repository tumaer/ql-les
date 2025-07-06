import json


def load_metadata(data_path: str):
    """Load metadata from a JSON file."""
    if not data_path.endswith(".json"):
        data_path = f"{data_path}/metadata.json"
    with open(data_path, "rt") as f:
        metadata = json.loads(f.read())
    return metadata
