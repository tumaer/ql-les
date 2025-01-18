import json
def load_metadata(data_path):
    with open(f'{data_path}/metadata.json', 'rt') as f:
        metadata = json.loads(f.read())
    return metadata