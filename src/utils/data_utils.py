import json
import jax.numpy as jnp
import h5py
import numpy as np


EPS = jnp.finfo(float).eps


def load_metadata(data_path: str):
    """Load metadata from a JSON file."""
    if not data_path.endswith(".json"):
        data_path = f"{data_path}/metadata.json"
    with open(data_path, "rt") as f:
        metadata = json.loads(f.read())
    return metadata


def read_h5(file_name: str, array_type: str = "jax"):
    """Read an .h5 file and return a dict of numpy or jax arrays."""
    hf = h5py.File(file_name, "r")

    data_dict = {}
    for k, v in hf.items():
        if array_type == "jax":
            data_dict[k] = jnp.array(v)
        elif array_type == "numpy":
            data_dict[k] = np.array(v)
        else:
            raise ValueError('array_type must be either "jax" or "numpy"')

    hf.close()

    return data_dict
