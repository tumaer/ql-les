"""Compute normalization stats for avu_target in V2U and update metadata in place."""

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np

from src.utils.nbrs_utils import displ_fn


def infer_every_nth(metadata_path: Path) -> int:
    """Infer the every_nth value from the metadata filename."""
    match = re.search(r"metadata_every(\d+)\.json$", metadata_path.name)
    if match is None:
        raise ValueError("Expected metadata path like metadata_every{n}.json")
    return int(match.group(1))


def compute_avu_stats(metadata_path: Path):
    """Compute mean and std of avu_target on all trajectories in train.h5 and update metadata."""
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    every_nth = infer_every_nth(metadata_path)
    dataset_dir = metadata_path.parent
    h5_path = dataset_dir / "train.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing train split: {h5_path}")

    dim = int(metadata["dim"])
    box = np.asarray(metadata["bounds"], dtype=np.float64)[:dim]
    box = box[:, 1] - box[:, 0]
    pbc = bool(np.asarray(metadata["periodic_boundary_conditions"]).any())
    effective_dt = float(metadata["dt"]) * float(metadata["write_every"])

    sum_x = np.zeros(dim, dtype=np.float64)
    sum_x2 = np.zeros(dim, dtype=np.float64)
    count = 0

    with h5py.File(h5_path, "r") as hf:
        for traj in hf.values():
            fluid = traj["particle_type"][:] == 0
            if not np.any(fluid):
                continue

            pos = traj["position"][::every_nth, fluid, :dim]
            u = traj["u"][::every_nth, fluid, :dim]
            if pos.shape[0] < 2:
                continue

            next_v = displ_fn(pos[1:], pos[:-1], box=box, pbc=pbc)
            avu_target = u[1:] - next_v / effective_dt

            sum_x += avu_target.sum(axis=(0, 1), dtype=np.float64)
            sum_x2 += np.square(avu_target, dtype=np.float64).sum(axis=(0, 1), dtype=np.float64)
            count += avu_target.shape[0] * avu_target.shape[1]

    if count == 0:
        raise RuntimeError("No valid samples found in train split.")

    mean = sum_x / count
    var = np.maximum(sum_x2 / count - np.square(mean), 0.0)
    std = np.sqrt(var)
    std = np.where(std < 1e-7, 1.0, std)

    metadata["avu_mean"] = mean.tolist()
    metadata["avu_std"] = std.tolist()

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"metadata_path={metadata_path}")
    print(f"every_nth={every_nth}")
    print(f"avu_mean={mean}")
    print(f"avu_std={std}")
    print(f"num_samples={count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", type=str, required=True)
    args = parser.parse_args()
    compute_avu_stats(Path(args.metadata_path))
