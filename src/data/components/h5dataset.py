"""Dataset modules for loading HDF5 simulation trajectories."""
from __future__ import print_function
import enum
import bisect
import importlib
import json
import os
import os.path as osp
import re
import warnings
import zipfile
from typing import Optional

import h5py
import numpy as np
import wget
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data, InMemoryDataset




ZENODO_PREFIX = "https://zenodo.org/records/10491868/files/"
URLS = {
    "tgv2d": f"{ZENODO_PREFIX}2D_TGV_2500_10kevery100.zip",
    "rpf2d": f"{ZENODO_PREFIX}2D_RPF_3200_20kevery100.zip",
    "ldc2d": f"{ZENODO_PREFIX}2D_LDC_2708_10kevery100.zip",
    "dam2d": f"{ZENODO_PREFIX}2D_DAM_5740_20kevery100.zip",
    "tgv3d": f"{ZENODO_PREFIX}3D_TGV_8000_10kevery100.zip",
    "rpf3d": f"{ZENODO_PREFIX}3D_RPF_8000_10kevery100.zip",
    "ldc3d": f"{ZENODO_PREFIX}3D_LDC_8160_10kevery100.zip",
}

class NodeType(enum.IntEnum):
    """Particle types."""

    PAD_VALUE = -1
    FLUID = 0
    SOLID_WALL = 1
    MOVING_WALL = 2
    RIGID_BODY = 3
    SIZE = 9

class H5Dataset(Dataset):
    """Dataset for loading HDF5 simulation trajectories.

    Reference on parallel loading of h5 samples see:
    https://github.com/pytorch/pytorch/issues/11929

    Implementation inspired by:
    https://github.com/Open-Catalyst-Project/ocp/blob/main/ocpmodels/datasets/lmdb_dataset.py
    """

    def __init__(
        self,
        split: str,
        dataset_path: str,
        name: Optional[str] = None,
        input_seq_length: int = 6,
        extra_seq_length: int = 0,
        nl_backend: str = "matscipy",
    ):
        """Initialize the dataset. If the dataset is not present, it is downloaded.

        Args:
            split: "train", "valid", or "test"
            dataset_path: Path to the dataset. Download will start automatically if
                dataset_path does not exist.
            name: Name of the dataset. If None, it is inferred from the path.
            input_seq_length: Length of the input sequence. The number of historic
                velocities is input_seq_length - 1. And during training, the returned
                number of past positions is input_seq_length + 1, to compute target
                acceleration.
            extra_seq_length: During training, this is the maximum number of pushforward
                unroll steps. During validation/testing, this specifies the largest
                N-step MSE loss we are interested in, e.g. for best model checkpointing.
            nl_backend: Which backend to use for the neighbor list
        """

        dataset_path = osp.normpath(dataset_path)  # remove potential trailing slash

        if name is None:
            self.name = get_dataset_name_from_path(dataset_path)
        else:
            self.name = name
        if not osp.exists(dataset_path):
            dataset_path = self.download(self.name, dataset_path)

        assert split in ["train", "valid", "test"]
        assert (
            input_seq_length > 1
        ), "To compute at least one past velocity, input_seq_length must be >= 2."
        self.dataset_path = dataset_path
        self.file_path = osp.join(dataset_path, split + ".h5")
        self.input_seq_length = input_seq_length
        self.nl_backend = nl_backend
        self.split = split

        force_fn_path = osp.join(dataset_path, "force.py")
        if osp.exists(force_fn_path):
            # load force_fn if `force.py` exists in dataset_path
            spec = importlib.util.spec_from_file_location("force_module", force_fn_path)
            force_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(force_module)

            self.external_force_fn = force_module.force_fn
        else:
            if self.name in ["dam2d", "rpf2d", "rpf3d"]:
                raise FileNotFoundError(
                    f"External force function not found in {dataset_path}. "
                    "Download the latest LagrangeBench dataset from Zenodo."
                )
            self.external_force_fn = None

        # load dataset metadata
        with open(osp.join(dataset_path, "metadata.json"), "r") as f:
            self.metadata = json.loads(f.read())

        self.db_hdf5 = None

        with h5py.File(self.file_path, "r") as f:
            self.traj_keys = list(f.keys())

            # (num_steps, num_particles, dim) = f["00000/position"].shape
            self.sequence_length = f["00000/position"].shape[0]

        if split == "train":
            # During training, the first input_seq_length steps can only be used as
            # input, and the last one to compute the target acceleration. If we use
            # pushforward, then we need to provide extra_seq_length more steps
            # from the end of a trajectory. Thus, the number of training samples per
            # trajectory becomes:

            self.subseq_length = input_seq_length + 1 + extra_seq_length
            samples_per_traj = self.sequence_length - self.subseq_length + 1 #number of trajectory samples for a given trajectory

            keylens = np.array([samples_per_traj for _ in range(len(self.traj_keys))]) # 
            self._keylen_cumulative = np.cumsum(keylens).tolist()

            self.num_samples = sum(keylens)
            self.getter = self.get_window

        else:
            assert (
                extra_seq_length > 0
            ), "extra_seq_length must be > 0 for validation and testing."
            # Compute the number of splits per validation trajectory. If the length of
            # each trajectory is 1000, we want to compute a 20-step MSE, and
            # intput_seq_length=6, then we should split the trajectory into
            # _split_valid_traj_into_n = 1000 // (20 + 6) chunks.
            self.subseq_length = input_seq_length + extra_seq_length
            self._split_valid_traj_into_n = self.sequence_length // self.subseq_length

            self.num_samples = self._split_valid_traj_into_n * len(self.traj_keys)
            self.getter = self.get_trajectory

        assert self.sequence_length >= self.subseq_length, (
            f"# steps in dataset trajectory ({self.sequence_length}) must be >= "
            f"subsequence length ({self.subseq_length}). Reduce either "
            f"input_seq_length or extra_seq_length/max pushforward steps."
        )

    # def download(self, name: str, path: str) -> str:
    #     """Download the dataset.

    #     Args:
    #         name: Name of the dataset
    #         path: Destination path to the downloaded dataset
    #     """

    #     assert name in URLS, f"Dataset {name} not available."
    #     url = URLS[name]

    #     # path could be e.g. "./data/2D_TGV_2500_10kevery100/"
    #     # remove trailing slash if present and get the root of the datasets
    #     path = path[:-1] if path.endswith("/") else path
    #     path_root = osp.split(path)[0]  # e.g. # "./data"

    #     # download the dataset as a zip file, e.g. "./data/2D_TGV_2500_10kevery100.zip"
    #     os.makedirs(path_root, exist_ok=True)
    #     filename = wget.download(url, out=path_root)
    #     print(f"\nDataset {name} downloaded to {filename}")

    #     # unzip the dataset and then remove the zip file
    #     zipfile.ZipFile(filename, "r").extractall(path_root)
    #     os.remove(filename)

    #     return path

    def _open_hdf5(self) -> h5py.File:
        if self.db_hdf5 is None:
            return h5py.File(self.file_path, "r")
        else:
            return self.db_hdf5


    def get_trajectory(self, idx: int):
        """Get a (full) trajectory and index idx."""
        # open the database file
        self.db_hdf5 = self._open_hdf5()

        if self._split_valid_traj_into_n > 1:
            traj_idx = idx // self._split_valid_traj_into_n
            slice_from = (idx % self._split_valid_traj_into_n) * self.subseq_length
            slice_to = slice_from + self.subseq_length
        else:
            traj_idx = idx
            slice_from = 0
            slice_to = self.sequence_length

        # get a pointer to the trajectory. That is not yet the real trajectory.
        traj = self.db_hdf5[f"{self.traj_keys[traj_idx]}"]
        # get a pointer to the positions of the traj. Still nothing in memory.
        traj_pos = traj["position"]
        # load and transpose the trajectory
        pos_input = torch.Tensor(traj_pos[slice_from:slice_to].transpose((1, 0, 2)))
        
        particle_type = torch.Tensor( traj["particle_type"][:]).int()
        
        position_dict = {"position": pos_input, "particle_type": particle_type}
        
        #if ds contains physical velocity target
        if "u" in traj:
            traj_u_vel = traj["u"]
            u_input_and_target = torch.Tensor(traj_u_vel[slice_from:slice_to].transpose((1, 0, 2)))
            position_dict["u"] = u_input_and_target
        
        return position_dict

    def get_window(self, idx: int):
        """Get a window of the trajectory and index idx."""
        # figure out which trajectory this should be indexed from.
        traj_idx = bisect.bisect(self._keylen_cumulative, idx)
        # extract index of element within that trajectory.
        el_idx = idx
        if traj_idx != 0:
            el_idx = idx - self._keylen_cumulative[traj_idx - 1]
        assert el_idx >= 0

        # open the database file
        self.db_hdf5 = self._open_hdf5()

        # get a pointer to the trajectory. That is not yet the real trajectory.
        traj = self.db_hdf5[f"{self.traj_keys[traj_idx]}"]
        # get a pointer to the positions of the traj. Still nothing in memory.
        traj_pos = traj["position"]
        # load only a slice of the positions. Now, this is an array in memory.
        pos_input_and_target = traj_pos[el_idx : el_idx + self.subseq_length]
        pos_input_and_target = torch.Tensor(pos_input_and_target.transpose((1, 0, 2)))

        particle_type = torch.Tensor(traj["particle_type"][:]).int()
        
        position_dict = {"position": pos_input_and_target, "particle_type": particle_type}
        
        #if ds contains physical velocity target
        if "u" in traj:
            traj_u_vel = traj["u"]
            u_input_and_target = traj_u_vel[el_idx : el_idx + self.subseq_length]
            u_input_and_target = torch.Tensor(u_input_and_target.transpose((1, 0, 2)))
            position_dict["u"] = u_input_and_target
        
        return position_dict
    
    def __getitem__(self, idx: int):
        """
        Get a sequence of positions (of size windows) from the dataset at index idx.

        Returns:
            Array of shape (num_particles_max, input_seq_length + 1, dim). Along axis=1
                the position sequence (length input_seq_length) and the last position to
                compute the target acceleration.
        """
        position_dict = self.getter(idx)
        
        if "u" in position_dict:
            return Data(
                enc_pos=position_dict["position"][:, :self.input_seq_length],
                target_pos=position_dict["position"][:, self.input_seq_length:],
                enc_u=position_dict["u"][:, :self.input_seq_length],
                target_u=position_dict["u"][:, self.input_seq_length:],
                n_particles_per_trajectory=position_dict["position"].shape[0],
                particle_type=position_dict["particle_type"]
            )
        else:
            return Data(
            enc_pos=position_dict["position"][:, :self.input_seq_length],
            target_pos=position_dict["position"][:, self.input_seq_length:],
            n_particles_per_trajectory=position_dict["position"].shape[0],
            particle_type=position_dict["particle_type"]
            )

    def __len__(self):
        return self.num_samples


def get_dataset_name_from_path(path: str) -> str:
    """Infer the dataset name from the provided path.

    Variant 1:
        If the dataset directory contains {2|3}D_{ABC}, then the name is inferred as
        {abc2d|abc3d}. These names are based on the lagrangebench dataset directories:
        {2D|3D}_{TGV|RPF|LDC|DAM}_{num_particles_max}_{num_steps}every{sampling_rate}
        The shorter dataset names then become one of the following:
        {tgv2d|tgv3d|rpf2d|rpf3d|ldc2d|ldc3d|dam2d}
    Variant 2:
        If the condition {2|3}D_{ABC} is not met, the name is the dataset directory
    """

    dir = osp.basename(osp.normpath(path))
    name = re.search(r"(?:2D|3D)_[A-Z]{3}", dir)

    if name is not None:  # lagrangebench convention used
        name = name.group(0)
        name = f"{name.split('_')[1]}{name.split('_')[0]}".lower()
    else:
        warnings.warn(
            f"Dataset directory {dir} does not follow the lagrangebench convention. "
            "Valid name formats: {2D|3D}_{TGV|RPF|LDC|DAM}. Alternatively, you can "
            "specify the dataset name explicitly."
        )
        name = dir
    return name

   