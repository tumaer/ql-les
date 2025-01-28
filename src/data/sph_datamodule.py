import os
from typing import Optional
from torch.utils.data import random_split
from lightning import LightningDataModule
from torch_geometric.loader import DataLoader as PyGDataLoader
from src.data.components.h5dataset import H5Dataset


class SPHDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/2D_TGV_2500_10kevery100",
        batch_size: int = 1,
        input_seq_length: int = 6,
        max_pushforward_steps: int = 0,
        max_rollout_steps: int = 0,
        nl_backend: str = "jaxmd_vmap",
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.input_seq_length = input_seq_length
        self.max_rollout_steps = max_rollout_steps
        self.max_pushforward_steps = max_pushforward_steps
        self.nl_backend = nl_backend
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self):
        pass
        
    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train_dataset = H5Dataset(split="train",
                                           dataset_path=self.data_dir,
                                           input_seq_length=self.input_seq_length,
                                           extra_seq_length=self.max_pushforward_steps,
                                           nl_backend=self.nl_backend)
            self.val_dataset = H5Dataset(split="valid", 
                                         dataset_path=self.data_dir, 
                                         input_seq_length=self.input_seq_length, 
                                         extra_seq_length=self.max_rollout_steps, 
                                         nl_backend=self.nl_backend)
        elif stage == "test":
            self.test_dataset = H5Dataset(split="test", 
                                         dataset_path=self.data_dir, 
                                         input_seq_length=self.input_seq_length, 
                                         extra_seq_length=self.max_rollout_steps, 
                                         nl_backend=self.nl_backend)
        else:
            raise ValueError(f"Stage {stage} not recognized.")
        
    def train_dataloader(self) -> PyGDataLoader:
        return PyGDataLoader(
            dataset=self.train_dataset,
            batch_size= self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> PyGDataLoader:
        return PyGDataLoader(
            dataset=self.val_dataset,
            batch_size= self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> PyGDataLoader:
        return PyGDataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
        