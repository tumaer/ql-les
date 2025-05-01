from typing import Optional
from lightning import LightningDataModule
from torch_geometric.loader import DataLoader as PyGDataLoader
from src.data.components.h5dataset import H5Dataset
from torch.utils.data import Subset


class SPHDataModule(LightningDataModule):
    """DataModule for SPH datasets."""

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
        shuffle: bool = True,
        limit_train_batches: Optional[float] = None,
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
        self.shuffle = shuffle
        self.limit_train_batches = limit_train_batches

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Prepare the data for training, validation, and testing."""
        # TODO: is this needed?
        pass

    def setup(self, stage: Optional[str] = None):
        """Setup the datasets for training, validation, and testing."""
        if stage == "fit":
            self.train_dataset = H5Dataset(
                split="train",
                dataset_path=self.data_dir,
                input_seq_length=self.input_seq_length,
                extra_seq_length=self.max_pushforward_steps,
                nl_backend=self.nl_backend,
            )
            if self.limit_train_batches is not None:
                limited_len = self.batch_size * self.limit_train_batches
                indices = list(range(limited_len))
                self.train_dataset = Subset(self.train_dataset, indices)
            self.val_dataset = H5Dataset(
                split="valid",
                dataset_path=self.data_dir,
                input_seq_length=self.input_seq_length,
                extra_seq_length=self.max_rollout_steps,
                nl_backend=self.nl_backend,
            )
        elif stage == "test":
            self.test_dataset = H5Dataset(
                split="test",
                dataset_path=self.data_dir,
                input_seq_length=self.input_seq_length,
                extra_seq_length=self.max_rollout_steps,
                nl_backend=self.nl_backend,
            )
        else:
            raise ValueError(f"Stage {stage} not recognized.")

    def train_dataloader(self) -> PyGDataLoader:
        """Create the training data loader."""
        return PyGDataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=self.shuffle,
        )

    def val_dataloader(self) -> PyGDataLoader:
        """Create the validation data loader."""
        return PyGDataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> PyGDataLoader:
        """Create the test data loader."""
        return PyGDataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )
