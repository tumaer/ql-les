from pathlib import Path
from typing import Any, Dict, Tuple
from functools import partial
import time
import pickle
import os

import torch
import torch.nn as nn
from lightning import LightningModule
from torch import Tensor

from src.utils.data_utils import load_metadata
from src.utils.nbrs_utils import shift_fn, displ_fn
from src.models.components.sph import relax_wrapper


class BaseSimulator(nn.Module):
    """Base class with common methods for Eulerian and Lagrangian approaches."""

    def __init__(
        self,
        model_name,
        device,
        isotropic_norm,
        noise_std,
        dataset_path,
    ):
        super().__init__()
        self.model_name = model_name
        self._device = device
        self.isotropic_norm = isotropic_norm
        self.noise_std = noise_std

        self.metadata = load_metadata(Path(dataset_path))
        self._boundaries = self.metadata["bounds"]
        self._connectivity_radius = self.metadata["default_connectivity_radius"]
        self._case = self.metadata["case"]
        self._pbc = self.metadata["periodic_boundary_conditions"]
        self._effective_dt = self.metadata["dt"] * self.metadata["write_every"]
        self.dim = self.metadata["dim"]

    def set_metadata_device(self, device=None):
        """Extract normalization stats and more from metadata, and set device."""
        # TODO: make register buffer
        if device is None:
            device = self._device

        # box size
        self._boundaries = (
            torch.tensor(self.metadata["bounds"], requires_grad=False).float().to(device)
        )
        # Subtract the ends of the box to get its size (used for PBC)
        self._boundaries = self._boundaries[:, 1] - self._boundaries[:, 0]

        # v stats
        va_m = torch.FloatTensor(self.metadata["acc_mean"]).to(device)
        va_s = torch.FloatTensor(self.metadata["acc_std"]).to(device)
        vv_m = torch.FloatTensor(self.metadata["vel_mean"]).to(device)
        vv_s = torch.FloatTensor(self.metadata["vel_std"]).to(device)
        if self.isotropic_norm:
            va_m = va_m.mean() * torch.ones_like(va_m)
            va_s = va_s.mean() * torch.ones_like(va_s)
            vv_m = vv_m.mean() * torch.ones_like(vv_m)
            vv_s = vv_s.mean() * torch.ones_like(vv_s)
        self.normalization_stats = {
            "v_acceleration": {"mean": va_m, "std": torch.sqrt(va_s**2 + self.noise_std**2)},
            "v_velocity": {"mean": vv_m, "std": torch.sqrt(vv_s**2 + self.noise_std**2)},
        }

        # u stats
        try:
            ua_m = torch.FloatTensor(self.metadata["au_mean"]).to(device)
            ua_s = torch.FloatTensor(self.metadata["au_std"]).to(device)
            uu_m = torch.FloatTensor(self.metadata["u_mean"]).to(device)
            uu_s = torch.FloatTensor(self.metadata["u_std"]).to(device)
            if self.isotropic_norm:
                ua_m = ua_m.mean() * torch.ones_like(ua_m)
                ua_s = ua_s.mean() * torch.ones_like(ua_s)
                uu_m = uu_m.mean() * torch.ones_like(uu_m)
                uu_s = uu_s.mean() * torch.ones_like(uu_s)
            self.normalization_stats["u_acceleration"] = {
                "mean": ua_m,
                "std": torch.sqrt(ua_s**2 + self.noise_std**2),
            }
            self.normalization_stats["u_velocity"] = {
                "mean": uu_m,
                "std": torch.sqrt(uu_s**2 + self.noise_std**2),
            }
        except Exception:
            print("No u stats found.")

    def _norm(self, vel, key):
        """From displacement of positions `v=x1-x0` to a normal distribution."""
        # key = "ua/uu/va/vv" (uu: velocity u; vv: velocity v; ua/va: acceleration u/v)
        kv, ka = key
        ka = {"a": "acceleration", "v": "velocity", "u": "velocity"}[ka]

        stats = self.normalization_stats[f"{kv}_{ka}"]
        return (vel - stats["mean"]) / stats["std"]

    def _denorm(self, vel, key):
        """From a normal distribution to displacement of positions `v=x1-x0`."""
        kv, ka = key
        ka = {"a": "acceleration", "v": "velocity", "u": "velocity"}[ka]

        stats = self.normalization_stats[f"{kv}_{ka}"]
        return vel * stats["std"] + stats["mean"]

    def shift_fn(self, r, dr):
        """Shift the positions `r` by `dr`, respecting potential periodic boundaries."""
        return shift_fn(r, dr, self._boundaries, self._pbc)

    def displ_fn(self, r1, r2):
        """Compute the displacement between two position vectors `r1` and `r2`."""
        return displ_fn(r1, r2, self._boundaries, self._pbc)

    def _u2v(self, u):
        """Convert velocity `u` to displacement of positions `v = x1 - x0 = dt * u`."""
        return self._effective_dt * u

    def _v2u(self, v):
        """Convert displacement of positions `v` to velocity `u = (x1 - x0) / dt`."""
        return v / self._effective_dt

    def _sph_rlx(self, r, n_part_per_traj, is_tvf, dt_factor, num_steps):
        """Relax a point cloud in the exact same way as during dataset generation."""

        # Relax a point cloud using SPH without viscosity, but with transport vel.
        if not hasattr(self, "_relax_fn"):
            self._relax_fn = relax_wrapper(
                Nx=int(round(r.shape[0]) ** (1 / self.metadata["dim"])),
                dim=self.metadata["dim"],
                L=self._boundaries[0].item(),
                is_physical=True,
                u_ref=self.metadata["u_ref"],
                is_tvf=is_tvf,  # our relaxations always use tvf
                nu=0.0,  # relaxations assume zero velocity, so this term drops
                box=self._boundaries,
            )

        dt_factor = dt_factor
        v = 0.0
        # r_input = r.detach().clone()
        for i in range(num_steps):
            a_temp = self._relax_fn(
                r, n_part_per_traj
            )  # , verbose=True if (i==num_steps-1) else False)
            dr = (dt_factor * self.metadata["dt"]) ** 2 * a_temp
            r = shift_fn(r, dr)
            v += dr

        # The following line gives the same v up to >3 digits
        # v = self.displ_fn(r, r_input)  # v=x1-x0 as defined everywhere else in our code
        return v, v, r

    def _sph(self, r, n_part_per_traj, u):
        """Compute the right hand side of the NSE using SPH. Use for NeuralSPH."""

        if not hasattr(self, "_sph_fn"):
            self._sph_fn = relax_wrapper(
                Nx=int(round(r.shape[0]) ** (1 / self.metadata["dim"])),
                dim=self.metadata["dim"],
                L=self._boundaries[0].item(),
                is_physical=True,
                u_ref=self.metadata["u_ref"],
                is_tvf=False,  # our relaxations always use tvf
                nu=self.metadata["viscosity"],
                box=self._boundaries,
            )
        # pressure term std: 53; viscous term std: 0.059; tvf std: 90
        return self._sph_fn(r, n_part_per_traj, u)

    def forward(self):
        """Forward pass of the model."""
        pass


class BaseLitModule(LightningModule):
    """Base class for Lightning modules, providing common functionality."""

    def __init__(
        self,
        net: torch.nn.Module,
        accelerator: torch.device,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        visualize: Dict[str, Dict[str, Any]] = None,
        compile: bool = False,
        seed: int = 0,
        neuralsph: Dict[str, Any] = None,
        num_rollout_steps: int = 1,
        active_metrics: Dict[str, Any] = None,
        metric_space: Dict[str, str] = "norm",
    ) -> None:
        super().__init__()

        # Save hyperparameters and initialize the network
        # For model checkpointing
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net(device="cuda" if accelerator == "gpu" else "cpu")
        self.neuralsph = neuralsph

        self.num_rollout_steps = num_rollout_steps
        self.active_metrics = active_metrics
        self.metric_space = metric_space
        self.visualize = visualize
        self.trajectory_idx = 0

    def model_step(self):
        """Batch in, loss out."""
        raise NotImplementedError("model_step method must be implemented.")

    def validation_step(self):
        """Batch in, no return. Log the loss and potentially write trajectory."""
        raise NotImplementedError("validation_step method must be implemented.")

    def training_step(self, batch: Any) -> torch.Tensor:
        """Extend model_step to include logging the loss."""
        # Evaluate the model on the training batch and calculate the loss
        loss = self.model_step(batch)
        # Log metrics
        self.log("train/loss", loss, prog_bar=True, batch_size=batch.batch_size, on_epoch=True)
        return loss

    def on_validation_start(self) -> None:
        """Reset trajectory index for writing metrics and trajectories."""
        self.trajectory_idx = 0
        self.metrics_dump = {}

    def on_test_start(self):
        """Reset trajectory index for writing metrics and trajectories."""
        self.net.set_metadata_device(self.net._device)
        self.trajectory_idx = 0
        self.metrics_dump = {}

    def on_test_end(self):
        """Write the metrics to a file after testing."""
        # write full metrics to file
        metrics_path = os.path.join(
            self.visualize.vis_test.rollout_dir,
            f"metrics_{time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())}.pkl",
        )
        print(f"Writing metrics to {metrics_path}")
        with open(metrics_path, "wb") as f:
            pickle.dump(self.metrics_dump, f)

    def test_step(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, Tensor]:
        """Perform a single test step, using the forward method to infer positions."""
        # Evaluate the trajectory rollout
        self.validation_step(batch, testing=True)

    def on_fit_start(self) -> None:
        """Set the device of the metadata variables."""
        self.net.set_metadata_device(self.net._device)

    def on_train_epoch_start(self):
        """Get the learning rate at the start of each epoch."""
        # TODO: check whether needed
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        return lr

    def setup(self, stage: str) -> None:
        """Setup model for training or evaluation."""

        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)
            print("Model compiled!")

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the optimizer and learning rate scheduler."""
        # To increment the learning rate at every gradient descent step
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            if (type(self.hparams.scheduler) is partial) and (
                self.hparams.scheduler.func.__name__
                in ["LinearWarmupCosineAnnealingLR", "ExpDecayLR", "StepLR"]
            ):
                interval = "step"
                print("########### step interval ############")
            else:
                interval = "epoch"
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": interval,
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
