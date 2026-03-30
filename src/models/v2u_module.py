from typing import Any, Dict

import torch
from torch import Tensor
from src.models.components.gns import EncodeProcessDecode
from src.models.base_module import BaseSimulator, BaseLitModule


class V2USimulator(BaseSimulator):
    """Class for the V2U model, which predicts acceleration from velocity and position."""

    def __init__(
        self,
        model_name: str,
        latent_dim: int,
        num_message_passing_steps: int,
        noise_std: float,
        metadata_path: str,
        device: str,
        isotropic_norm=False,
        mlp_num_layers: int = 1,
        mlp_hidden_dim: int = 128,
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            isotropic_norm=isotropic_norm,
            noise_std=noise_std,
            metadata_path=metadata_path,
        )
        self._num_particle_types = 1

        if model_name == "gns":
            self.v2u_gnn = EncodeProcessDecode(
                node_in=self.dim,
                node_out=self.dim,
                edge_in=self.dim + 1,
                latent_dim=latent_dim,
                num_message_passing_steps=num_message_passing_steps,
                mlp_num_layers=mlp_num_layers,
                mlp_hidden_dim=mlp_hidden_dim,
                alpha_u=0,
            )

    def set_metadata_device(self, device=None):
        """Akin to BaseSimulator.set_metadata_device but for the normalization stats for V2U."""
        super().set_metadata_device(device)
        if device is None:
            device = self._device

        if "avu_mean" not in self.metadata or "avu_std" not in self.metadata:
            raise KeyError("Metadata must contain 'avu_mean' and 'avu_std' for V2U normalization.")

        avu_m = torch.FloatTensor(self.metadata["avu_mean"]).to(device)
        avu_s = torch.FloatTensor(self.metadata["avu_std"]).to(device)
        if self.isotropic_norm:
            avu_m = avu_m.mean() * torch.ones_like(avu_m)
            avu_s = avu_s.mean() * torch.ones_like(avu_s)

        self.normalization_stats["vu_acceleration"] = {
            "mean": avu_m,
            "std": torch.sqrt(avu_s**2 + self.noise_std**2),
        }

    def _norm_avu(self, avu):
        """Normalize the avu target from physical units (u) to a normal distribution."""
        stats = self.normalization_stats["vu_acceleration"]
        return (avu - stats["mean"]) / stats["std"]

    def _denorm_avu(self, avu):
        """Denormalize the avu from the GNN output space back to physical units."""
        stats = self.normalization_stats["vu_acceleration"]
        return avu * stats["std"] + stats["mean"]

    def predict_accelerations(
        self,
        next_position,
        position_sequence,
        next_u_velocity,
        n_particles_per_trajectory,
        pbc=True,
    ):
        """Predict acceleration using GNS and also return the target acceleration."""
        # Evaluate velocity from two positions:
        next_v_velocity = self.displ_fn(next_position, position_sequence[:, -1])

        # Compute and normalize avu target: avu = next_u - v2u(next_v)
        avu_target_norm = self._norm_avu(next_u_velocity - self._v2u(next_v_velocity))

        # Construct the input graph to the gnn
        node_features, edge_index, e_features = self._build_graph_from_raw(
            position_sequence,
            n_particles_per_trajectory,
            pbc=pbc,
            node_features_type=[],
            connectivity_on_nth_to_last=-2,
        )
        node_features["v_flat_velocity_sequence"] = self._norm(next_v_velocity, "vv")

        # Run gnn
        avu_pred_norm = self.v2u_gnn(node_features, edge_index, e_features)
        return avu_pred_norm, avu_target_norm


class V2ULitModule(BaseLitModule):
    """Class for training the V2U model."""

    def __init__(
        self,
        net: torch.nn.Module,
        accelerator: torch.device,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        visualize: Any = None,
        compile: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(
            net=net,
            accelerator=accelerator,
            optimizer=optimizer,
            scheduler=scheduler,
            compile=compile,
            seed=seed,
            metric_space=None,
        )
        self.save_hyperparameters(logger=False)
        self.net = net(device="cuda" if accelerator == "gpu" else "cpu")
        self.visualize = visualize
        self.seed = seed

    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        return self.net.predict_accelerations(**features)

    def model_step(self, batch: Dict[str, Any]) -> torch.Tensor:
        """During training."""
        # Run GNN
        features = {
            "next_position": batch.target_pos.squeeze(1),  # (N, D)
            "position_sequence": batch.enc_pos,  # (N, 2, D)
            "next_u_velocity": batch.target_u.squeeze(1),  # (N, D)
            "n_particles_per_trajectory": batch.n_particles_per_trajectory,
            "pbc": any(self.net._pbc),
        }
        pred_norm, target_norm = self.forward(features)

        # Compute loss
        loss = ((pred_norm - target_norm) ** 2).sum(dim=-1).mean()
        return loss

    def validation_step(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """During inference."""
        loss = self.model_step(batch)
        split = "test" if (("testing" in kwargs) and kwargs["testing"]) else "val"
        self.log(f"{split}/loss", loss, prog_bar=True, batch_size=batch.batch_size, on_epoch=True)
