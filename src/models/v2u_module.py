from typing import Any, Dict

import torch
from torch import Tensor
from src.models.components.gns import EncodeProcessDecode
from src.models.base_module import BaseSimulator, BaseLitModule


class V2USimulator(BaseSimulator):
    def __init__(
        self,
        model_name: str,
        latent_dim: int,
        num_message_passing_steps: int,
        noise_std: float,
        dataset_path: str,
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
            dataset_path=dataset_path,
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

        # Compute the target normalized acceleration -> normalization with "ua"/"va" overshoots
        a_u_target = next_u_velocity - self._v2u(next_v_velocity)  # TODO: consider u over ua
        # a_u_target *= 5  # manually tuned number for normalization, TODO: remove this

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
        a_u_pred = self.v2u_gnn(node_features, edge_index, e_features)
        # print(f"target_mean/std: {a_u_target.mean():.3f}/{a_u_target.std():.3f}, pred_mean/std: {a_u_pred.mean():.3f}/{a_u_pred.std():.3f}")

        return a_u_pred, a_u_target  # TODO: implement a training step for this


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
        # Run GNN
        features = {
            "next_position": batch.target_pos.squeeze(1),  # (N, D)
            "position_sequence": batch.enc_pos,  # (N, 2, D)
            "next_u_velocity": batch.target_u.squeeze(1),  # (N, D)
            "n_particles_per_trajectory": batch.n_particles_per_trajectory,
            "pbc": any(self.net._pbc),
        }
        pred, target = self.forward(features)

        # Compute loss
        loss = ((pred - target) ** 2).sum(dim=-1).mean()
        return loss

    def validation_step(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        loss = self.model_step(batch)
        split = "test" if (("testing" in kwargs) and kwargs["testing"]) else "val"
        self.log(f"{split}/loss", loss, prog_bar=True, batch_size=batch.batch_size, on_epoch=True)
