# src.models.gns_module.py
from typing import Any, Dict, Tuple
import warnings

import torch
from torch import Tensor, nn
from neuraloperator.neuralop.models.gino import GINO
from neuraloperator.neuralop.models.fno import FNO

from src.models.base_module import BaseSimulator, BaseLitModule
from src.models.components.gns import EncodeProcessDecode
from src.utils.metrics import particle_mse
from src.utils.eval_utils import eval_rollout
from src.utils.nbrs_utils import gen_grid_points
from src.utils.interpolate import Interpolator


class InterpFNO(nn.Module):
    """FNO mapping from any input point cloud to any output point cloud."""

    def __init__(
        self,
        is_periodic,
        domain_size,
        dim,
        dx,
        condition="knn",
        k=None,
        cutoff_factor=None,
        kernel="xsqinv",
        **fno_kwargs,
    ):
        super().__init__()
        self.fno = FNO(**fno_kwargs)

        self.interpolate = Interpolator(
            is_periodic=is_periodic,
            domain_size=domain_size,
            dim=dim,
            dx=dx,
            condition=condition,
            k=k,
            cutoff_factor=cutoff_factor,
            kernel=kernel,
        )
        # grid = gen_grid_points(query_resolution, self.domain_size)
        # self.register_buffer("grid", torch.tensor(grid.reshape(-1, self.dim)))

    def forward(
        self, input_geom, latent_queries, output_queries, x, x_grid=None, return_x_grid=False
    ):
        """Forward pass of the InterpFNO model.

        Args:
            input_geom (torch.Tensor): Input geometry (N, D).
            latent_queries (torch.Tensor): Latent queries (G, G, D).
            output_queries (torch.Tensor): Output queries (M, D).
            x (torch.Tensor): Input features (B, N, FNO_IN_CHANNELS).
            x_grid (torch.Tensor, optional): Grid features (B, D, N,...N). If x_grid not None, run rollout purely Eulerian.

        Returns:
            if x_grid is None:
                torch.Tensor: Output features (B, M, FNO_OUT_CHANNELS).
            else:
                (torch.Tensor, torch.Tensor): Output features (B, M, FNO_OUT_CHANNELS), grid features (B, D, N,...N).
        """

        # Interpolate velocities from points to grid
        r_latent_queries = latent_queries.reshape(-1, input_geom.shape[-1])

        if x_grid is None:
            x = x[0]  # remove batch dimension
            x_grid = self.interpolate(r=input_geom, r_target=r_latent_queries, f=x)

            # Reshape grid to match FNO input
            x_grid = x_grid.reshape(latent_queries.shape[:-1] + x.shape[-1:])
            # below: (N,... N, D) -> (B, D, N,... N)
            x_grid = x_grid.permute(*torch.arange(x_grid.ndim - 1, -1, -1))[None, ...]

        # Run FNO on grid
        x_grid_out = self.fno(x_grid)
        # Reverse reshaping
        # next two lines: (B, D, N,... N) -> (N,... N, D)
        x_grid = x_grid_out.squeeze(0)
        x_grid = x_grid.permute(*torch.arange(x_grid.ndim - 1, -1, -1))
        x_grid = x_grid.reshape(-1, x_grid.shape[-1])  # (N,... N, D) -> (N*N..., D)
        # Interpolate velocities from grid to points
        x = self.interpolate(r=r_latent_queries, r_target=output_queries, f=x_grid)

        if return_x_grid:  # rollout mode of InterpFNO
            return x, x_grid_out
        else:
            # TODO: this double smoothing procedure inevitably kills high frequencies
            #   Does this work at all? FNO has to implicitly compensate for the smoothing
            return x[None, ...]  # add batching dimension back in
        # else: # inference mode
        #     return {"x": x[None, ...], "x_grid": x_grid_out}  # add batching dimension back in


class GINOSimulator(BaseSimulator):
    """A GINO simulator for SPH simulations."""

    def __init__(
        self,
        model_name: str,
        device: str,
        isotropic_norm: bool,
        noise_std: float,
        metadata_path: str,
        # v_solver: str, # "rlx", "neural", "hybrid"
        return_x_grid: bool = False,  # applies only to InterpFNO and during rollout
        **gino_kwargs,
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            isotropic_norm=isotropic_norm,
            noise_std=noise_std,
            metadata_path=metadata_path,
        )

        box_size = [x[1] for x in self._boundaries]
        query_resolution = gino_kwargs.get("query_resolution", [64] * self.dim)
        self.latent_points = torch.tensor(gen_grid_points(query_resolution, box_size)).to(
            self._device
        )
        self.return_x_grid = return_x_grid

        if model_name == "gino":
            print("Warning: Tested only on batch size 1!")
            assert return_x_grid is False, "return_x_grid must be False for GINO model"

            self.network = GINO(
                in_channels=self.dim,  # input past u
                out_channels=self.dim,  # directly output u
                gno_coord_dim=self.dim,
                gno_radius=self._connectivity_radius,
                is_periodic=any(self._pbc),
                domain_size=box_size,
                **gino_kwargs,
            )
        elif model_name == "interp_fno":
            print("Warning: Tested only on batch size 1!")

            # remove the prefix "fno_" from the keys in gino_kwargs
            fno_kwargs = {k[4:]: v for k, v in gino_kwargs.items() if k.startswith("fno_")}
            nbrs_kwargs = {k[5:]: v for k, v in gino_kwargs.items() if k.startswith("nbrs_")}
            self.network = InterpFNO(
                # in_channels=self.dim,  # input past u
                out_channels=self.dim,  # directly output u
                is_periodic=any(self._pbc),
                domain_size=box_size,
                dim=self.dim,
                dx=self.metadata["dx"],
                **nbrs_kwargs,
                **fno_kwargs,
            )
        elif model_name == "gns":
            self.network = EncodeProcessDecode(
                node_in=self.dim,  # input past u
                node_out=self.dim,  # output acceleration_u
                edge_in=self.dim + 1,  # displacement vector and its length
                latent_dim=gino_kwargs["latent_dim"],
                num_message_passing_steps=gino_kwargs["num_message_passing_steps"],
                mlp_num_layers=gino_kwargs["mlp_num_layers"],
                mlp_hidden_dim=gino_kwargs["mlp_hidden_dim"],
                alpha_u=0.0,
            )
            self._num_particle_types = 1  # for compatibility with _build_graph_from_raw
        else:
            raise ValueError(f"Model name {model_name} not recognized.")

    def predict_positions(
        self,
        current_positions: Tensor,
        u_velocity: Tensor,
        n_particles_per_trajectory: Tensor,
        particle_types: Tensor,
        pbc,
        **kwargs,
    ):
        if pbc:
            current_positions = current_positions % self._boundaries

        assert hasattr(self, "neuralsph"), "neuralsph must be defined in the model."

        most_recent_position = current_positions[:, -1]
        most_recent_u_velocity = u_velocity[:, -1]

        # Evolve particles
        new_pos_temp = self.shift_fn(most_recent_position, self._u2v(most_recent_u_velocity))
        av_rlx, v_rlx, new_position = self._sph_rlx(
            new_pos_temp,
            n_particles_per_trajectory,
            is_tvf=self.metadata["rlx_is_tvf"],
            dt_factor=self.metadata["rlx_dt_factor"],
            num_steps=self.metadata["rlx_num_steps"],
        )
        if self.neuralsph["num_steps"] > 0:
            _, _, new_position = self._sph_rlx(
                new_position,
                n_particles_per_trajectory,
                is_tvf=self.neuralsph["is_tvf"],
                dt_factor=self.neuralsph["dt_factor"],
                num_steps=self.neuralsph["num_steps"],
            )

        # Evolve u
        most_recent_u_velocity_norm = self._norm(most_recent_u_velocity, "uu")
        if self.model_name in ["gino", "interp_fno"]:
            new_u_velocity_norm = self.network(
                input_geom=most_recent_position,  # (N, D)
                latent_queries=self.latent_points,  # (G, G, D)
                output_queries=new_position,  # (M, D)
                x=most_recent_u_velocity_norm[None, ...],  # (B, N, FNO_IN_CHANNELS)
                x_grid=kwargs.get("x_grid", None),  # (B, D, N,...N)
                return_x_grid=self.return_x_grid,
            )
        elif self.model_name == "gns":
            node_features, edge_index, edge_features = self._build_graph_from_raw(
                position_sequence=most_recent_position.unsqueeze(1),  # (N, 1, D)
                n_particles_per_trajectory=n_particles_per_trajectory,  # (B,)
                u_velocity=most_recent_u_velocity_norm,  # (N, D)
                node_features_type=["u"],
            )
            u_acc = self.network(node_features, edge_index, edge_features)
            new_u_velocity = most_recent_u_velocity + self._denorm(u_acc, "ua")
            return new_position, new_u_velocity

        if self.return_x_grid:  # optional with interp_fno mode
            new_u_velocity_norm, u_grid_norm = new_u_velocity_norm
            new_u_velocity = self._denorm(new_u_velocity_norm, "uu")
            u_grid = self._denorm(u_grid_norm, "uu")
            return new_position, new_u_velocity, u_grid
        else:
            # (B, M, FNO_OUT_CHANNELS); add and remove batching with [None, ...] and [0]
            new_u_velocity_norm = new_u_velocity_norm[0]
            new_u_velocity = self._denorm(new_u_velocity_norm, "uu")
            return new_position, new_u_velocity


class GINOLitModule(BaseLitModule):
    """A LightningModule for training a Geometry-Informed Neural Operator (GINO)."""

    def __init__(
        self,
        net: torch.nn.Module,
        accelerator: torch.device,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        visualize: Dict[str, Any] = None,
        loss_fn: torch.nn.Module = torch.nn.MSELoss(),
        compile: bool = False,
        seed: int = 0,
        neuralsph: Dict[str, Any] = None,
        num_rollout_steps: int = 1,
        active_metrics: Dict[str, Any] = None,
        metric_space: Dict[str, str] = "norm",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            net=net,
            accelerator=accelerator,
            optimizer=optimizer,
            scheduler=scheduler,
            visualize=visualize,
            compile=compile,
            seed=seed,
            neuralsph=neuralsph,
            num_rollout_steps=num_rollout_steps,
            active_metrics=active_metrics,
            metric_space=metric_space,
        )

        self.loss_fn = loss_fn

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Remove `_metadata` key from the checkpoint.
        See https://github.com/neuraloperator/neuraloperator/pull/493
        """
        # metadata = checkpoint["state_dict"]["_metadata"]
        state_dict = checkpoint["state_dict"]
        metadata = state_dict.pop("_metadata", None)

        if metadata is not None:
            saved_version = metadata.get("_version", None)
            if saved_version is None:
                warnings.warn(
                    f"Saved instance of {self.__class__} has no stored version attribute."
                )
            if saved_version != self.net.network._version:
                warnings.warn(
                    f"Attempting to load a {self.__class__} of version {saved_version},"
                    f"But current version of {self.__class__} is {saved_version}"
                )

    def model_step(self, batch: Any) -> Tensor:
        """Forward pass and loss calculation."""

        u_in = batch.enc_u[:, -1]
        u_in_norm = self.net._norm(u_in, "uu")
        if self.net.noise_std > 0:
            u_in_norm += torch.randn(u_in.shape, device=u_in.device) * self.net.noise_std

        target_u_norm = self.net._norm(batch.target_u.squeeze(), "uu")

        if self.net.model_name == "gino":
            # Train GINO on `u`
            u_pred_norm = self.net.network(
                input_geom=batch.enc_pos[:, -1],  # (N, D)
                latent_queries=self.net.latent_points,  # (G, G, D)
                output_queries=batch.target_pos[:, 0],  # (M, D)
                x=u_in_norm[None, ...],  # (B, N, FNO_IN_CHANNELS)
            )[0]  # (B, M, FNO_OUT_CHANNELS); add and remove batching with [None, ...] and [0]
        elif self.net.model_name == "interp_fno":
            # Train InterpFNO on `u`
            _, u_pred_norm = self.net.network(
                input_geom=batch.enc_pos[:, -1],  # (N, D)
                latent_queries=self.net.latent_points,  # (G, G, D)
                output_queries=batch.target_pos[:, 0],  # (M, D)
                x=u_in_norm[None, ...],  # (B, N, FNO_IN_CHANNELS)
                return_x_grid=True,
            )
            # interpolate target to same grid as FNO grid
            r_latent_queries = self.net.latent_points.reshape(-1, self.net.dim)
            target_u_norm = self.net.network.interpolate(
                r=batch.target_pos[:, 0],
                r_target=r_latent_queries,
                f=target_u_norm,
            )
            # reshape predictions from grid to points
            u_pred_norm = u_pred_norm.squeeze(0)
            u_pred_norm = u_pred_norm.permute(*torch.arange(u_pred_norm.ndim - 1, -1, -1))
            # (N,... N, D) -> (N*N..., D)
            u_pred_norm = u_pred_norm.reshape(-1, u_pred_norm.shape[-1])
        elif self.net.model_name == "gns":
            node_features, edge_index, edge_features = self.net._build_graph_from_raw(
                position_sequence=batch.enc_pos[:, -1:],  # (N, 1, D)
                n_particles_per_trajectory=batch["n_particles_per_trajectory"],  # (B,)
                u_velocity=u_in_norm,  # (N, D)
                node_features_type=["u"],
            )
            u_acc_norm = self.net.network(node_features, edge_index, edge_features)
            u_pred_norm = self.net._norm(u_in + self.net._denorm(u_acc_norm, "ua"), "uu")

        non_kinematic_mask = (batch.particle_types != 3).clone().detach()
        loss_u = particle_mse(u_pred_norm, target_u_norm, non_kinematic_mask)
        return loss_u

    def validation_step(self, batch: Tuple[Tensor, Tensor], **kwargs) -> Dict[str, Tensor]:
        """Perform a single validation step, using the forward method to infer positions."""
        # ROLLOUT EVALUATION
        # Evaluate validation loss, meaning a N-Step rollout
        is_test = ("testing" in kwargs) and kwargs["testing"]
        split = "test" if is_test else "val"
        self.net.neuralsph = self.neuralsph[split]
        loss = eval_rollout(
            batch=batch,
            simulator=self.net,
            metadata=self.net.metadata,
            num_rollout_steps=self.num_rollout_steps,
            pbc=any(self.net._pbc),
            device=self.net._device,
            active_metrics=self.active_metrics,
            vis_config=self.visualize[f"vis_{split}"],
            trajectory_idx=self.trajectory_idx,
            u_vel=True,
            metric_space=self.metric_space.test if is_test else self.metric_space.val,
            interpolate=self.metrics_interpolate,
        )

        kwargs_log = {"prog_bar": True, "on_epoch": True, "batch_size": batch.batch_size}
        position_loss, u_vel_loss = loss
        self.log(f"{split}/loss", u_vel_loss.mean(), **kwargs_log)
        self.log(f"{split}/u_loss_grid", u_vel_loss.mean(), **kwargs_log)

        self.log(f"{split}/v_loss", position_loss["mse"].mean(), **kwargs_log)
        self.log(f"{split}/loss_ekin", position_loss["e_kin"]["mse"].mean(), **kwargs_log)  # v
        if "mse_pos" in position_loss:
            self.log(f"{split}/mse_pos", position_loss["mse_pos"].mean(), **kwargs_log)

        self.metrics_dump[self.trajectory_idx] = loss
        self.trajectory_idx += 1
