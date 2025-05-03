# src.models.gns_module.py
from typing import Any, Dict, Tuple
import warnings

import torch
from torch import Tensor, nn
from neuralop.models.gino import GINO
from neuralop.models.fno import FNO
from torch_scatter import scatter_add

from src.models.components.sph import QuinticKernel
from src.models.base_module import BaseSimulator, BaseLitModule
from src.utils.metrics import particle_mse
from src.utils.eval_utils import eval_rollout
from src.utils.nbrs_utils import gen_grid_points, nearest, displ_fn


class XsqinvKernel:
    """The 1/x^2 kernel function from PointNet++."""

    def __init__(self, h, dim=3):
        self._one_over_h = 1.0 / h
        self._normalized_cutoff = 3.0  # arbitrarily chosen
        self.cutoff = self._normalized_cutoff * h

    def w(self, r):
        return 1 / (r * self._one_over_h) ** 2


class InterpFNO(nn.Module):
    """FNO mapping from any input point cloud to any output point cloud."""

    def __init__(
        self,
        is_periodic,
        domain_size,
        dim,
        dx,
        query_resolution=[64, 64],
        condition="knn",
        k=None,
        cutoff=None,
        kernel="1/x^2",
        **fno_kwargs,
    ):
        super().__init__()
        self.fno = FNO(**fno_kwargs)

        # Dataset metadata
        self.is_periodic = is_periodic  # e.g. True
        self.register_buffer("domain_size", torch.tensor(domain_size))  # e.g. [1.0, 1.0]
        self.dim = dim  # e.g. 2

        # Neighbors search algorithm
        self.condition = condition
        if self.condition == "radius":
            assert cutoff is not None, "cutoff must be provided for radius condition"
        elif self.condition == "knn":
            assert k is not None, "k must be provided for knn condition"
        self.k = k
        self.cutoff = cutoff

        if kernel == "1/x^2":
            self.kernel_fn = XsqinvKernel(dx)
        elif kernel == "quintic":
            self.kernel_fn = QuinticKernel(2 / 3 * dx, dim=self.dim)

        grid = gen_grid_points(query_resolution, self.domain_size)
        self.register_buffer("grid", torch.tensor(grid.reshape(-1, self.dim)))

    def displ_fn(self, r1, r2):
        return displ_fn(r1, r2, self.domain_size, self.is_periodic)

    def _interpolate(self, r, r_target, f):
        """Shepard interpolation between two point clouds."""
        # Compute distances
        # TODO: works only for batch size 1 for now
        n_part_per_traj = torch.tensor([r.shape[0]], dtype=torch.int64, device=r.device)
        n_pptr_query = torch.tensor([r_target.shape[0]], dtype=torch.int64, device=r.device)
        edge_index = nearest(
            r,
            n_part_per_traj,
            self.is_periodic,
            self.domain_size,
            condition=self.condition,
            k=self.k,
            cutoff=self.cutoff,
            query=r_target,
            n_pptr_query=n_pptr_query,
        )
        i_s, j_s = edge_index
        r_i, r_j = r_target[i_s], r[j_s]
        dr_ij = self.displ_fn(r_i, r_j)
        dist = torch.norm(dr_ij, dim=-1)

        # Compute weights
        w_dist = self.kernel_fn.w(dist)
        w_dist_sum = scatter_add(w_dist, i_s, dim=0, dim_size=len(r_target))

        # Interpolate
        num_targets = r_target.shape[-2]  # Shape (B, N, D)
        f_interp = scatter_add(
            w_dist[:, None] * f[j_s], i_s, dim=0, dim_size=num_targets
        )  # TODO: check j_s
        f_interp /= w_dist_sum[:, None]
        # assert no nan or inf numbers
        assert torch.all(torch.isfinite(f_interp)), "Interpolation resulted in NaN or Inf values"
        assert torch.all(w_dist_sum > 0), "Interpolation resulted in zero weights"

        # def plt_interp(ind, r, r_target, f, f_interp, i_s, j_s):
        #     # r_target = r_target.cpu().numpy()
        #     # r = r.cpu().numpy()
        #     # f = f.cpu().numpy()
        #     # f_interp = f_interp.cpu().numpy()
        #     # i_s = i_s.cpu().numpy()
        #     # j_s = j_s.cpu().numpy()
        #     import matplotlib.pyplot as plt
        #     fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        #     axs[0].scatter(r[:,0], r[:,1], c=f[:,0], s=1)
        #     axs[1].scatter(r[:,0], r[:,1], c=f[:,0], s=1)
        #     axs[1].scatter(r_target[:,0], r_target[:,1], c=f_interp[:,0], s=1, marker='x')
        #     # for particle 0 in r_target, plot edges to corresponding particles in r
        #     for i in range(len(i_s)):
        #         if i_s[i] == ind:
        #             axs[1].plot([r_target[i_s[i],0], r[j_s[i],0]], [r_target[i_s[i],1], r[j_s[i],1]], 'k-', lw=0.5)
        #     axs[0].set_title("Input geom")
        #     axs[1].set_title("Grid geom")
        #     plt.tight_layout()
        #     fig.savefig("interp_fno_input_grid.png", dpi=600)
        #     plt.close(fig)
        # plt_interp(2, r, r_target, f, f_interp, i_s, j_s)

        return f_interp

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
        r_latent_queries = latent_queries.reshape(-1, self.dim)

        if x_grid is None:
            x = x[0]  # remove batch dimension
            x_grid = self._interpolate(r=input_geom, r_target=r_latent_queries, f=x)

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
        x = self._interpolate(r=r_latent_queries, r_target=output_queries, f=x_grid)

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
        dataset_path: str,
        # v_solver: str, # "rlx", "neural", "hybrid"
        return_x_grid: bool = False,  # applies only to InterpFNO and during rollout
        **gino_kwargs,
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            isotropic_norm=isotropic_norm,
            noise_std=noise_std,
            dataset_path=dataset_path,
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

            self.gino = GINO(
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
            self.gino = InterpFNO(
                # in_channels=self.dim,  # input past u
                out_channels=self.dim,  # directly output u
                query_resolution=gino_kwargs["query_resolution"],
                is_periodic=any(self._pbc),
                domain_size=box_size,
                dim=self.dim,
                dx=self.metadata["dx"],
                **nbrs_kwargs,
                **fno_kwargs,
            )
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
        most_recent_u_velocity = self._norm(most_recent_u_velocity, "uu")
        new_u_velocity_norm = self.gino(
            input_geom=most_recent_position,  # (N, D)
            latent_queries=self.latent_points,  # (G, G, D)
            output_queries=new_position,  # (M, D)
            x=most_recent_u_velocity[None, ...],  # (B, N, FNO_IN_CHANNELS)
            x_grid=kwargs.get("x_grid", None),  # (B, D, N,...N)
            return_x_grid=self.return_x_grid,
        )
        if self.return_x_grid:  # optional with interp_fno mode
            new_u_velocity_norm, u_grid = new_u_velocity_norm
            new_u_velocity = self._denorm(new_u_velocity_norm, "uu")
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
            if saved_version != self.net.gino._version:
                warnings.warn(
                    f"Attempting to load a {self.__class__} of version {saved_version},"
                    f"But current version of {self.__class__} is {saved_version}"
                )

    def model_step(self, batch: Any) -> Tensor:
        # Train GINO on `u`
        u_pred = self.net.gino(
            input_geom=batch.enc_pos[:, -1],  # (N, D)
            latent_queries=self.net.latent_points,  # (G, G, D)
            output_queries=batch.target_pos[:, 0],  # (M, D)
            x=batch.enc_u[:, -1][None, ...],  # (B, N, FNO_IN_CHANNELS)
        )[0]  # (B, M, FNO_OUT_CHANNELS); add and remove batching with [None, ...] and [0]

        non_kinematic_mask = (batch.particle_types != 3).clone().detach()
        target_u_norm = self.net._norm(batch.target_u.squeeze(), "uu")
        loss_u = particle_mse(u_pred, target_u_norm, non_kinematic_mask)
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
        )

        kwargs_log = {"prog_bar": True, "on_epoch": True, "batch_size": batch.batch_size}
        position_loss, u_vel_loss = loss
        self.log(f"{split}/loss", position_loss["mse"].mean() + u_vel_loss.mean(), **kwargs_log)
        self.log(f"{split}/u_loss", u_vel_loss.mean(), **kwargs_log)

        self.log(f"{split}/v_loss", position_loss["mse"].mean(), **kwargs_log)
        self.log(
            f"{split}/loss_ekin", position_loss["e_kin"]["mse"].mean(), **kwargs_log
        )  # only shifting velocity currently
        if "mse_pos" in position_loss:
            self.log(f"{split}/mse_pos", position_loss["mse_pos"].mean(), **kwargs_log)

        self.metrics_dump[self.trajectory_idx] = loss
        self.trajectory_idx += 1
