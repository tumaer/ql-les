# src.models.gns_module.py
from typing import Any, Dict, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from src.models.components.gns import (
    get_random_walk_noise_for_position_sequence,
    EncodeProcessDecode,
    time_diff,
)
from src.models.components.lles import LLES
from src.models.components.segnn import SEGNN, WeightBalancedIrreps, Irreps
from src.models.base_module import BaseSimulator, BaseLitModule
from src.utils.train_utils import pushforward_sample_steps, pushforward_fn
from src.utils.metrics import particle_mse
from src.utils.eval_utils import eval_rollout
from src.utils.nbrs_utils import nearest
from src.utils.interpolate import Interpolator


class GNNSimulator(BaseSimulator):
    """Graph Network Simulator (GNS) for learning both u and v simultaneously."""

    def __init__(
        self,
        model_name,
        input_seq_length,
        latent_dim,
        num_message_passing_steps,
        noise_std,
        dataset_path,
        num_particle_types,
        particle_type_embedding_size,
        device,
        alpha_u,
        alpha_field,
        vel_solver=None,
        v2u_solver="none",
        isotropic_norm=False,
        **kwargs,  # model specific params
    ):
        super().__init__(
            model_name=model_name,
            device=device,
            isotropic_norm=isotropic_norm,
            noise_std=noise_std,
            dataset_path=dataset_path,
        )
        self._num_particle_types = num_particle_types
        if model_name != "lles":
            self._particle_type_embedding = nn.Embedding(
                num_particle_types, particle_type_embedding_size
            )  # (9, 16)
        self.alpha_u = alpha_u
        self.alpha_field = alpha_field
        self.vel_solver = vel_solver
        if alpha_u != 0:
            assert vel_solver is not None, "vel_solver must be specified if alpha_u!=0."
        self.v2u_solver = v2u_solver

        num_vs = input_seq_length - 1
        num_us = input_seq_length if alpha_u != 0 else 0

        if model_name == "gns":
            self._encode_process_decode = EncodeProcessDecode(
                node_in=self.dim * (num_vs + num_us) + particle_type_embedding_size,
                node_out=self.dim if self.alpha_u == 0 else 2 * self.dim,  # acceleration(s)
                edge_in=self.dim + 1,  # displacement and its magnitude
                latent_dim=latent_dim,
                num_message_passing_steps=num_message_passing_steps,
                mlp_num_layers=kwargs["mlp_num_layers"],
                mlp_hidden_dim=kwargs["mlp_hidden_dim"],
                alpha_u=alpha_u,
                local_interaction=kwargs["local_interaction"],
            )
        elif model_name == "segnn":
            input_irreps = Irreps(
                f"{num_vs + num_us}x1o + {particle_type_embedding_size}x0e"
            )  # node features
            output_irreps = Irreps("1x1o" if alpha_u == 0 else "2x1o")  # output (node) features
            edge_attr_irreps = Irreps.spherical_harmonics(kwargs["lmax_attr"])  # edge attributes
            node_attr_irreps = Irreps.spherical_harmonics(kwargs["lmax_attr"])  # node attributes
            additional_message_irreps = Irreps("1x1o + 1x0e")  # edge features

            hidden_irreps = WeightBalancedIrreps(
                Irreps("{}x0e".format(latent_dim)),
                node_attr_irreps,
                sh=True,
                lmax=kwargs["lmax_hidden"],
            )

            self._encode_process_decode = SEGNN(
                self.dim,
                num_vs,
                alpha_u,
                input_irreps,  # 5x1o / 11x1o
                hidden_irreps,  # 32x0e+32x1o
                output_irreps,  # 1x1o / 2x1o
                edge_attr_irreps,  # 1x0e+1x1o
                node_attr_irreps,  # 1x0e+1x1o
                num_layers=num_message_passing_steps,  # 10
                velocity_aggregate=kwargs["velocity_aggregate"],  # "avg", "last"
                norm=kwargs["norm"],  # None, "batch", "instance"
                additional_message_irreps=additional_message_irreps,  # 2x0e
            )
        elif model_name == "lles":
            self._encode_process_decode = LLES(
                input_features=kwargs["input_features"],
                hidden_dims=kwargs["hidden_dims"],
                output_dim=kwargs["output_dim"],
                activation=kwargs["activation"],
                metadata=self.metadata,
            )
        else:
            raise Warning(f"Model name {model_name} not recognized.")

        if v2u_solver == "smooth":
            self.v2u_smoothen = Interpolator(
                is_periodic=any(self._pbc),
                domain_size=[x[1] for x in self._boundaries],
                dim=self.dim,
                dx=self.metadata["dx"],
                condition=kwargs["v2u_smooth"]["condition"],
                k=kwargs["v2u_smooth"]["k"],
                cutoff_factor=kwargs["v2u_smooth"]["cutoff_factor"],
                kernel=kwargs["v2u_smooth"]["kernel"],
            )

    def _build_graph_from_raw(
        self, position_sequence, n_particles_per_trajectory, particle_types, pbc=True, **kwargs
    ):
        """
        Build a graph from raw data, including node and edge features.

        Args:
            position_sequence (Tensor): Sequence of positions.
            n_particles_per_trajectory (Tensor): Number of particles per trajectory.
            particle_types (Tensor): Particle types.
            pbc (bool): Periodic boundary conditions. Defaults to True.

        Raises:
            ValueError: If the interpolation condition is not valid.

        Returns:
            Tuple[Dict[str, Tensor], Tensor, Dict[str, Tensor]]: Node features, edge index, and edge features.
        """
        device = position_sequence.device
        n_total_points = position_sequence.shape[0]
        most_recent_position = position_sequence[:, -1]  # (N, D)

        v_velocity_sequence = time_diff(position_sequence, self._boundaries, pbc)

        # Normalized velocity sequence, merging spatial an time axis.
        v_normalized_velocity_sequence = self._norm(v_velocity_sequence, "vv")
        v_flat_velocity_sequence = v_normalized_velocity_sequence.view(n_total_points, -1)

        batch_ids = torch.cat(
            [
                torch.LongTensor([i for _ in range(n)])
                for i, n in enumerate(n_particles_per_trajectory)
            ]
        ).to(device)

        node_features = {
            "batch_ids": batch_ids,
            "v_flat_velocity_sequence": v_flat_velocity_sequence,
        }
        if self.alpha_u != 0:
            u_velocity_sequence = kwargs["u_velocity"]
            u_normalized_velocity_sequence = self._norm(u_velocity_sequence, "uu")
            u_flat_velocity_sequence = u_normalized_velocity_sequence.view(n_total_points, -1)
            node_features["u_flat_velocity_sequence"] = u_flat_velocity_sequence

        # Normalized clipped distances to lower and upper boundaries, if not PBC.
        if not pbc:
            # Normalized clipped distances to lower and upper boundaries.
            # boundaries are an array of shape [num_dimensions, 2], where the second
            # axis, provides the lower/upper bofundaries.
            boundaries = self._boundaries.clone().detach().float().requires_grad_(False).to(device)
            distance_to_lower_boundary = most_recent_position - boundaries[:, 0][None]
            distance_to_upper_boundary = boundaries[:, 1][None] - most_recent_position
            distance_to_boundaries = torch.cat(
                [distance_to_lower_boundary, distance_to_upper_boundary], dim=1
            )
            normalized_clipped_distance_to_boundaries = torch.clamp(
                distance_to_boundaries / self._connectivity_radius, -1.0, 1.0
            )
            node_features["normalized_clipped_distance_to_boundaries"] = (
                normalized_clipped_distance_to_boundaries
            )

        if self._num_particle_types > 1:
            node_features["particle_type_embeddings"] = self._particle_type_embedding(
                particle_types
            )

        # --- Edge Construction ---
        interp = kwargs.get("interpolate_params", {"condition": "radius"})
        if interp["condition"] == "radius":
            senders, receivers = nearest(
                most_recent_position,
                n_particles_per_trajectory,
                pbc,
                self._boundaries,
                cutoff=self._connectivity_radius,
            )
        elif interp["condition"] == "knn":
            senders, receivers = nearest(
                most_recent_position,
                n_particles_per_trajectory,
                pbc,
                self._boundaries,
                k=interp["k"],
                condition="knn",
            )
        else:
            raise ValueError("Invalid neighbor condition")

        edge_features = {}
        r_ij = self.displ_fn(most_recent_position[senders, :], most_recent_position[receivers, :])
        r_ij /= self._connectivity_radius
        edge_features["normalized_relative_displacements"] = r_ij
        edge_features["normalized_relative_distances"] = torch.norm(r_ij, dim=1, keepdim=True)

        if self.model_name == "lles":
            EPS = 1e-6
            most_recent_u_velocity = u_velocity_sequence[:, -1]
            v_ij = self._norm(
                most_recent_u_velocity[senders] - most_recent_u_velocity[receivers], key="uu"
            )
            r_sq = torch.sum(r_ij**2, dim=1, keepdim=True)
            v_sq = torch.sum(v_ij**2, dim=1, keepdim=True)
            dot_rv = torch.sum(r_ij * v_ij, dim=1, keepdim=True)

            edge_features = torch.cat([torch.sqrt(r_sq), torch.sqrt(v_sq), dot_rv], dim=1)
            edge_directions = torch.cat(
                [r_ij / torch.sqrt(r_sq + EPS), v_ij / torch.sqrt(v_sq + EPS)], dim=1
            )
            return senders, receivers, edge_features, edge_directions

        edge_index = torch.stack([senders, receivers], dim=0)
        return node_features, edge_index, edge_features

    def _decoder_postprocessor(self, a_v_pred, position_sequence, pbc=True, **kwargs):
        """
        Decoder postprocessor with PBC support.

        Args:
            normalized_acceleration (Tensor): Normalized accelerations.
            position_sequence (Tensor): Sequence of positions.
            boundaries (Tensor): Tensor of shape [1, dim] specifying the boundary lengths for each dimension.

        Returns:
            Tensor: New positions wrapped within the boundaries.
        """
        assert hasattr(self, "neuralsph"), "neuralsph must be defined in the model."

        # The model produces the output in normalized space so we apply inverse normalization.
        v_acceleration = self._denorm(a_v_pred, "va")

        most_recent_position = position_sequence[:, -1]
        most_recent_v_velocity = self.displ_fn(most_recent_position, position_sequence[:, -2])

        if self.alpha_u != 0:
            u_velocity = kwargs["u_velocity"]
            a_u_pred = kwargs["a_u_pred"]
            u_acceleration = self._denorm(a_u_pred, "ua")

            most_recent_u_velocity = u_velocity[:, -1]

            # Update velocity and position, use an Euler integrator to go from acceleration to
            # position, assuming dt = 1.
            if self.vel_solver == "simple":
                new_v_velocity = most_recent_v_velocity + v_acceleration  # * dt = 1
                new_u_velocity = most_recent_u_velocity + u_acceleration  # * dt = 1
                new_position = self.shift_fn(most_recent_position, new_v_velocity)

            elif self.vel_solver == "tvf":
                new_u_velocity = most_recent_u_velocity + u_acceleration
                new_v_velocity = self._u2v(new_u_velocity) + v_acceleration
                new_position = self.shift_fn(most_recent_position, new_v_velocity)

            elif self.vel_solver == "neural_sph":
                new_v_velocity = most_recent_v_velocity + v_acceleration
                new_u_velocity = self._v2u(new_v_velocity) + u_acceleration
                new_position = self.shift_fn(most_recent_position, new_v_velocity)

            elif self.vel_solver == "simple_u":
                new_u_velocity = most_recent_u_velocity + u_acceleration
                new_v_velocity = self._u2v(most_recent_u_velocity) + v_acceleration
                new_position = self.shift_fn(most_recent_position, new_v_velocity)

            elif self.vel_solver == "simple_u_closure":
                au_sph = self._sph(
                    most_recent_position, kwargs["n_part_per_traj"], most_recent_u_velocity
                )
                # TODO: au_sph * dt 20x larger than the predicted one
                new_u_velocity = (
                    most_recent_u_velocity + u_acceleration + au_sph * self._effective_dt
                )
                new_v_velocity = most_recent_v_velocity + v_acceleration
                new_position = self.shift_fn(most_recent_position, new_v_velocity)

            elif self.vel_solver == "simple_rlx":
                new_u_velocity = most_recent_u_velocity + u_acceleration
                new_pos_temp = self.shift_fn(most_recent_position, self._u2v(new_u_velocity))
                av_rlx, v_rlx, new_position = self._sph_rlx(
                    new_pos_temp,
                    kwargs["n_part_per_traj"],
                    is_tvf=self.metadata["rlx_is_tvf"],
                    dt_factor=self.metadata["rlx_dt_factor"],
                    num_steps=self.metadata["rlx_num_steps"],
                )
                if self.metadata["write_every"] > 1:
                    new_position = self.shift_fn(new_position, v_acceleration)
                    # # equivalent to:
                    # new_v_velocity = self._u2v(new_u_velocity) + av_rlx + v_acceleration
                    # new_position = self.shift_fn(most_recent_position, new_v_velocity)

            if self.neuralsph["num_steps"] > 0:
                _, _, new_position = self._sph_rlx(
                    new_position,
                    kwargs["n_part_per_traj"],
                    is_tvf=self.neuralsph["is_tvf"],
                    dt_factor=self.neuralsph["dt_factor"],
                    num_steps=self.neuralsph["num_steps"],
                )

            return new_position, new_u_velocity
        elif self.v2u_solver != "none":
            # Update velocity and position, use an Euler integrator to go from acceleration to
            # position, assuming dt = 1.
            if self.vel_solver == "simple":
                new_v_velocity = most_recent_v_velocity + v_acceleration  # * dt = 1
                new_position = self.shift_fn(most_recent_position, new_v_velocity)

            if self.v2u_solver == "same":
                new_u_velocity = self._v2u(new_v_velocity)
            elif self.v2u_solver == "smooth":
                new_u_velocity = self._v2u(new_v_velocity)
                new_u_velocity = self.v2u_smoothen(new_position, new_position, new_u_velocity)
            elif self.v2u_solver == "gns":
                raise NotImplementedError("v2u_solver=gns is not implemented yet.")

            if self.neuralsph["num_steps"] > 0:
                _, _, new_position = self._sph_rlx(
                    new_position,
                    kwargs["n_part_per_traj"],
                    is_tvf=self.neuralsph["is_tvf"],
                    dt_factor=self.neuralsph["dt_factor"],
                    num_steps=self.neuralsph["num_steps"],
                )

            return new_position, new_u_velocity
        else:
            new_v_velocity = most_recent_v_velocity + v_acceleration
            new_position = self.shift_fn(most_recent_position, new_v_velocity)

            if self.neuralsph["num_steps"] > 0:
                _, _, new_position = self._sph_rlx(
                    new_position,
                    kwargs["n_part_per_traj"],
                    is_tvf=self.neuralsph["is_tvf"],
                    dt_factor=self.neuralsph["dt_factor"],
                    num_steps=self.neuralsph["num_steps"],
                )
        return new_position

    def predict_positions(
        self, current_positions, n_particles_per_trajectory, particle_types, pbc=True, **kwargs
    ):
        """Predict the next position using GNS. Used for rollout."""
        if pbc:
            current_positions = current_positions % self._boundaries

        if self.alpha_u != 0:
            u_velocity = kwargs["u_velocity"]
            if self.model_name == "lles":
                interpolate_params = {
                    "condition": kwargs["interpolate"].condition,
                    "k": kwargs["interpolate"].k,
                }

                senders, receivers, edge_features, edge_directions = self._build_graph_from_raw(
                    current_positions,
                    n_particles_per_trajectory,
                    particle_types,
                    pbc,
                    u_velocity=u_velocity,
                    interpolate_params=interpolate_params,
                )
                # Get predicted accelerations
                a_v_pred, a_u_pred = self._encode_process_decode(
                    edge_features,
                    edge_directions,
                    art_visc_h=self._connectivity_radius,
                    senders=senders,
                )
            else:
                node_features, edge_index, e_features = self._build_graph_from_raw(
                    current_positions,
                    n_particles_per_trajectory,
                    particle_types,
                    pbc,
                    u_velocity=u_velocity,
                )
                a_v_pred, a_u_pred = self._encode_process_decode(
                    node_features, edge_index, e_features
                )

            next_position, new_u_velocity = self._decoder_postprocessor(
                a_v_pred,
                current_positions,
                pbc,
                a_u_pred=a_u_pred,
                u_velocity=u_velocity,
                n_part_per_traj=n_particles_per_trajectory,
            )
            return next_position, new_u_velocity
        elif self.v2u_solver != "none":
            u_velocity = kwargs["u_velocity"]
            node_features, edge_index, e_features = self._build_graph_from_raw(
                current_positions, n_particles_per_trajectory, particle_types, pbc
            )
            a_v_pred = self._encode_process_decode(node_features, edge_index, e_features)

            if self.v2u_solver in ["same", "smooth"]:
                next_position, new_u_velocity = self._decoder_postprocessor(
                    a_v_pred,
                    current_positions,
                    pbc,
                    n_part_per_traj=n_particles_per_trajectory,
                )

            return next_position, new_u_velocity

        else:
            node_features, edge_index, e_features = self._build_graph_from_raw(
                current_positions, n_particles_per_trajectory, particle_types, pbc
            )
            predicted_normalized_acceleration = self._encode_process_decode(
                node_features, edge_index, e_features
            )
            next_position = self._decoder_postprocessor(
                predicted_normalized_acceleration, current_positions, pbc
            )
            return next_position

    def predict_accelerations(
        self,
        next_position,
        position_sequence_noise,
        position_sequence,
        n_particles_per_trajectory,
        particle_types,
        pbc=True,
        **kwargs,
    ):
        """Predict acceleration using GNS and also return the target acceleration."""
        # next_position needs to drop dim=1 because the dataloader is configured for both training and validation, but for training this dim is not necessary.
        next_position = next_position.squeeze(1)
        noisy_position_sequence = self.shift_fn(position_sequence, position_sequence_noise)
        next_position_adjusted = self.shift_fn(next_position, position_sequence_noise[:, -1])

        # Compute the target normalized acceleration
        if self.alpha_u != 0:
            u_velocity = kwargs["u_velocity"]
            next_u_velocity = kwargs["next_u_velocity"]

            # Compute target accelerations
            a_v_target, a_u_target = self._inverse_decoder_postprocessor(
                next_position_adjusted,
                noisy_position_sequence,
                pbc,
                next_u_velocity=next_u_velocity,
                u_velocity=u_velocity,
                n_part_per_traj=n_particles_per_trajectory,
            )

            if self.model_name == "lles":
                interpolate = kwargs["interpolate"]
                interpolate_params = kwargs["interpolate_params"]

                senders, receivers, edge_features, edge_directions = self._build_graph_from_raw(
                    noisy_position_sequence,
                    n_particles_per_trajectory,
                    particle_types,
                    pbc,
                    u_velocity=u_velocity,
                    interpolate_params=interpolate_params,
                )
                # Get predicted accelerations
                a_v_pred, a_u_pred = self._encode_process_decode(
                    edge_features,
                    edge_directions,
                    art_visc_h=self._connectivity_radius,
                    senders=senders,
                )
            else:
                node_features, edge_index, e_features = self._build_graph_from_raw(
                    noisy_position_sequence,
                    n_particles_per_trajectory,
                    particle_types,
                    pbc,
                    u_velocity=u_velocity,
                )
                a_v_pred, a_u_pred = self._encode_process_decode(
                    node_features, edge_index, e_features
                )

            if self.alpha_field != 0:
                # Integrate predicted accelerations to get positions
                pred_positions, pred_velocities = self._integrate_accelerations(
                    a_v_pred=a_v_pred,
                    position_sequence=position_sequence,
                    pbc=pbc,
                    a_u_pred=a_u_pred,
                    u_velocity=kwargs["u_velocity"],
                )
                # Pred field
                u_field_pred = interpolate(
                    r=pred_positions, f=pred_velocities, npptr=n_particles_per_trajectory
                )

                # Target field
                u_field_gt = interpolate(
                    r=next_position_adjusted,
                    f=next_u_velocity.squeeze(1),
                    npptr=n_particles_per_trajectory,
                )
                return (a_v_pred, a_u_pred, u_field_gt), (
                    a_v_target,
                    a_u_target,
                    u_field_pred,
                )
            else:
                return (a_v_pred, a_u_pred), (a_v_target, a_u_target)

        elif self.v2u_solver != "none":
            u_velocity = kwargs["u_velocity"]
            next_u_velocity = kwargs["next_u_velocity"]
            node_features, edge_index, e_features = self._build_graph_from_raw(
                noisy_position_sequence, n_particles_per_trajectory, particle_types, pbc
            )
            a_v_pred = self._encode_process_decode(node_features, edge_index, e_features)

            if self.v2u_solver in ["same", "smooth"]:
                a_u_pred = torch.zeros_like(a_v_pred)

                a_v_target, a_u_target = self._inverse_decoder_postprocessor(
                    next_position_adjusted,
                    noisy_position_sequence,
                    pbc,
                    next_u_velocity=next_u_velocity,
                    u_velocity=u_velocity,
                    n_part_per_traj=n_particles_per_trajectory,
                )
                return (a_v_pred, a_u_pred), (a_v_target, a_u_target)
            elif self.v2u_solver == "gns":
                # TODO: implement second GNN for v2u
                raise NotImplementedError("v2u_solver=gns is not implemented yet.")

                # copy and stop gradients
                a_u_pred = a_v_pred.clone().detach()
                a_u_pred.requires_grad = False

        else:
            node_features, edge_index, e_features = self._build_graph_from_raw(
                noisy_position_sequence, n_particles_per_trajectory, particle_types, pbc
            )
            predicted_normalized_acceleration = self._encode_process_decode(
                node_features, edge_index, e_features
            )
            target_nomralized_acceleration = self._inverse_decoder_postprocessor(
                next_position_adjusted, noisy_position_sequence, pbc
            )

            return predicted_normalized_acceleration, target_nomralized_acceleration

    # PBC compatible implementation
    def _inverse_decoder_postprocessor(self, next_position, position_sequence, pbc=True, **kwargs):
        """Inverse of `_decoder_postprocessor`, with PBC support."""
        # Handle PBC for position differences
        previous_position = position_sequence[:, -1]
        previous_v_velocity = self.displ_fn(previous_position, position_sequence[:, -2])
        next_v_velocity = self.displ_fn(next_position, previous_position)

        if self.alpha_u != 0 or self.v2u_solver != "none":
            next_u_velocity = kwargs["next_u_velocity"].squeeze(1)
            previous_u_velocity = kwargs["u_velocity"][:, -1]

            # Update velocity and position, use an Euler integrator to go from acceleration to position, assuming dt = 1.
            if self.vel_solver == "simple":
                u_acceleration = next_u_velocity - previous_u_velocity
                v_acceleration = next_v_velocity - previous_v_velocity

            elif self.vel_solver == "tvf":
                u_acceleration = next_u_velocity - previous_u_velocity
                v_acceleration = next_v_velocity - self._u2v(next_u_velocity)

            elif self.vel_solver == "neural_sph":
                v_acceleration = next_v_velocity - previous_v_velocity
                u_acceleration = next_u_velocity - self._v2u(next_v_velocity)

            elif self.vel_solver == "simple_u":
                u_acceleration = next_u_velocity - previous_u_velocity
                v_acceleration = next_v_velocity - self._u2v(previous_u_velocity)

            elif self.vel_solver == "simple_u_closure":
                au_sph = self._sph(
                    previous_position, kwargs["n_part_per_traj"], previous_u_velocity
                )
                # au_sph is 3x larger than difference in u's
                u_acceleration = (
                    next_u_velocity - previous_u_velocity - au_sph * self._effective_dt
                )
                v_acceleration = next_v_velocity - previous_v_velocity

                """ Cosine similarity analysis
                import numpy as np
                def cosine(u, v, norm=True):
                    dot = (u*v).sum(dim=-1)
                    if norm:
                        norm_u = torch.linalg.norm(u, dim=-1)
                        norm_v = torch.linalg.norm(v, dim=-1)
                        dot /= (norm_u * norm_v)
                    return dot.mean()
                tuples = ((0.1, 1, False), (0.2, 1, False), (0.5, 1, False), (2, 1, False), (1, 0, False), (0, 1, False), (0, 0, True), (1, 1, False), (1, 1, True))  # (nu_p, nu, is_tvf) -> (0.01, )
                au_target = next_u_velocity - previous_u_velocity

                def eval_sph(t):
                    self._sph_fn = relax_wrapper(
                        Nx=int(round(previous_position.shape[0])**(1/self.metadata["dim"])),
                        dim=self.metadata["dim"],
                        L=self._boundaries[0].item(),
                        is_physical=True,
                        u_ref=self.metadata["u_ref"],
                        is_tvf=t[2],  # our relaxations always use tvf
                        nu=t[1],
                        box=self._boundaries,
                    )
                    return self._sph(previous_position, kwargs["n_part_per_traj"], previous_u_velocity, nu_p=t[0])

                a = eval_sph((1, 0, False))
                b = eval_sph((0, 1, False))
                c = eval_sph((0, 0, True))
                # print(f"{cosine(a, b):.4f}", f"{cosine(a, c):.4f}", f"{cosine(b, c):.4f}")
                with open("cos_a_b.txt", "a") as f:
                    f.write(f"{cosine(a, b):.6f}\n")
                with open("cos_a_c.txt", "a") as f:
                    f.write(f"{cosine(a, c):.6f}\n")
                with open("cos_b_c.txt", "a") as f:
                    f.write(f"{cosine(b, c):.6f}\n")

                # au_target /= au_target.std()
                for t in tuples:
                    lst = []
                    for scale in [0.0, 0.001, 0.002, 0.003, 0.01, 0.03]:
                        au_sph = eval_sph(t)
                        # au_sph /= au_sph.std()
                        u_acceleration = next_u_velocity - previous_u_velocity - au_sph * self._effective_dt * scale
                        lst.append(f"{u_acceleration.std():.6f}")

                    # Write u_acceleration.std() by appending to a file named target_std_{t[0]}_{t[1]}_{t[2]}.txt
                    cos = cosine(au_sph, au_target, norm=True).item()
                    with open(f"cos_{t[0]}_{t[1]}_{t[2]}.txt", "a") as f:
                        f.write(f"{cos:.6f}\n")
                    # print(t, f"{cos:4f}", lst, np.argmin(lst))
                    # Write np.argmin(lst) to a file named scale_argmin_{t[0]}_{t[1]}_{t[2]}.txt
                    with open(f"scale_argmin_{t[0]}_{t[1]}_{t[2]}.txt", "a") as f:
                        f.write(f"{np.argmin(lst)}\n")
                # print("#############################################")
                # import os
                # for prefix in ['cos', 'scale_argmin']:
                #     target_files = [f for f in os.listdir('.') if f.startswith(prefix) and f.endswith('.txt')]
                #     target_files.sort()
                #     for target_file in target_files:
                #         inner_list = []
                #         with open(target_file, 'r') as f:
                #             for line in f:
                #                 inner_list.append(float(line.strip()))
                #         print(f"{target_file:<30} {len(inner_list):<10} {np.array(inner_list).mean():<10.4f}")
                #     print("#"*50)

                cos_0.1_1_False.txt            1005       0.0244
                cos_0.2_1_False.txt            1005       0.0187
                cos_0.5_1_False.txt            1005       0.0105
                cos_0_0_True.txt               1004       0.0332
                cos_0_1_False.txt              1005       0.0346
                cos_1_0_False.txt              1005       -0.0111
                cos_1_1_False.txt              1004       0.0041
                cos_1_1_True.txt               1004       0.0380
                cos_2_1_False.txt              1005       -0.0014
                cos_a_b.txt                    1005       0.0161
                cos_a_c.txt                    1005       -0.5016
                cos_b_c.txt                    1005       0.1257
                ##################################################
                scale_argmin_0.1_1_False.txt   1005       0.3562
                scale_argmin_0.2_1_False.txt   1005       0.4010
                scale_argmin_0.5_1_False.txt   1005       0.4418
                scale_argmin_0_0_True.txt      1004       2.0976
                scale_argmin_0_1_False.txt     1005       0.3841
                scale_argmin_1_0_False.txt     1005       0.3015
                scale_argmin_1_1_False.txt     1004       0.3685
                scale_argmin_1_1_True.txt      1004       2.3386
                scale_argmin_2_1_False.txt     1005       0.2905
                """

            elif self.vel_solver == "simple_rlx":
                u_acceleration = next_u_velocity - previous_u_velocity
                if self.metadata["write_every"] > 1:
                    new_pos_temp = self.shift_fn(previous_position, self._u2v(previous_u_velocity))
                    av_rlx, v_rlx, new_position_temp = self._sph_rlx(
                        new_pos_temp,
                        kwargs["n_part_per_traj"],
                        is_tvf=self.metadata["rlx_is_tvf"],
                        dt_factor=self.metadata["rlx_dt_factor"],
                        num_steps=self.metadata["rlx_num_steps"],
                    )
                    v_acceleration = self.displ_fn(next_position, new_position_temp)
                    # # equivalent to:
                    # v_acceleration = (next_v_velocity - self._u2v(previous_u_velocity) - av_rlx).std()
                else:
                    v_acceleration = torch.zeros_like(u_acceleration)

            v_normalized_acceleration = self._norm(v_acceleration, "va")
            u_normalized_acceleration = self._norm(u_acceleration, "ua")
            return v_normalized_acceleration, u_normalized_acceleration
        else:
            # Compute acceleration
            v_acceleration = next_v_velocity - previous_v_velocity
            v_normalized_acceleration = self._norm(v_acceleration, "va")
            return v_normalized_acceleration

    def _integrate_accelerations(
        self, a_v_pred, position_sequence, pbc=True, a_u_pred=None, u_velocity=None
    ):
        """Integrate accelerations to get new positions and velocities for field prediction.

        Args:
            a_v_pred (Tensor): Normalized accelerations.
            position_sequence (Tensor): Sequence of positions.
            pbc (bool): Periodic boundary conditions. Defaults to True.
            a_u_pred (Tensor, optional): Normalized accelerations for u velocity. Defaults to None.
            u_velocity (Tensor, optional): u velocity. Defaults to None.

        Returns:
            Tuple[Tensor, Tensor]: New positions and velocities.
        """
        # The model produces the output in normalized space so we apply inverse normalization.
        v_acceleration = self._denorm(a_v_pred, "va")

        most_recent_position = position_sequence[:, -1]
        most_recent_v_velocity = self.displ_fn(most_recent_position, position_sequence[:, -2])

        if self.alpha_u != 0:
            u_acceleration = self._denorm(a_u_pred, "ua")

            most_recent_u_velocity = u_velocity[:, -1]

            # Update velocity and position, use an Euler integrator to go from acceleration to position, assuming dt = 1.
            if self.vel_solver == "simple":
                new_v_velocity = most_recent_v_velocity + v_acceleration  # * dt = 1
                new_u_velocity = most_recent_u_velocity + u_acceleration  # * dt = 1
                new_position = self.shift_fn(most_recent_position, new_v_velocity)
            elif self.vel_solver == "tvf":
                new_u_velocity = most_recent_u_velocity + u_acceleration
                new_v_velocity = self._u2v(new_u_velocity) + v_acceleration
                new_position = self.shift_fn(most_recent_position, new_v_velocity)
            elif self.vel_solver == "simple_u":
                new_u_velocity = most_recent_u_velocity + u_acceleration
                new_v_velocity = self._u2v(most_recent_u_velocity) + v_acceleration
                new_position = self.shift_fn(most_recent_position, new_v_velocity)

            return new_position, new_u_velocity


class GNNLitModule(BaseLitModule):
    """A LightningModule for training a Graph Network Simulator (GNS) model."""

    def __init__(
        self,
        net: torch.nn.Module,
        accelerator: torch.device,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        visualize: Dict[str, Dict[str, Any]] = None,
        compile: bool = False,
        seed: int = 0,
        pushforward: Dict[str, Any] = None,
        neuralsph: Dict[str, Any] = None,
        num_rollout_steps: int = 1,
        vel_solver: str = "simple",
        alpha_u: float = 1.0,
        alpha_v: float = 1.0,
        alpha_field: float = 1.0,
        active_metrics: Dict[str, Any] = None,
        metric_space: Dict[str, str] = "norm",
        v2u_solver: str = "none",
    ) -> None:
        """Initialize the GNS model's LightningModule.

        :param net: The GNS model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
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

        # Loss weights
        self.alpha_u = alpha_u
        self.alpha_v = alpha_v
        self.alpha_field = alpha_field

        if (alpha_u != 0.0 or v2u_solver != "none") and ("KOLM" not in self.net._case):
            raise NotImplementedError(
                "Alpha_u > 0.0 and v2u_solver != 'none' are only implemented for Kolmogorov."
            )
        # Push forward configuration
        self.seed = seed
        self.pushforward = pushforward
        if (self.pushforward is not None) and (self.alpha_u != 0.0 or v2u_solver != "none"):
            raise NotImplementedError(
                "Pushforward is only implemented for alpha_u = 0.0, i.e. LagrangeBench setting."
            )

        # Number of eval steps
        self.vel_solver = vel_solver
        if v2u_solver != "none":
            assert vel_solver == "simple", "v2u_solver is only implemented for vel_solver=simple."
            assert alpha_u == 0.0, "v2u_solver is only implemented for alpha_u = 0.0."
        self.v2u_solver = v2u_solver

    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        """Forward pass through the model."""
        return self.net.predict_accelerations(**features)

    def model_step(self, batch: Any) -> Tensor:
        """Forward pass through the model and compute the loss for a_u and a_v."""
        # Determine the number of pushforward steps, if applicable
        unroll_steps = 0
        if self.global_step != 0 and self.pushforward is not None:
            updated_seed, unroll_steps = pushforward_sample_steps(
                seed=self.seed, step=self.global_step, pushforward=self.pushforward
            )
            self.seed = updated_seed
        # Random walk noise
        position_sequence_noise = get_random_walk_noise_for_position_sequence(
            position_sequence=batch.enc_pos,
            boundaries=self.net._boundaries,
            noise_std_last_step=self.net.noise_std,
        ).to(self.device)

        non_kinematic_mask = (batch.particle_types != 3).clone().detach()
        position_sequence_noise *= non_kinematic_mask.view(-1, 1, 1)
        if not self.training:
            position_sequence_noise *= 0.0

        # Prepare features for the forward pass
        features = {
            "next_position": batch.target_pos[:, 0],  # Only the first target position
            "position_sequence": batch.enc_pos,
            "n_particles_per_trajectory": batch.n_particles_per_trajectory,
            "particle_types": batch.particle_types,
            "pbc": any(self.net._pbc),
            "position_sequence_noise": position_sequence_noise,
            "interpolate": self.metrics_interpolate,
            "interpolate_params": self.metric_space.interpolate,
        }
        if self.pushforward is not None:
            features["normalization_stats"] = self.net.normalization_stats
            features["boundaries"] = self.net._boundaries
        if self.alpha_u != 0.0 or self.v2u_solver != "none":
            features["u_velocity"] = batch.enc_u
            features["next_u_velocity"] = batch.target_u

        # Perform pushforward if unroll_steps > 0
        if unroll_steps > 0:
            # print(f"Pushing forward {unroll_steps} steps!!!")
            target_positions = batch.target_pos
            pred, target = pushforward_fn(features, target_positions, unroll_steps, self.forward)
        else:
            # Forward pass
            # pred and target should be tuples: (a_u_pred, a_v_pred), (a_u_target, a_v_target)
            pred, target = self.forward(features)

        kwargs_log = {"prog_bar": True, "on_epoch": True, "batch_size": batch.batch_size}
        if self.alpha_u != 0.0 or self.v2u_solver != "none":
            # Split predictions and targets
            if self.alpha_field != 0.0:
                a_v_pred, a_u_pred, field_pred = pred
                a_v_target, a_u_target, field_target = target
                # Calculate MSE field loss
                loss_field = torch.nn.functional.mse_loss(
                    field_pred, field_target, reduction="sum"
                )
                self.log("train/loss_field", loss_field, **kwargs_log)
            else:
                loss_field = 0.0
                a_v_pred, a_u_pred = pred
                a_v_target, a_u_target = target
            # Calculate MSE particle loss
            loss_v = particle_mse(a_v_pred, a_v_target, non_kinematic_mask)
            loss_u = particle_mse(a_u_pred, a_u_target, non_kinematic_mask)
            self.log("train/loss_u", loss_u, **kwargs_log)
            self.log("train/loss_v", loss_v, **kwargs_log)

            # Weighted combined loss
            loss = self.alpha_v * loss_v + self.alpha_u * loss_u + self.alpha_field * loss_field

            # TODO: add optional further models/losses for v2u_solver
        else:
            # Calculate loss
            loss = particle_mse(pred, target, non_kinematic_mask)
            self.log("train/loss_v", loss, **kwargs_log)

        return loss

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
            u_vel=True if (self.alpha_u != 0.0 or self.v2u_solver != "none") else False,
            metric_space=self.metric_space.test if is_test else self.metric_space.val,
            interpolate=self.metrics_interpolate,
        )

        kwargs_log = {"prog_bar": True, "on_epoch": True, "batch_size": batch.batch_size}
        if self.alpha_u != 0.0 or self.v2u_solver != "none":
            position_loss, u_vel_loss = loss
            if self.v2u_solver in ["same", "smooth"]:
                # these methods do not have trainable parameters toward improving u.
                # thus, u should not be used in early stopping criterium.
                _loss = position_loss["mse"].mean()
            else:
                _loss = position_loss["mse"] + u_vel_loss.mean()
            self.log(f"{split}/loss", _loss, **kwargs_log)
            self.log(f"{split}/u_loss", u_vel_loss.mean(), **kwargs_log)
        else:
            position_loss = loss
            self.log(f"{split}/loss", position_loss["mse"].mean(), **kwargs_log)
            # for k in ["mse", "mse1", "mse5", "mse10"]:  #, "mse20", "mse50", "mse100"]:
            #     self.log(f"val/{k}", position_loss[k].mean(), **kwargs_log)
            #     self.log(f"val/{k}std", position_loss[k].std(), **kwargs_log)

        self.log(f"{split}/v_loss", position_loss["mse"].mean(), **kwargs_log)
        self.log(
            f"{split}/loss_ekin", position_loss["e_kin"]["mse"].mean(), **kwargs_log
        )  # only shifting velocity currently
        if "mse_pos" in position_loss:
            self.log(f"{split}/mse_pos", position_loss["mse_pos"].mean(), **kwargs_log)

        self.metrics_dump[self.trajectory_idx] = loss
        self.trajectory_idx += 1
