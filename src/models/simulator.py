from pathlib import Path

import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph

from src.utils.data_utils import load_metadata
from src.utils.nbrs_utils import shift_fn, displ_fn, radius_graph_pbc
from src.models.components.sph import relax_wrapper
from src.models.components.gns import EncodeProcessDecode, time_diff
from src.models.components.segnn import SEGNN, WeightBalancedIrreps, Irreps


class Simulator(nn.Module):
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
        vel_solver=None,
        **kwargs  # model specific params
    ):
        super(Simulator, self).__init__()
        self._num_particle_types = num_particle_types
        self.noise_std = noise_std
        self.metadata = load_metadata(Path(dataset_path))
        self._boundaries = self.metadata["bounds"]
        self._connectivity_radius = self.metadata["default_connectivity_radius"]
        self._case = self.metadata["case"]
        self._pbc = self.metadata["periodic_boundary_conditions"]
        self._effective_dt = self.metadata["dt"] * self.metadata["write_every"]
        self._device = device
        self._particle_type_embedding = nn.Embedding(num_particle_types, particle_type_embedding_size) # (9, 16)
        self.alpha_u = alpha_u
        self.vel_solver = vel_solver
        self.model_name = model_name
        if alpha_u != 0:
            assert vel_solver is not None, "vel_solver must be specified if alpha_u!=0."

        self.dim = self.metadata["dim"]
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
                alpha_u=alpha_u
            )
        elif model_name == "segnn":
            input_irreps = Irreps(f"{num_vs + num_us}x1o + {particle_type_embedding_size}x0e")  # node features
            output_irreps = Irreps("1x1o" if alpha_u == 0 else "2x1o")  # output (node) features
            edge_attr_irreps = Irreps.spherical_harmonics(kwargs["lmax_attr"])  # edge attributes
            node_attr_irreps = Irreps.spherical_harmonics(kwargs["lmax_attr"])  # node attributes
            additional_message_irreps = Irreps("1x1o + 1x0e")  # edge features

            hidden_irreps = WeightBalancedIrreps(
                Irreps("{}x0e".format(latent_dim)), 
                node_attr_irreps, sh=True, lmax=kwargs["lmax_hidden"]
            )
            
            self._encode_process_decode = SEGNN(
                self.dim,
                num_vs,
                alpha_u,
                input_irreps,       # 5x1o / 11x1o
                hidden_irreps,      # 32x0e+32x1o
                output_irreps,      # 1x1o / 2x1o
                edge_attr_irreps,   # 1x0e+1x1o
                node_attr_irreps,   # 1x0e+1x1o
                num_layers=num_message_passing_steps,  # 10
                velocity_aggregate=kwargs["velocity_aggregate"], # "avg", "last"
                norm=kwargs["norm"],  # None, "batch", "instance"
                additional_message_irreps=additional_message_irreps  # 2x0e
            )
        else:
            raise ValueError(f"Model name {model_name} not recognized.")
        
    def set_metadata_device(self, device=None):
        if device is None:
            device = self._device

        # when working with equivariant models, we need to scale dimensions isotropically
        is_isotropic = True if self.model_name == "segnn" else False

        # box size
        self._boundaries = (
            torch.tensor(self.metadata["bounds"], requires_grad=False).float().to(device)
        )
        #Subract the ends of the box to get its size (used for PBC)
        self._boundaries = self._boundaries[:,1] - self._boundaries[:,0]

        # v stats
        va_m = torch.FloatTensor(self.metadata["acc_mean"]).to(device)
        va_s = torch.FloatTensor(self.metadata["acc_std"]).to(device)
        vv_m = torch.FloatTensor(self.metadata["vel_mean"]).to(device)
        vv_s = torch.FloatTensor(self.metadata["vel_std"]).to(device)
        if is_isotropic:
            va_m = va_m.mean() * torch.ones_like(va_m)
            va_s = va_s.mean() * torch.ones_like(va_s)
            vv_m = vv_m.mean() * torch.ones_like(vv_m)
            vv_s = vv_s.mean() * torch.ones_like(vv_s)
        self.normalization_stats = {
                "v_acceleration": {"mean": va_m, "std": torch.sqrt(va_s ** 2 + self.noise_std**2)},
                "v_velocity": {"mean": vv_m, "std": torch.sqrt(vv_s ** 2 + self.noise_std**2)}
            }
        
        # u stats
        if self.alpha_u != 0:
            ua_m = torch.FloatTensor(self.metadata["au_mean"]).to(device)
            ua_s = torch.FloatTensor(self.metadata["au_std"]).to(device)
            uu_m = torch.FloatTensor(self.metadata["u_mean"]).to(device)
            uu_s = torch.FloatTensor(self.metadata["u_std"]).to(device)
            if is_isotropic:
                ua_m = ua_m.mean() * torch.ones_like(ua_m)
                ua_s = ua_s.mean() * torch.ones_like(ua_s)
                uu_m = uu_m.mean() * torch.ones_like(uu_m)
                uu_s = uu_s.mean() * torch.ones_like(uu_s)
            self.normalization_stats["u_acceleration"] = {
                "mean": ua_m, "std": torch.sqrt(ua_s ** 2 + self.noise_std**2)
            }
            self.normalization_stats["u_velocity"] = {
                "mean": uu_m, "std": torch.sqrt(uu_s ** 2 + self.noise_std**2)
            }
    
    def _norm(self, vel, key):
        """From displacement of positions `v=x1-x0` to a normal distribution."""
        # key = "ua/uu/va/vv"
        kv, ka = key
        ka = {"a": "acceleration", "v": "velocity", "u": "velocity"}[ka]
    
        stats = self.normalization_stats[f"{kv}_{ka}"]
        return (vel - stats['mean']) / stats['std']

    def _denorm(self, vel, key):
        """From a normal distribution to displacement of positions `v=x1-x0`."""
        kv, ka = key
        ka = {"a": "acceleration", "v": "velocity", "u": "velocity"}[ka]
    
        stats = self.normalization_stats[f"{kv}_{ka}"]
        return vel * stats['std'] + stats['mean']
        
    def shift_fn(self, r, dr):
        return shift_fn(r, dr, self._boundaries, self._pbc)
    
    def displ_fn(self, r1, r2):
        return displ_fn(r1, r2, self._boundaries, self._pbc)

    def _u2v(self, u):
        """Convert velocity `u` to displacement of positoins `v = x1 - x0 = dt * u`."""
        return self._effective_dt * u
    
    def _v2u(self, v):
        """Convert displacement of positoins `v` to velocity `u = (x1 - x0) / dt`."""
        return v / self._effective_dt

    def forward(self):
        pass

    def _build_graph_from_raw(
        self, position_sequence, n_particles_per_trajectory, particle_types, pbc=True, **kwargs
    ):
        n_total_points = position_sequence.shape[0]
        most_recent_position = position_sequence[:, -1] # (n_nodes, 2)
        v_velocity_sequence = time_diff(position_sequence, self._boundaries, pbc)
        
        # senders and receivers are integers of shape (E,)

        if not pbc:
            senders, receivers = self._compute_connectivity(
                most_recent_position, n_particles_per_trajectory, self._connectivity_radius
            )
        elif pbc:
            #Pytorch Geometric Implementation
            senders, receivers = self._compute_connectivity_pbc_pyg(
                most_recent_position, n_particles_per_trajectory, self._connectivity_radius
            )

        batch_ids = torch.cat([
            torch.LongTensor([i for _ in range(n)])
            for i, n in enumerate(n_particles_per_trajectory)
        ]).to(self._device)
        node_features = {"batch_ids": batch_ids}
        
        # Normalized velocity sequence, merging spatial an time axis.
        v_normalized_velocity_sequence = self._norm(v_velocity_sequence, "vv")
        v_flat_velocity_sequence = v_normalized_velocity_sequence.view(n_total_points, -1)
        node_features["v_flat_velocity_sequence"] = v_flat_velocity_sequence
        
        if self.alpha_u != 0:
            u_velocity_sequence = kwargs["u_velocity"]  # (N, T_in=6, D)
            u_normalized_velocity_sequence = self._norm(u_velocity_sequence, "uu")
            u_flat_velocity_sequence = u_normalized_velocity_sequence.view(n_total_points, -1)
            node_features["u_flat_velocity_sequence"] = u_flat_velocity_sequence

        # Normalized clipped distances to lower and upper boundaries, if not PBC.
        if not pbc:
            # Normalized clipped distances to lower and upper boundaries.
            # boundaries are an array of shape [num_dimensions, 2], where the second
            # axis, provides the lower/upper bofundaries.
            boundaries = self._boundaries.clone().detach().float().requires_grad_(False).to(self._device) # torch.tensor(self._boundaries, requires_grad=False).float().to(self._device)
            distance_to_lower_boundary = (most_recent_position - boundaries[:, 0][None])
            distance_to_upper_boundary = (boundaries[:, 1][None] - most_recent_position)
            distance_to_boundaries = torch.cat([distance_to_lower_boundary, distance_to_upper_boundary], dim=1)
            normalized_clipped_distance_to_boundaries = torch.clamp(distance_to_boundaries / self._connectivity_radius, -1., 1.)
            node_features["normalized_clipped_distance_to_boundaries"] = normalized_clipped_distance_to_boundaries


        if self._num_particle_types > 1:
            particle_type_embeddings = self._particle_type_embedding(particle_types)
            node_features["particle_type_embeddings"] = particle_type_embeddings

        # Collect edge features.
        edge_features = {}

        # Relative displacement and distances normalized to radius
        # (E, 2)
        # normalized_relative_displacements = (
        #     torch.gather(most_recent_position, 0, senders) - torch.gather(most_recent_position, 0, receivers)
        # ) / self._connectivity_radius
        normalized_relative_displacements = self.displ_fn(most_recent_position[senders, :], most_recent_position[receivers, :])
        
        normalized_relative_displacements /= self._connectivity_radius
        edge_features["normalized_relative_displacements"] = normalized_relative_displacements

        normalized_relative_distances = torch.norm(normalized_relative_displacements, dim=-1, keepdim=True)
        edge_features["normalized_relative_distances"] = normalized_relative_distances
            
        return node_features, torch.stack([senders, receivers]), edge_features

    def _compute_connectivity(self, node_features, n_particles_per_trajectory, radius, add_self_edges=True):
        # handle batches. Default is 2 examples per batch
        # Specify examples id for particles/points
        batch_ids = torch.cat([torch.LongTensor([i for _ in range(n)]) for i, n in enumerate(n_particles_per_trajectory)]).to(self._device)
        # radius = radius + 0.00001 # radius_graph takes r < radius not r <= radius
        edge_index = radius_graph(node_features, r=radius, batch=batch_ids, loop=add_self_edges) # (2, n_edges)
        receivers = edge_index[0, :]
        senders = edge_index[1, :]
        return receivers, senders
    
    def _compute_connectivity_pbc_pyg(self, most_recent_position, n_particles_per_trajectory, radius, add_self_edges=True):
        return radius_graph_pbc(most_recent_position, n_particles_per_trajectory, radius, self._boundaries, add_self_edges)
    
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

            # Update velocity and position, use an Euler integrator to go from acceleration to position, assuming dt = 1.
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
                au_sph = self._sph(most_recent_position, kwargs["n_part_per_traj"], most_recent_u_velocity)
                # TODO: au_sph * dt 20x larger than the predicted one
                new_u_velocity = most_recent_u_velocity + u_acceleration + au_sph * self._effective_dt
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
                    num_steps=self.metadata["rlx_num_steps"]
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
                    num_steps=self.neuralsph["num_steps"]
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
                    num_steps=self.neuralsph["num_steps"]
                )
        return new_position

    def _sph_rlx(self, r, n_part_per_traj, is_tvf, dt_factor, num_steps):
        # This relaxation is the exact same from dataset generation, iff no coarsening."

        # Relax a point cloud using SPH without viscosity, but with transport vel.
        if not hasattr(self, "_relax_fn"):
            self._relax_fn = relax_wrapper(
                Nx=int(round(r.shape[0])**(1/self.metadata["dim"])),
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
            a_temp = self._relax_fn(r, n_part_per_traj) #, verbose=True if (i==num_steps-1) else False)
            dr = (dt_factor * self.metadata["dt"]) ** 2 * a_temp
            r = shift_fn(r, dr)
            v += dr

        # The following line gives the same v up to >3 digits
        # v = self.displ_fn(r, r_input)  # v=x1-x0 as defined everywhere else in our code
        return v, v, r

    def _sph(self, r, n_part_per_traj, u):
        # Compute NSE right hand side term.
        if not hasattr(self, "_sph_fn"):
            self._sph_fn = relax_wrapper(
                Nx=int(round(r.shape[0])**(1/self.metadata["dim"])),
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

    def predict_positions(self, current_positions, n_particles_per_trajectory, particle_types, pbc=True, **kwargs):
        if pbc:
            current_positions = current_positions % self._boundaries
            
        if self.alpha_u != 0:
            u_velocity = kwargs["u_velocity"]
            node_features, edge_index, e_features = self._build_graph_from_raw(
                current_positions, n_particles_per_trajectory, particle_types, pbc, u_velocity=u_velocity)
            a_v_pred, a_u_pred = self._encode_process_decode(node_features, edge_index, e_features)
            next_position, new_u_velocity = self._decoder_postprocessor(
                a_v_pred, current_positions, pbc, 
                a_u_pred=a_u_pred, u_velocity=u_velocity, n_part_per_traj=n_particles_per_trajectory
            )
            return next_position, new_u_velocity
        else:
            node_features, edge_index, e_features = self._build_graph_from_raw(
                current_positions, n_particles_per_trajectory, particle_types, pbc)
            predicted_normalized_acceleration = self._encode_process_decode(node_features, edge_index, e_features)
            next_position = self._decoder_postprocessor(predicted_normalized_acceleration, current_positions, pbc)
            return next_position

    def predict_accelerations(self, next_position, position_sequence_noise, position_sequence, n_particles_per_trajectory, particle_types, pbc=True, **kwargs):
        #next_position needs to drop dim=1 because the dataloader is configured for both training and validation, but for training this dim is not necessary.
        next_position = next_position.squeeze(1)
        noisy_position_sequence = self.shift_fn(position_sequence, position_sequence_noise)
        next_position_adjusted = self.shift_fn(next_position, position_sequence_noise[:, -1])

        #Compute the target normalized acceleration
        if self.alpha_u != 0:
            u_velocity = kwargs["u_velocity"]
            next_u_velocity = kwargs["next_u_velocity"]
            node_features, edge_index, e_features = self._build_graph_from_raw(
                noisy_position_sequence, n_particles_per_trajectory, particle_types, pbc, u_velocity=u_velocity)
            a_v_pred, a_u_pred = self._encode_process_decode(node_features, edge_index, e_features)
            a_v_target, a_u_target = self._inverse_decoder_postprocessor(
                next_position_adjusted, noisy_position_sequence, pbc, 
                next_u_velocity=next_u_velocity, u_velocity=u_velocity,
                n_part_per_traj=n_particles_per_trajectory
            )
            
            return (a_v_pred, a_u_pred), (a_v_target, a_u_target)   
        else:
            node_features, edge_index, e_features = self._build_graph_from_raw(
                noisy_position_sequence, n_particles_per_trajectory, particle_types, pbc)
            predicted_normalized_acceleration = self._encode_process_decode(node_features, edge_index, e_features)
            target_nomralized_acceleration = self._inverse_decoder_postprocessor(
                next_position_adjusted, noisy_position_sequence, pbc)
            
            return predicted_normalized_acceleration, target_nomralized_acceleration
         
    #PBC COMPATIBLE IMPLEMENTATION
    def _inverse_decoder_postprocessor(self, next_position, position_sequence, pbc=True, **kwargs):
        """Inverse of `_decoder_postprocessor`, with PBC support."""
        # Handle PBC for position differences
        previous_position = position_sequence[:, -1]
        previous_v_velocity = self.displ_fn(previous_position, position_sequence[:, -2])
        next_v_velocity = self.displ_fn(next_position, previous_position)

        if self.alpha_u != 0:
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
                au_sph = self._sph(previous_position, kwargs["n_part_per_traj"], previous_u_velocity)
                # au_sph is 3x larger than difference in u's
                u_acceleration = next_u_velocity - previous_u_velocity -  au_sph * self._effective_dt
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
                        num_steps=self.metadata["rlx_num_steps"]
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

