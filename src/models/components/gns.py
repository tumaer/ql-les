import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, radius_graph
from src.utils.train_utils import wrap_displacement
from src.utils.data_utils import load_metadata
from src.utils.nbrs_utils import shift_fn, displ_fn, radius_graph_pbc
from src.models.components.sph import relax_wrapper
from pathlib import Path


EPSILON = 1e-8

def build_mlp(
    input_size,
    layer_sizes,
    output_size=None,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ReLU,
):
    sizes = [input_size] + layer_sizes
    if output_size:
        sizes.append(output_size)

    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)

def time_diff(input_sequence, boundaries, pbc=True):
    if pbc:
        raw_diff = input_sequence[:, 1:] - input_sequence[:, :-1]
        wrapped_diff = wrap_displacement(raw_diff, boundaries)
        return wrapped_diff
    else:
        return input_sequence[:, 1:] - input_sequence[:, :-1]


def get_random_walk_noise_for_position_sequence(
    position_sequence, noise_std_last_step, boundaries, pbc=True
):
    """Returns random-walk noise in the velocity applied to the position."""
    velocity_sequence = time_diff(position_sequence, boundaries, pbc)
    
    num_velocities = velocity_sequence.shape[1]
    
    velocity_sequence_noise = torch.randn(list(velocity_sequence.shape)) * (
        noise_std_last_step/num_velocities**0.5
    )
    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)
    
    position_sequence_noise = torch.cat([
        torch.zeros_like(velocity_sequence_noise[:, 0:1]),
        torch.cumsum(velocity_sequence_noise, dim=1)], dim=1)

    return position_sequence_noise

class Encoder(nn.Module):
    def __init__(
        self, 
        node_in, 
        node_out, 
        edge_in, 
        edge_out,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(Encoder, self).__init__()
        self.node_fn = nn.Sequential(*[build_mlp(node_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out), 
            nn.LayerNorm(node_out)])
        self.edge_fn = nn.Sequential(*[build_mlp(edge_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], edge_out), 
            nn.LayerNorm(edge_out)])

    def forward(self, x, edge_index, e_features): # global_features
        # x: (E, node_in)
        # edge_index: (2, E)
        # e_features: (E, edge_in)         
        return self.node_fn(x), self.edge_fn(e_features)

class InteractionNetwork(MessagePassing):
    def __init__(
        self, 
        node_in, 
        node_out, 
        edge_in, 
        edge_out,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(InteractionNetwork, self).__init__(aggr='add')
        self.node_fn = nn.Sequential(*[build_mlp(node_in+edge_out, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out), 
            nn.LayerNorm(node_out)])
        self.edge_fn = nn.Sequential(*[build_mlp(node_in+node_in+edge_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], edge_out), 
            nn.LayerNorm(edge_out)])

    def forward(self, x, edge_index, e_features):
        # x: (E, node_in)
        # edge_index: (2, E)
        # e_features: (E, edge_in)
        x_residual = x
        e_features_residual = e_features
        x, e_features = self.propagate(edge_index=edge_index, x=x, e_features=e_features)
        return x+x_residual, e_features+e_features_residual

    def message(self, edge_index, x_i, x_j, e_features):
        e_features = torch.cat([x_i, x_j, e_features], dim=-1)
        e_features = self.edge_fn(e_features)
        return e_features

    def update(self, x_updated, x, e_features):
        # x_updated: (E, edge_out)
        # x: (E, node_in)
        x_updated = torch.cat([x_updated, x], dim=-1)
        x_updated = self.node_fn(x_updated)
        return x_updated, e_features

class Processor(MessagePassing):
    def __init__(
        self, 
        node_in, 
        node_out, 
        edge_in, 
        edge_out,
        num_message_passing_steps,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(Processor, self).__init__(aggr='max')
        self.gnn_stacks = nn.ModuleList([
            InteractionNetwork(
                node_in=node_in, 
                node_out=node_out,
                edge_in=edge_in, 
                edge_out=edge_out,
                mlp_num_layers=mlp_num_layers,
                mlp_hidden_dim=mlp_hidden_dim,
            ) for _ in range(num_message_passing_steps)])

    def forward(self, x, edge_index, e_features):
        for gnn in self.gnn_stacks:
            x, e_features = gnn(x, edge_index, e_features)
        return x, e_features

class Decoder(nn.Module):
    def __init__(
        self, 
        node_in, 
        node_out,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(Decoder, self).__init__()
        self.node_fn = build_mlp(node_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out)

    def forward(self, x):
        # x: (E, node_in)
        return self.node_fn(x)

class EncodeProcessDecode(nn.Module):
    def __init__(
        self, 
        node_in,
        node_out,
        edge_in,
        latent_dim,
        num_message_passing_steps,
        mlp_num_layers,
        mlp_hidden_dim,
        alpha_u
        
    ):
        super(EncodeProcessDecode, self).__init__()
        self._encoder = Encoder(
            node_in=node_in, 
            node_out=latent_dim,
            edge_in=edge_in, 
            edge_out=latent_dim,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self._processor = Processor(
            node_in=latent_dim, 
            node_out=latent_dim,
            edge_in=latent_dim, 
            edge_out=latent_dim,
            num_message_passing_steps=num_message_passing_steps,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self._decoder = Decoder(
            node_in=latent_dim,
            node_out=node_out,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        )
        self.alpha_u = alpha_u

    def forward(self, x, edge_index, e_features):
        # x: (E, node_in)
        x, e_features = self._encoder(x, edge_index, e_features)
        x, e_features = self._processor(x, edge_index, e_features)
        x = self._decoder(x)
        if self.alpha_u != 0:
            a_v, a_u = torch.chunk(x, 2, dim=-1)
            return a_v, a_u
        else:
            return x

class Simulator(nn.Module):
    def __init__(
        self,
        #particle_dimension, #FIXME: given the dim of the dataset, this used to be how the model determined the output dimension
        node_in,
        edge_in,
        latent_dim,
        node_out,
        num_message_passing_steps,
        mlp_num_layers,
        mlp_hidden_dim,
        noise_std,
        dataset_path,
        num_particle_types,
        particle_type_embedding_size,
        device,
        alpha_u,
        vel_solver=None,
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
        if alpha_u != 0:
            assert vel_solver is not None, "vel_solver must be specified if alpha_u!=0."

        self._encode_process_decode = EncodeProcessDecode(
            node_in=node_in,
            node_out=node_out, #used to be particle_dimension
            edge_in=edge_in,
            latent_dim=latent_dim,
            num_message_passing_steps=num_message_passing_steps,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            alpha_u=alpha_u
        
        )
        
    def set_metadata_device(self, device=None):
        if device is None:
            device = self._device
        self.normalization_stats = {
                "v_acceleration": {
                    "mean": torch.FloatTensor(self.metadata["acc_mean"]).to(device),
                    "std": torch.sqrt(
                        torch.FloatTensor(self.metadata["acc_std"]) ** 2 + self.noise_std**2
                    ).to(device),
                },
                "v_velocity": {
                    "mean": torch.FloatTensor(self.metadata["vel_mean"]).to(device),
                    "std": torch.sqrt(
                        torch.FloatTensor(self.metadata["vel_std"]) ** 2 + self.noise_std**2
                    ).to(device),
                },
            }
        
        self._boundaries = (
            torch.tensor(self.metadata["bounds"], requires_grad=False).float().to(device)
        )
        if self.alpha_u != 0:
            self.normalization_stats["u_acceleration"] = {
                "mean": torch.FloatTensor(self.metadata["au_mean"]).to(device),
                "std": torch.sqrt(
                    torch.FloatTensor(self.metadata["au_std"]) ** 2 + self.noise_std**2
                ).to(device),
            }
            self.normalization_stats["u_velocity"] = {
                "mean": torch.FloatTensor(self.metadata["u_mean"]).to(device),
                "std": torch.sqrt(
                    torch.FloatTensor(self.metadata["u_std"]) ** 2 + self.noise_std**2
                ).to(device),
            }
    
        #Subract the ends of the box to get its size (used for PBC)
        self._boundaries = self._boundaries[:,1] - self._boundaries[:,0]
    
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
            senders, receivers = self._compute_connecitivity_pbc_pyg(
                most_recent_position, n_particles_per_trajectory, self._connectivity_radius
            )

        node_features = []
        
        # Normalized velocity sequence, merging spatial an time axis.
        v_normalized_velocity_sequence = self._norm(v_velocity_sequence, "vv")
        v_flat_velocity_sequence = v_normalized_velocity_sequence.view(n_total_points, -1)
        node_features.append(v_flat_velocity_sequence)
        
        if self.alpha_u != 0:
            u_velocity_sequence = kwargs["u_velocity"]  # (N, T_in=6, D)
            u_normalized_velocity_sequence = self._norm(u_velocity_sequence, "uu")
            u_flat_velocity_sequence = u_normalized_velocity_sequence.view(n_total_points, -1)
            node_features.append(u_flat_velocity_sequence)

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
            node_features.append(normalized_clipped_distance_to_boundaries)


        if self._num_particle_types > 1:
            particle_type_embeddings = self._particle_type_embedding(particle_types)
            node_features.append(particle_type_embeddings)

        # Collect edge features.
        edge_features = []

        # Relative displacement and distances normalized to radius
        # (E, 2)
        # normalized_relative_displacements = (
        #     torch.gather(most_recent_position, 0, senders) - torch.gather(most_recent_position, 0, receivers)
        # ) / self._connectivity_radius
        normalized_relative_displacements = self.displ_fn(most_recent_position[senders, :], most_recent_position[receivers, :])
        
        normalized_relative_displacements /= self._connectivity_radius
        edge_features.append(normalized_relative_displacements)

        normalized_relative_distances = torch.norm(normalized_relative_displacements, dim=-1, keepdim=True)
        edge_features.append(normalized_relative_distances)
            
        return torch.cat(node_features, dim=-1), torch.stack([senders, receivers]), torch.cat(edge_features, dim=-1)

    def _compute_connectivity(self, node_features, n_particles_per_trajectory, radius, add_self_edges=True):
        # handle batches. Default is 2 examples per batch
        # Specify examples id for particles/points
        batch_ids = torch.cat([torch.LongTensor([i for _ in range(n)]) for i, n in enumerate(n_particles_per_trajectory)]).to(self._device)
        # radius = radius + 0.00001 # radius_graph takes r < radius not r <= radius
        edge_index = radius_graph(node_features, r=radius, batch=batch_ids, loop=add_self_edges) # (2, n_edges)
        receivers = edge_index[0, :]
        senders = edge_index[1, :]
        return receivers, senders
    
    def _compute_connecitivity_pbc_pyg(self, most_recent_position, n_particles_per_trajectory, radius, add_self_edges=True):
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
                av_rlx, v_rlx, new_position = self._sph_rlx(new_pos_temp, kwargs["n_part_per_traj"])
                if self.metadata["write_every"] > 1:
                    new_position = self.shift_fn(new_position, v_acceleration/self._effective_dt)
                    # # equivalent to:
                    # new_v_velocity = self._u2v(new_u_velocity) + av_rlx + v_acceleration/self._effective_dt
                    # new_position = self.shift_fn(most_recent_position, new_v_velocity)

            return new_position, new_u_velocity
        
        else:
            new_v_velocity = most_recent_v_velocity + v_acceleration
            new_position = self.shift_fn(most_recent_position, new_v_velocity)
            
        return new_position

    def _sph_rlx(self, r, n_part_per_traj):
        # This relaxation is the exact same from dataset generation, iff no coarsening."

        # Relax a point cloud using SPH without viscosity, but with transport vel.
        if not hasattr(self, "_relax_fn"):
            self._relax_fn = relax_wrapper(
                Nx=int(round(r.shape[0])**(1/self.metadata["dim"])),
                dim=self.metadata["dim"],
                L=self._boundaries[0].item(),
                is_physical=True,
                u_ref=self.metadata["u_ref"],
                is_tvf=True,  # our relaxations always use tvf
                nu=0.0,  # relaxations assume zero velocity, so this term drops
                box=self._boundaries,
            )

        dt_factor = 2  # TODO: "2" should be somehow passed from metadata.
        v = 0.0
        # r_input = r.detach().clone()
        for _ in range(2):  # TODO: "2" should be somehow passed from metadata.
            a_temp = self._relax_fn(r, n_part_per_traj)
            dr = (dt_factor * self._effective_dt) ** 2 * a_temp
            r = shift_fn(r, dr)
            v += dr

        # The following line gives save v up to >3 digits
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
                    av_rlx, v_rlx, new_position_temp = self._sph_rlx(new_pos_temp, kwargs["n_part_per_traj"])
                    # TODO: the multiplication with self._effective_dt is a heuristic
                    v_acceleration = self.displ_fn(next_position, new_position_temp) * self._effective_dt
                    # v_acceleration = next_v_velocity - previous_v_velocity

                    # v_acceleration1 = self.displ_fn(next_position, previous_position) * self._effective_dt
                    # v_normalized_acceleration = self._norm(v_acceleration, "va")
                    
                    # a = self.displ_fn(next_position, new_position_temp)
                    # v_t = self.displ_fn(next_position, previous_position)
                    # au_t = u_acceleration

                    # u1_min_u0 = next_u_velocity - previous_u_velocity
                    # v1_min_u1 = next_v_velocity - self._u2v(next_u_velocity)
                    # dr_rlx = v_rlx
                    # print(cosine(u1_min_u0,v1_min_u1))
                    # print(cosine(v1_min_u1,dr_rlx))
                    
                    # # equivalent to (if we didn't have the `*self._effective_dt` above and here):
                    # v_acceleration = (next_v_velocity - self._u2v(previous_u_velocity) - av_rlx*self._effective_dt)
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
