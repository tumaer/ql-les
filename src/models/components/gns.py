import os
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, radius_graph
from src.utils.train_utils import wrap_displacement, wrap_position
from src.utils.neighbor_search_algorithms import compute_connectivity_pbc #Alternative to radius_graph
from src.utils.data_utils import load_metadata
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
os.makedirs('train_log', exist_ok=True)
os.makedirs('rollouts', exist_ok=True)

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


def pbc_duplication(most_recent_positions, domain_size):
    """
    Apply periodic boundary condition duplication based on the specified PBC list.

    Args:
        most_recent_positions (torch.Tensor): Tensor of positions with shape [N, D].
        domain_size (torch.Tensor): Tensor of domain size for each dimension with shape [D].
        pbc (list of bool): List of boolean values indicating whether PBC should be applied for each dimension.

    Returns:
        torch.Tensor: Combined positions with periodic boundary duplication applied.
    """
    
    D = most_recent_positions.size(1)  # Number of spatial dimensions
    if D == 2:
        pbc = [True, True]
    elif D == 3:
        pbc = [True, True, True]

    # Create shifts based on pbc
    shift_ranges = [
        [-1, 0, 1] if p else [0] for p in pbc
    ]
    shifts = torch.cartesian_prod(*[torch.tensor(r) for r in shift_ranges]).float()
    shifts = shifts[~torch.all(shifts == 0, dim=1)].to(most_recent_positions.device) # Exclude the center (original frame)

    # Scale shifts by the domain size
    shifts = (shifts * domain_size).to(most_recent_positions.device)

    # Apply shifts to generate duplicated frames
    duplicated_frames = torch.cat([
        most_recent_positions + shift for shift in shifts
    ], dim=0).to(most_recent_positions.device)  # Shape: [N * len(shifts), D]

    # Combine original frame and duplicated frames
    combined_positions = torch.cat([most_recent_positions, duplicated_frames], dim=0).to(most_recent_positions.device)
    return combined_positions

def get_random_walk_noise_for_position_sequence(position_sequence, noise_std_last_step, boundaries, pbc=True):
    """Returns random-walk noise in the velocity applied to the position."""
    velocity_sequence = time_diff(position_sequence, boundaries, pbc)
    num_velocities = velocity_sequence.shape[1]
    velocity_sequence_noise = torch.randn(list(velocity_sequence.shape)) * (noise_std_last_step/num_velocities**0.5)

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

    def forward(self, x, edge_index, e_features):
        # x: (E, node_in)
        x, e_features = self._encoder(x, edge_index, e_features)
        x, e_features = self._processor(x, edge_index, e_features)
        x = self._decoder(x)
        return x

class Simulator(nn.Module):
    def __init__(
        self,
        particle_dimension,
        node_in,
        edge_in,
        latent_dim,
        num_message_passing_steps,
        mlp_num_layers,
        mlp_hidden_dim,
        connectivity_radius,
        noise_std,
        dataset_path,
        num_particle_types,
        particle_type_embedding_size,
        device='cuda',
    ):
        super(Simulator, self).__init__()
        self._connectivity_radius = connectivity_radius
        self._num_particle_types = num_particle_types
        self.noise_std = noise_std
        self.metadata = load_metadata(Path(dataset_path))
        self._boundaries = self.metadata["bounds"]
        self._pbc = self.metadata["periodic_boundary_conditions"]
        self._device = device
        self._particle_type_embedding = nn.Embedding(num_particle_types, particle_type_embedding_size) # (9, 16)

        self._encode_process_decode = EncodeProcessDecode(
            node_in=node_in,
            node_out=particle_dimension,
            edge_in=edge_in,
            latent_dim=latent_dim,
            num_message_passing_steps=num_message_passing_steps,
            mlp_num_layers=mlp_num_layers,
            mlp_hidden_dim=mlp_hidden_dim,
        
        )
        
    def set_metadata_device(self, device=None):
        if device is None:
            device = self._device
        self.normalization_stats = {
            "acceleration": {
                "mean": torch.FloatTensor(self.metadata["acc_mean"]).to(device),
                "std": torch.sqrt(
                    torch.FloatTensor(self.metadata["acc_std"]) ** 2 + self.noise_std**2
                ).to(device),
            },
            "velocity": {
                "mean": torch.FloatTensor(self.metadata["vel_mean"]).to(device),
                "std": torch.sqrt(
                    torch.FloatTensor(self.metadata["vel_std"]) ** 2 + self.noise_std**2
                ).to(device),
            },
        }
        self._boundaries = (
            torch.tensor(self.metadata["bounds"], requires_grad=False).float().to(device)
        )
        #Preprocess the boundaries
        self._boundaries = self._boundaries[:,1] - self._boundaries[:,0]
    
    def forward(self):
        pass

    def _build_graph_from_raw(self, position_sequence, n_particles_per_trajectory, particle_types, pbc=True, batch_size=1):
        n_total_points = position_sequence.shape[0]
        most_recent_position = position_sequence[:, -1] # (n_nodes, 2)
        velocity_sequence = time_diff(position_sequence, self._boundaries, pbc)
        # senders and receivers are integers of shape (E,)

        if not pbc:
            senders, receivers = self._compute_connectivity(most_recent_position, 
                                                            n_particles_per_trajectory, 
                                                            self._connectivity_radius)
        elif pbc:
            #PBC Compatible Implementation ver 1: "Simulate time-integrated coarse-grained MD with multi-scale graph neural networks"
            # single_cell = ((self._boundaries[:,1] - self._boundaries[:,0]).to(self._device)).unsqueeze(0)
            stacked_cells = self._boundaries.repeat(batch_size, 1)  
            #O(n) implementation
            # senders, receivers, displacements, distances, edge_types = compute_connectivity_pbc_celllist(positions=most_recent_position, 
            #                                                         box=self._boundaries, 
            #                                                         radius=self._connectivity_radius, 
            #                                                         bonds=None, 
            #                                                         add_self_edges=False)
            #O(n^2) implementation
            # senders, receivers, displacements, distances, edge_type = self.compute_connectivity_pbc(positions=most_recent_position, 
            #                                                                                         lattices=stacked_cells, 
            #                                                                                         n_node=n_particles_per_trajectory,
            #                                                                                         radius=self._connectivity_radius, 
            #                                                                                         bonds=None, 
            #                                                                                         add_self_edges=False)
            
            #Pytorch Geometric Implementation
            senders, receivers = self._compute_connecitivity_pbc_pyg(most_recent_position, 
                                                            n_particles_per_trajectory, 
                                                            self._connectivity_radius)

        node_features = []
        
        # Normalized velocity sequence, merging spatial an time axis.
        velocity_stats = self.normalization_stats["velocity"]
        normalized_velocity_sequence = (velocity_sequence - velocity_stats['mean']) / velocity_stats['std']
        flat_velocity_sequence = normalized_velocity_sequence.view(n_total_points, -1)
        node_features.append(flat_velocity_sequence)

        # Normalized clipped distances to lower and upper boundaries, if not PBC.
        if not pbc:
            # Normalized clipped distances to lower and upper boundaries.
            # boundaries are an array of shape [num_dimensions, 2], where the second
            # axis, provides the lower/upper boundaries.
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

        # if not pbc:
        # Relative displacement and distances normalized to radius
        # (E, 2)
        # normalized_relative_displacements = (
        #     torch.gather(most_recent_position, 0, senders) - torch.gather(most_recent_position, 0, receivers)
        # ) / self._connectivity_radius
        normalized_relative_displacements = (
            most_recent_position[senders, :] - most_recent_position[receivers, :]
        ) / self._connectivity_radius
        edge_features.append(normalized_relative_displacements)

        normalized_relative_distances = torch.norm(normalized_relative_displacements, dim=-1, keepdim=True)
        edge_features.append(normalized_relative_distances)
            
        # elif pbc:
        #     edge_features.append(displacements) # displacements calculated in compute_connectivity_pbc
        #     edge_features.append(distances) # distances calculated in compute_connectivity_pbc
        
        # else:
        #     raise ValueError("Invalid periodic boundary condition type.")

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
         #Default is 2 examples per batch
         # # radius = radius + 0.00001 # radius_graph takes r < radius not r <= radius
        radius = radius + 1e-5
        combined_positions = pbc_duplication(most_recent_position, self._boundaries)
        n_particles_per_trajectory = torch.tensor([combined_positions.shape[0]], requires_grad=False).to(self._device)
        batch_ids = torch.cat([torch.LongTensor([i for _ in range(n)]) for i, n in enumerate(n_particles_per_trajectory)]).to(self._device)
        edge_index = radius_graph(x=combined_positions, r=radius, batch=batch_ids, loop=add_self_edges) # (2, n_edges)
        # Filter edges to only keep those where the source is in the original frame
        original_frame_size = most_recent_position.size(0)
        is_from_original = edge_index[0] < original_frame_size #Mask all particles not from the original frame
        filtered_edge_index = edge_index[:, is_from_original]
        # filtered_edge_index = filtered_edge_index[1, :] % frame_tensor.size(0)
        filtered_edge_index = filtered_edge_index % most_recent_position.size(0)
        sorted_receiver, order = filtered_edge_index[1,:].sort()
        filtered_edge_index[1,:] = sorted_receiver
        filtered_edge_index[0,:] = filtered_edge_index[0,:][order]
        receivers = filtered_edge_index[0, :]
        senders = filtered_edge_index[1, :]
        return receivers, senders
    
    def _decoder_postprocessor(self, normalized_acceleration, position_sequence, pbc=True):
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
        acceleration_stats = self.normalization_stats["acceleration"]
        acceleration = (
            normalized_acceleration * acceleration_stats['std']
        ) + acceleration_stats['mean']

        # Use an Euler integrator to go from acceleration to position, assuming dt = 1.
        most_recent_position = position_sequence[:, -1]
        most_recent_velocity = (most_recent_position - position_sequence[:, -2])
        if pbc:
            most_recent_velocity = wrap_displacement(most_recent_velocity, self._boundaries)

        # Update velocity and position
        new_velocity = most_recent_velocity + acceleration  # * dt = 1
        new_position = most_recent_position + new_velocity  # * dt = 1

        if pbc:
        # Wrap new positions within boundaries
            new_position = wrap_position(new_position, self._boundaries)
            
        return new_position

    def predict_positions(self, current_positions, n_particles_per_trajectory, particle_types, pbc=True, batch_size=1):
        if pbc:
            current_positions = current_positions % self._boundaries
        node_features, edge_index, e_features = self._build_graph_from_raw(current_positions, n_particles_per_trajectory, particle_types, pbc, batch_size)
        predicted_normalized_acceleration = self._encode_process_decode(node_features, edge_index, e_features)
        next_position = self._decoder_postprocessor(predicted_normalized_acceleration, current_positions, pbc)
        return next_position

    def predict_accelerations(self, next_position, position_sequence_noise, position_sequence, n_particles_per_trajectory, particle_types, pbc=True, batch_size=1, unroll_steps=0):
        #FIXME: next_position needs to drop dim=1 because the dataloader is configured for both training and validation, but for training this dimension is not necessary.
        next_position = next_position.squeeze(1)
        if pbc:
            noisy_position_sequence = wrap_position(position_sequence + position_sequence_noise, self._boundaries)
            next_position_adjusted = wrap_position(next_position + position_sequence_noise[:, -1], self._boundaries)
        elif not pbc:
            noisy_position_sequence = position_sequence + position_sequence_noise
            next_position_adjusted = next_position + position_sequence_noise[:, -1]
        else: 
            raise ValueError("Invalid periodic boundary condition type.")
        
        node_features, edge_index, e_features = self._build_graph_from_raw(noisy_position_sequence, n_particles_per_trajectory, particle_types, pbc, batch_size)
        predicted_normalized_acceleration = self._encode_process_decode(node_features, edge_index, e_features)

        #Compute the target normalized acceleration
        target_normalized_acceleration = self._inverse_decoder_postprocessor(next_position_adjusted, noisy_position_sequence, pbc)
        
        return predicted_normalized_acceleration, target_normalized_acceleration
    
    #PBC COMPATIBLE IMPLEMENTATION
    def _inverse_decoder_postprocessor(self, next_position, position_sequence, pbc=True):
        """Inverse of `_decoder_postprocessor`, with PBC support."""
        #TODO: Go over the shapes with Artur
        # Handle PBC for position differences
        previous_position = position_sequence[:, -1]
        if pbc:
            previous_velocity = wrap_displacement(previous_position - position_sequence[:, -2], self._boundaries)
            next_velocity = wrap_displacement(next_position - previous_position, self._boundaries)
        elif not pbc:
            previous_velocity = previous_position - position_sequence[:, -2]
            next_velocity = next_position - previous_position
        
        # Compute acceleration
        acceleration = next_velocity - previous_velocity
        # Normalize acceleration (reverse normalization for loss calculation)
        acceleration_stats = self.normalization_stats["acceleration"]        
        normalized_acceleration = (acceleration - acceleration_stats['mean']) / acceleration_stats['std'] 
        
        return normalized_acceleration 