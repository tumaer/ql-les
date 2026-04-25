import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from src.utils.train_utils import wrap_displacement


EPSILON = 1e-8


def build_mlp(
    input_size,
    layer_sizes,
    output_size=None,
    output_activation=torch.nn.Identity,
    activation=torch.nn.ReLU,
):
    """Builds an MLP with the specified layer sizes and activations."""
    sizes = [input_size] + layer_sizes
    if output_size:
        sizes.append(output_size)

    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [torch.nn.Linear(sizes[i], sizes[i + 1]), act()]
    return torch.nn.Sequential(*layers)


def time_diff(input_sequence, boundaries, pbc=True):
    """Returns the time difference between two consecutive frames in a sequence."""
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
    velocity_sequence_shape = list(position_sequence.shape)
    velocity_sequence_shape[1] -= 1
    n_velocities = velocity_sequence_shape[1]

    velocity_sequence_noise = torch.randn(velocity_sequence_shape) * (
        noise_std_last_step / n_velocities**0.5
    )
    velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)

    position_sequence_noise = torch.cat(
        [
            torch.zeros_like(velocity_sequence_noise[:, 0:1]),
            torch.cumsum(velocity_sequence_noise, dim=1),
        ],
        dim=1,
    )

    return position_sequence_noise


class Encoder(nn.Module):
    """Encoder module for the graph neural network."""

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
        self.node_fn = nn.Sequential(
            *[
                build_mlp(node_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out),
                nn.LayerNorm(node_out),
            ]
        )
        self.edge_fn = nn.Sequential(
            *[
                build_mlp(edge_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], edge_out),
                nn.LayerNorm(edge_out),
            ]
        )

    def forward(self, x, edge_index, e_features):  # global_features
        """Encodes the node and edge features."""
        # x: (E, node_in)
        # edge_index: (2, E)
        # e_features: (E, edge_in)
        return self.node_fn(x), self.edge_fn(e_features)


class InteractionNetwork(MessagePassing):
    """Interaction network module for the graph neural network."""

    def __init__(
        self,
        node_in,
        node_out,
        edge_in,
        edge_out,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(InteractionNetwork, self).__init__(aggr="add")
        self.node_fn = nn.Sequential(
            *[
                build_mlp(
                    node_in + edge_out, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out
                ),
                nn.LayerNorm(node_out),
            ]
        )
        self.edge_fn = nn.Sequential(
            *[
                build_mlp(
                    node_in + node_in + edge_in,
                    [mlp_hidden_dim for _ in range(mlp_num_layers)],
                    edge_out,
                ),
                nn.LayerNorm(edge_out),
            ]
        )

    def forward(self, x, edge_index, e_features):
        """Forward pass of the interaction network."""
        # x: (E, node_in)
        # edge_index: (2, E)
        # e_features: (E, edge_in)
        x_residual = x
        e_features_residual = e_features
        x, e_features = self.propagate(edge_index=edge_index, x=x, e_features=e_features)
        return x + x_residual, e_features + e_features_residual

    def message(self, edge_index, x_i, x_j, e_features):
        """Message construction for the interaction network."""
        e_features = torch.cat([x_i, x_j, e_features], dim=-1)
        e_features = self.edge_fn(e_features)
        return e_features

    def update(self, x_updated, x, e_features):
        """Node update for the interaction network."""
        # x_updated: (E, edge_out)
        # x: (E, node_in)
        x_updated = torch.cat([x_updated, x], dim=-1)
        x_updated = self.node_fn(x_updated)
        return x_updated, e_features


class Processor(MessagePassing):
    """Processor module for the graph neural network."""

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
        super(Processor, self).__init__(aggr="max")
        self.gnn_stacks = nn.ModuleList(
            [
                InteractionNetwork(
                    node_in=node_in,
                    node_out=node_out,
                    edge_in=edge_in,
                    edge_out=edge_out,
                    mlp_num_layers=mlp_num_layers,
                    mlp_hidden_dim=mlp_hidden_dim,
                )
                for _ in range(num_message_passing_steps)
            ]
        )

    def forward(self, x, edge_index, e_features):
        """Forward pass of the processor."""
        for gnn in self.gnn_stacks:
            x, e_features = gnn(x, edge_index, e_features)
        return x, e_features


class ZeroLevelAggregation(MessagePassing):
    """Interaction network module for the graph neural network with zero-level aggregation."""

    def __init__(
        self,
        node_in,
        node_out,
        edge_in,
        edge_out,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(ZeroLevelAggregation, self).__init__(aggr="add")
        self.node_fn = nn.Sequential(
            *[
                build_mlp(
                    node_in + edge_out, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out
                ),
                nn.LayerNorm(node_out),
            ]
        )
        self.edge_fn = nn.Sequential(
            *[
                build_mlp(
                    node_in + node_in + edge_in,
                    [mlp_hidden_dim for _ in range(mlp_num_layers)],
                    edge_out,
                ),
                nn.LayerNorm(edge_out),
            ]
        )

    def forward(self, x_0, x_t, edge_index, e_features):
        """Forward pass of the zero-level aggregation network."""
        x_residual = x_t
        e_features_residual = e_features
        x_t, e_features = self.propagate(edge_index, x_0, x_t, e_features)
        return x_t + x_residual, e_features + e_features_residual

    def propagate(self, edge_index, x_0, x_t, e_features, size=None):
        """Propagates the interaction (message) through the graph."""
        # Message
        out, x_i_index = self.message(x_0, x_t, edge_index, e_features)

        # Aggregation
        # Aggregation
        out = self.aggregate(out, index=x_i_index, ptr=None, dim_size=x_0.shape[0])

        # Node update

        # Node update
        x_updated = torch.cat([out, x_t], dim=-1)
        x_updated = self.node_fn(x_updated)
        return x_updated, e_features

    def message(self, x_0, x_t, edge_index, e_features, flow="source_to_target"):
        """Pseudo-message construction for the zero-level aggregation network."""
        i, j = (1, 0) if flow == "source_to_target" else (0, 1)
        x_i = x_t[edge_index[i]]
        x_j = x_0[edge_index[j]]
        e_features = torch.cat([x_i, x_j, e_features], dim=-1)
        e_features = self.edge_fn(e_features)
        return e_features, edge_index[i]


class LocalProcessor(nn.Module):
    def __init__(
        self,
        node_in,
        node_out,
        edge_in,
        edge_out,
        num_interaction_steps,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(LocalProcessor, self).__init__()
        self.gnn_stacks = nn.ModuleList(
            [
                ZeroLevelAggregation(
                    node_in=node_in,
                    node_out=node_out,
                    edge_in=edge_in,
                    edge_out=edge_out,
                    mlp_num_layers=mlp_num_layers,
                    mlp_hidden_dim=mlp_hidden_dim,
                )
                for _ in range(num_interaction_steps)
            ]
        )

    def forward(self, x, edge_index, e_features):
        """Forward pass of the local processor."""
        x_0, x_t = x, x
        for gnn in self.gnn_stacks:
            x_t, e_features = gnn(x_0, x_t, edge_index, e_features)
        return x, e_features


class Decoder(nn.Module):
    """Decoder module for the graph neural network."""

    def __init__(
        self,
        node_in,
        node_out,
        mlp_num_layers,
        mlp_hidden_dim,
    ):
        super(Decoder, self).__init__()
        self.node_fn = build_mlp(
            node_in, [mlp_hidden_dim for _ in range(mlp_num_layers)], node_out
        )

    def forward(self, x):
        """Decodes the features."""
        # x: (E, node_in)
        return self.node_fn(x)


class EncodeProcessDecode(nn.Module):
    """Graph Network Simulator (GNS) by Sanchez-Gonzalez et al. (2020)."""

    def __init__(
        self,
        node_in,
        node_out,
        edge_in,
        latent_dim,
        num_message_passing_steps,
        mlp_num_layers,
        mlp_hidden_dim,
        alpha_u,
        local_interaction=False,
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
        if not local_interaction:
            self._processor = Processor(
                node_in=latent_dim,
                node_out=latent_dim,
                edge_in=latent_dim,
                edge_out=latent_dim,
                num_message_passing_steps=num_message_passing_steps,
                mlp_num_layers=mlp_num_layers,
                mlp_hidden_dim=mlp_hidden_dim,
            )
        else:
            self._processor = LocalProcessor(
                node_in=latent_dim,
                node_out=latent_dim,
                edge_in=latent_dim,
                edge_out=latent_dim,
                num_interaction_steps=num_message_passing_steps,
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

    def _transform(self, x, edge_index, e_features):
        """Transforms the input data into the required format for the GNS."""
        node_features = [
            x[k]
            for k in [  # define a fixed order for the node features
                "v_flat_velocity_sequence",
                "u_flat_velocity_sequence",
                "normalized_clipped_distance_to_boundaries",
                "particle_type_embeddings",
            ]
            if k in x
        ]

        edge_features = [
            e_features[k]
            for k in [  # define a fixed order for the node features
                "normalized_relative_displacements",
                "normalized_relative_distances",
                "normalized_relative_velocities",
                "normalized_relative_velocity_distances",
            ]
            if k in e_features
        ]

        return torch.cat(node_features, dim=-1), edge_index, torch.cat(edge_features, dim=-1)

    def forward(self, x, edge_index, e_features):
        """Forward pass of the GNS."""
        # x: (E, node_in)
        x, edge_index, e_features = self._transform(x, edge_index, e_features)
        x, e_features = self._encoder(x, edge_index, e_features)
        x, e_features = self._processor(x, edge_index, e_features)
        x = self._decoder(x)
        # preserve linear momentum
        x -= x.mean(dim=0, keepdim=True)
        if self.alpha_u != 0:
            a_v, a_u = torch.chunk(x, 2, dim=-1)
            return a_v, a_u
        else:
            return x
