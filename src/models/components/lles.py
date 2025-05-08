from torch_geometric.nn import knn, radius
import numpy as np
import torch
import torch.nn as nn
from torch_scatter import scatter_add


# -------------------------------------------------------------------------
def make_grid(nfield, dim):
    x = np.linspace(0,2*np.pi,nfield+1)
    y = x
    if dim == 3:
        z = x
        grid = np.meshgrid(x[:nfield],y[:nfield],z[:nfield],indexing='ij')
        grid = np.stack([grid[0],grid[1],grid[2]]).transpose([1,2,3,0])
        grid = grid.reshape([-1,3])
    elif dim == 2:
        grid = np.meshgrid(x[:nfield],y[:nfield],indexing='ij')
        grid = np.stack([grid[0],grid[1]]).transpose([1,2,0])
        grid = grid.reshape([-1,2])
        
    return grid

def build_custom_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    activation: str = "tanh",
    output_activation: type = nn.Identity,
    use_layer_norm: bool = False,
) -> nn.Sequential:
    """
    Builds a configurable MLP with named layers like self.l1, self.l2, etc.

    Args:
        input_dim: Dimension of the input layer.
        hidden_dims: List of hidden layer sizes.
        output_dim: Size of the output layer.
        activation: Activation function to apply after each hidden layer.
        output_activation: Activation function to apply after final layer.
        use_layer_norm: Whether to apply LayerNorm after each linear layer.

    Returns:
        nn.Sequential model composed of the specified layers.
    """
    activations = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "sigmoid": nn.Sigmoid,
        }
    activation = activations.get(activation)
    
    layers = []
    layer_dims = [input_dim] + hidden_dims + [output_dim]
    
    for i in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
        if i < len(layer_dims) - 2:  # Hidden layers
            if use_layer_norm:
                layers.append(nn.LayerNorm(layer_dims[i+1]))
            layers.append(activation())
        else:  # Output layer
            layers.append(output_activation())

    return nn.Sequential(*layers)    

class LLES(nn.Module):
    """
    Lagrangian Large Eddy Simulation (L-LES) model for particle-based turbulence simulation.

    Uses neural network to approximate subgrid-scale interactions and 
    compute density, velocity, and acceleration evolution for a set of particles.

    Attributes:
        input_features (int): Number of input features for the neural network.
        hidden_dims (list): List of hidden layer sizes for the neural network.
        output_dim (int): Size of the output layer for the neural network.
        activation (str): Activation function to use in the neural network.
        num_particles (int): Maximum number of particles in the simulation.
        av_activation (nn.Module): Activation function for artificial viscosity.
        alpha1 (nn.Parameter): Parameter for artificial viscosity.
        alpha2 (nn.Parameter): Parameter for artificial viscosity.
    """

    def __init__(self,
                input_features,
                hidden_dims,
                output_dim,
                activation,
                metadata):
        """
        Initializes the L-LES model.

        Args:
            device (torch.device): Computational device (CPU/GPU).
            N (int): Number of Lagrangian particles.
            dt (float): Time step for numerical integration.
            uref (float): Reference velocity scale.
            tref (float): Reference time scale.
            grid_points_per_dim (int): Number of Eulerian grid points.
            kernel_wnn (Kernel): Learned smoothing kernel.
            max_num_neighbors (int): Number of nearest neighbors per particle.
        """
        super(LLES, self).__init__()
        self.input_features = input_features
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        self.activation = activation
        
        self.neural_net = build_custom_mlp(
            self.input_features,
            self.hidden_dims,
            self.output_dim,
            activation=activation)
        
        self.num_particles = metadata["num_particles_max"]
        
        # #Artificial viscosity parameters
        self.av_activation = nn.Tanh()
        self.alpha1 = nn.Parameter(torch.tensor(0.1),requires_grad=True)
        self.alpha2 = nn.Parameter(torch.tensor(0.1),requires_grad=True)
        
        self.output_scale = nn.Parameter(torch.tensor(1.0))


    
    def _calculate_artificial_viscosity(self, xx, xv, vec, art_visc_h):
        """
        Compute artificial viscosity correction term (no density).

        Args:
            xx (Tensor): |r| values [n_edges]
            xv (Tensor): r̂ · v̂ values [n_edges]
            vec (Tensor): r̂ vectors [n_edges, D]
            art_visc_h (float): Smoothing length

        Returns:
            Tensor: [n_edges, D] artificial viscosity contribution
        """
        denom = xx ** 2 + 0.1 * art_visc_h ** 2
        visc = -art_visc_h * self.av_activation(-xv) / denom
        visc = -self.alpha1.abs() * visc + self.alpha2.abs() * visc ** 2
        return visc.unsqueeze(1) * vec

    
    def forward(self, edge_features, edge_directions, art_visc_h, senders):
        """
        Predicts accelerations using learned interactions and artificial viscosity.

        Args:
            edge_features (Tensor): [n_edges, F] from _build_lles_graph
            edge_directions (Tensor): [n_edges, 2D] concatenated [r̂, v̂]
            art_visc_h (float): Smoothing length
            receivers (LongTensor): Edge destination indices

        Returns:
            normalized_a_v (Tensor): [n_nodes, D]
            normalized_a_u (Tensor): [n_nodes, D]
        """
        knn_out = self.neural_net(edge_features)
        a_v, a_u = torch.chunk(knn_out, 2, dim=1)

        D = a_v.shape[1]
        direction = edge_directions.view(-1, 2, D)  # [n_edges, 2, D]
        
        # Project predicted acceleration along directions
        a_v_proj = (a_v.view(-1, 2, 1) * direction).sum(dim=1)
        a_u_proj = (a_u.view(-1, 2, 1) * direction).sum(dim=1)

        #Artificial viscosity
        av = self._calculate_artificial_viscosity(
            xx=edge_features[:, 0],
            xv=edge_features[:, -1],
            vec=direction[:, 0],  # r̂
            art_visc_h=art_visc_h
        )

        a_v_proj += av
        a_u_proj += av

        # Aggregate to particles
        node_a_v = scatter_add(a_v_proj, senders, dim=0, dim_size=self.num_particles)
        node_a_u = scatter_add(a_u_proj, senders, dim=0, dim_size=self.num_particles)

        return node_a_v, node_a_u