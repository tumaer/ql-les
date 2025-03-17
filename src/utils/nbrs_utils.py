import copy
import torch
import numpy as np
from torch_geometric.nn import radius_graph


def pos_init_cartesian_2d(box_size: np.ndarray, dx: float):
    """Create a grid of particles in 2D.

    Particles are at the center of the corresponding Cartesian grid cells.
    Example: if box_size=np.array([1, 1]) and dx=0.1, then the first particle will be at
    position [0.05, 0.05].
    """
    n = np.array((box_size / dx).round(), dtype=int)
    grid = np.meshgrid(range(n[0]), range(n[1]), indexing="xy")
    r = (np.vstack(list(map(np.ravel, grid))).T + 0.5) * dx
    return r


def pos_init_cartesian_3d(box_size: np.ndarray, dx: float):
    """Create a grid of particles in 3D."""
    n = np.array((box_size / dx).round(), dtype=int)
    grid = np.meshgrid(range(n[0]), range(n[1]), range(n[2]), indexing="xy")
    r = (np.vstack(list(map(np.ravel, grid))).T + 0.5) * dx
    return r


def wrap_displacement(displacement, boundaries):
    """
    Wrap displacement to account for periodic boundary conditions.

    Args:
        displacement: Displacement tensor.
        boundaries: Tensor containing boundary conditions.

    Returns:
        Wrapped displacement tensor.
    """
    #floating point percision messes up the results
    return (displacement + 0.5 * boundaries) % boundaries - 0.5 * boundaries


def wrap_position(position, boundaries):
    """
    Wrap position to account for periodic boundary conditions.

    Args:
        position: Position tensor.
        boundaries: Tensor containing boundary conditions.

    Returns:
        Wrapped position tensor.
    """
    return position % boundaries


def shift_fn(r, dr, box=None, pbc=False):
    if pbc:
        return wrap_position(r + dr, box)
    else:
        return r + dr


def displ_fn(r1, r2, box=None, pbc=False):
    if pbc:
        return wrap_displacement(r1 - r2, box)
    else:
        return r1 - r2


def radius_graph_pbc(most_recent_position, n_particles_per_trajectory, radius, box, add_self_edges=True):
    #Default is 2 examples per batch
    # radius = radius + 0.00001 # radius_graph takes r < radius not r <= radius
    n_particles_copy = copy.deepcopy(n_particles_per_trajectory)
    combined_positions, n_particles_per_trajectory_combined = pbc_duplication(most_recent_position, n_particles_copy, box)
    batch_ids = torch.cat([torch.LongTensor([i for _ in range(n)]) for i, n in enumerate(n_particles_per_trajectory_combined)]).to(most_recent_position.device)
    edge_index = radius_graph(x=combined_positions, r=radius, batch=batch_ids, loop=add_self_edges)# (2, n_edges)
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


def pbc_duplication(most_recent_positions, n_particles_per_trajectory_combined, domain_size):
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
        n_particles_per_trajectory_combined *= 9

    elif D == 3:
        pbc = [True, True, True]
        n_particles_per_trajectory_combined *= 27

    # Create shifts based on pbc
    shift_ranges = [
        [-1, 0, 1] if p else [0] for p in pbc
    ]
    shifts = torch.cartesian_prod(*[torch.tensor(r) for r in shift_ranges]).float()

    # Scale shifts by the domain size
    shifts = (shifts * domain_size).to(most_recent_positions.device)

    # Apply shifts to generate duplicated frames
    duplicated_frames = torch.cat([
        most_recent_positions + shift for shift in shifts
    ], dim=0).to(most_recent_positions.device)  # Shape: [N * len(shifts), D]

    # Combine original frame and duplicated frames
    combined_positions = torch.cat([most_recent_positions, duplicated_frames], dim=0).to(most_recent_positions.device)
    return combined_positions, n_particles_per_trajectory_combined
