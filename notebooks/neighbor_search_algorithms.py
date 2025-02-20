import torch
import numpy as np
# from torch_geometric import radius_graph
from torch_scatter import scatter, segment_coo, segment_csr
# from torchCompactRadius import radiusSearch
from src.utils.nbrs_utils import wrap_displacement, wrap_position
# from torch_geometric.utils import radius_graph #TODO: Fix this import error

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

#TODO:(Learning to Simulate Time-integrated Coarse-grained Molecular Dynamics with Multi-scale Graph Networks) PBC Compatible Implementation 
EPSILON = 1e-6

def compute_connectivity_pbc(self, positions, lattices, n_node, radius, 
                            bonds=None, add_self_edges=True):
    """
    construct graph using radius cut-off under periodic boundary conditions.
    simplified computation of distances/displacements under PBC is possible
    due to the lattices being squares/cubes.
    Inputs:
        positions: N x dim
        lattices: B x dim
        n_node: B x 1
    Optional Inputs:
        bonds: E x 2. include these bonds even if they don't fall into the radius cut-off.
    Outputs:
        senders, receivers, displacements, distances, edge_type (1 if is_bond)
    """
    n_node_per_image_sqr = (n_node ** 2).long()

    # index offset between images
    index_offset = (torch.cumsum(n_node, dim=0) - n_node)
    index_offset_expand = torch.repeat_interleave(index_offset, n_node_per_image_sqr)
    n_node_per_image_expand = torch.repeat_interleave(n_node, n_node_per_image_sqr)

    # Compute a tensor containing sequences of numbers that range 
    # from 0 to n_node_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms.
    num_atom_pairs = torch.sum(n_node_per_image_sqr)
    index_sqr_offset = (torch.cumsum(n_node_per_image_sqr, dim=0) - n_node_per_image_sqr)
    index_sqr_offset = torch.repeat_interleave(index_sqr_offset, n_node_per_image_sqr)
    atom_count_sqr = (torch.arange(num_atom_pairs, device=positions.device) - index_sqr_offset)

    # Compute the indices for the pairs of atoms (using division and mod)
    index1 = ((atom_count_sqr.div(n_node_per_image_expand, rounding_mode='floor'))).long() + index_offset_expand
    index2 = (atom_count_sqr % n_node_per_image_expand).long() + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(positions, 0, index1)
    pos2 = torch.index_select(positions, 0, index2)

    lattices = lattices.repeat_interleave(n_node_per_image_sqr, dim=0)
    displacement = pos1 - pos2  
    lattices = lattices.expand_as(displacement)
    displacement = torch.where(displacement > (0.5 * lattices),
                                displacement - lattices, displacement)
    displacement = torch.where(displacement < (-0.5 * lattices),
                                displacement + lattices, displacement)
    displacement = displacement / radius
    distance = displacement.norm(dim=-1)

    mask = torch.le(distance, 1)
    if not add_self_edges:
        mask_not_same = torch.gt(distance, EPSILON)
        mask = torch.logical_and(mask, mask_not_same)
    
    if bonds is not None:
        all_edges = torch.cat([index1[:, None], index2[:, None]], dim=1)
        all_edges_non_unique = torch.cat([all_edges, bonds], dim=0)
        all_edges, counts = torch.unique(all_edges_non_unique, dim=0, return_counts=True)
        bond_mask = (counts > 1)  # this is correct because <all_edges> enumerates all edges.
        mask = torch.logical_or(mask, bond_mask)  # include both raidus-edges and bonds.
        # get edge type.
        all_edge_indices, counts = torch.unique(
            torch.cat([mask.nonzero(), bond_mask.nonzero()], dim=0), return_counts=True)
        edge_types = (counts > 1).int()
    else:
        edge_types = None
    # senders, receivers, displacements, distances, edge_type (is_bond)
    return (index1[mask], index2[mask], displacement[mask], 
            distance[mask].unsqueeze(1), edge_types)
    
#TODO: FAIRCHEM CODE (tensor memomy size inefficient)
def _compute_connectivity(self, node_features, radius, n_particles_per_trajectory):
        
        boundaries = torch.tensor(self.metadata["bounds"]).to(self._device)
        box_size = boundaries[0, 1] - boundaries[0, 0]
        
        node_features = torch.cat((node_features, torch.full((n_particles_per_trajectory, 1), 0.5).to(self._device)), dim=1)
        
        cell = torch.tensor([
            [box_size, 0.0, 0.0],
            [0.0, box_size, 0.0],
            [0.0, 0.0, box_size],  # Placeholder for z-dimension
        ], dtype=torch.float32).unsqueeze(0).to(self._device)  # (1, 3, 3) for batch size 1
        
        pbc = [True, True, True]
        
        class Data:
            def __init__(self, pos, natoms, cell, pbc):
                self.pos = pos
                self.natoms = natoms
                self.cell = cell
                self.pbc = torch.tensor(pbc, dtype=torch.bool)

        data = Data(node_features, n_particles_per_trajectory, cell, pbc)

        edge_index = radius_graph_pbc(data, 
                                      radius,  
                                      max_num_neighbors_threshold=32,
                                      pbc=pbc) # (2, n_edges) 
        receivers = edge_index[0, :]
        senders = edge_index[1, :]
        return receivers, senders
    
def radius_graph_pbc(
    data,
    radius,
    max_num_neighbors_threshold,
    enforce_max_neighbors_strictly: bool = False,
    pbc=None,
):
    if pbc is None:
        pbc = [True, True, True]
    device = data.pos.device
    batch_size = len(data.natoms)

    if hasattr(data, "pbc"):
        data.pbc = torch.atleast_2d(data.pbc)
        for i in range(3):
            if not torch.any(data.pbc[:, i]).item():
                pbc[i] = False
            elif torch.all(data.pbc[:, i]).item():
                pbc[i] = True
            else:
                raise RuntimeError(
                    "Different structures in the batch have different PBC configurations. This is not currently supported."
                )

    # position of the atoms
    atom_pos = data.pos

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = data.natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # index offset between images
    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image

    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor")
    ) + index_offset_expand
    index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).

    cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
    cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = data.cell.new_zeros(1)

    if pbc[1]:
        cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = data.cell.new_zeros(1)

    if pbc[2]:
        cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = data.cell.new_zeros(1)

    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = [rep_a1.max(), rep_a2.max(), rep_a3.max()]

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep.item(), rep.item() + 1, device=device, dtype=torch.float)
        for rep in max_rep
    ]
    unit_cell = torch.cartesian_prod(*cells_per_dim)
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(data.cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=data.natoms,
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        enforce_max_strictly=enforce_max_neighbors_strictly,
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(
            unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image

def get_max_neighbors_mask(
    natoms,
    index,
    atom_distance,
    max_num_neighbors_threshold,
    degeneracy_tolerance: float = 0.01,
    enforce_max_strictly: bool = False,
):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.

    Enforcing the max strictly can force the arbitrary choice between
    degenerate edges. This can lead to undesired behaviors; for
    example, bulk formation energies which are not invariant to
    unit cell choice.

    A degeneracy tolerance can help prevent sudden changes in edge
    existence from small changes in atom position, for example,
    rounding errors, slab relaxation, temperature, etc.
    """

    device = natoms.device
    num_atoms = natoms.sum()

    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clamp(max=max_num_neighbors_threshold)

    # Get number of (thresholded) neighbors per image
    image_indptr = torch.zeros(natoms.shape[0] + 1, device=device, dtype=torch.long)
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor([True], dtype=bool, device=device).expand_as(
            index
        )
        return mask_num_neighbors, num_neighbors_image

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full([num_atoms * max_num_neighbors], np.inf, device=device)

    # Create an index map to map distances from atom_distance to distance_sort
    # index_sort_map assumes index to be sorted
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + torch.arange(len(index), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)

    # Select the max_num_neighbors_threshold neighbors that are closest
    if enforce_max_strictly:
        distance_sort = distance_sort[:, :max_num_neighbors_threshold]
        index_sort = index_sort[:, :max_num_neighbors_threshold]
        max_num_included = max_num_neighbors_threshold

    else:
        effective_cutoff = (
            distance_sort[:, max_num_neighbors_threshold] + degeneracy_tolerance
        )
        is_included = torch.le(distance_sort.T, effective_cutoff)

        # Set all undesired edges to infinite length to be removed later
        distance_sort[~is_included.T] = np.inf

        # Subselect tensors for efficiency
        num_included_per_atom = torch.sum(is_included, dim=0)
        max_num_included = torch.max(num_included_per_atom)
        distance_sort = distance_sort[:, :max_num_included]
        index_sort = index_sort[:, :max_num_included]

        # Recompute the number of neighbors
        num_neighbors_thresholded = num_neighbors.clamp(max=num_included_per_atom)

        num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # Offset index_sort so that it indexes into index
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_included
    )
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    mask_num_neighbors.index_fill_(0, index_sort, True)

    return mask_num_neighbors, num_neighbors_image

#TODO: PERSONAL ATTEMPT PBC Compatible Implementation PERSONAL  
def cell_list_neighbor_search(
    positions: torch.Tensor,
    box: torch.Tensor = torch.tensor([1., 1.]),
    cutoff: float = 0.1
):
    """
    Cell-list based neighbor search in 2D with periodic boundary conditions (PBC).

    Args:
        positions (torch.Tensor): Shape (N, 2). Particle positions.
        box (torch.Tensor): Shape (2,). Box dimensions, default [1,1].
        cutoff (float): Cutoff distance for neighbor search.

    Returns:
        (senders, receivers): Two 1D tensors listing neighbor pairs (i, j) with i < j.
    """
    # Wrap positions to ensure they are inside [0, box[0]) × [0, box[1])
    positions = positions % box

    # Number of cells along each dimension
    n_cells = (box / cutoff).to(torch.int)
    # Ensure at least 1 cell in each dimension
    n_cells = torch.clamp(n_cells, min=1)

    # Cell size along each dimension
    cell_size = box / n_cells

    # Assign each particle to a cell index in x and y
    cell_indices = torch.floor(positions / cell_size).to(torch.int)

    # Create a list of empty lists to store the particles that fall in each cell
    max_cell_x, max_cell_y = n_cells[0].item(), n_cells[1].item()
    cell_list = [[] for _ in range(max_cell_x * max_cell_y)]

    def flatten_cell_ix(ix_x, ix_y):
        """Convert 2D cell coordinates to single index."""
        return ix_x * max_cell_y + ix_y

    # Populate the cell_list with particle indices
    for i, pos in enumerate(positions):
        cix_x = cell_indices[i, 0]
        cix_y = cell_indices[i, 1]
        cix_flat = flatten_cell_ix(cix_x, cix_y)
        cell_list[cix_flat].append(i)

    # Prepare to collect neighbor pairs
    senders = []
    receivers = []

    # Relative shifts to neighbor cells in 2D (including the cell itself)
    neighbor_shifts = [-1, 0, 1]

    # Check each cell and its neighbors
    for cix_x in range(max_cell_x):
        for cix_y in range(max_cell_y):
            # Indices of particles in the current cell
            cix_flat = flatten_cell_ix(cix_x, cix_y)
            current_particles = cell_list[cix_flat]

            # Loop over all neighboring cells (including the current one)
            for dx in neighbor_shifts:
                for dy in neighbor_shifts:
                    # Wrap neighbor cell indices modulo the number of cells for PBC
                    nx = (cix_x + dx) % max_cell_x
                    ny = (cix_y + dy) % max_cell_y
                    neighbor_particles = cell_list[flatten_cell_ix(nx, ny)]

                    # Build neighbor pairs
                    for i_idx in current_particles:
                        for j_idx in neighbor_particles:
                            # Avoid double-counting (only consider i < j)
                            if i_idx < j_idx:
                                # Compute distance with minimum-image convention
                                dr = positions[j_idx] - positions[i_idx]
                                # Apply nearest image under PBC
                                dr -= box * torch.round(dr / box)
                                dist2 = torch.sum(dr**2)
                                if dist2 < cutoff**2:
                                    senders.append(i_idx)
                                    receivers.append(j_idx)

    # Convert lists to PyTorch tensors
    senders = torch.tensor(senders, dtype=torch.long)
    receivers = torch.tensor(receivers, dtype=torch.long)

    return senders, receivers

def compute_connectivity_pbc_celllist(
    positions: torch.Tensor,
    box: torch.Tensor,
    radius: float,
    bonds: torch.Tensor = None,
    add_self_edges: bool = True
):
    """
    Optimized Cell-list based neighbor search with PBC.
    """
    # 1) Wrap positions to handle PBC
    positions = wrap_position(positions, box)
    N = positions.shape[0]
    dim = positions.shape[1]

    # 2) Build cell list
    cutoff = radius
    n_cells = (box / cutoff).to(torch.int)
    n_cells = torch.clamp(n_cells, min=1)

    cell_size = box / n_cells
    cell_indices = torch.floor(positions / cell_size).to(torch.int)

    # Flatten function for cell indices
    def flatten_ix(ixs):
        return ixs[0] * n_cells[1].item() + ixs[1]

    # Create the cell lists
    max_cell_x, max_cell_y = n_cells[0].item(), n_cells[1].item()
    cell_list = [[] for _ in range(max_cell_x * max_cell_y)]
    for i in range(N):
        cx, cy = cell_indices[i]
        cidx = flatten_ix((cx, cy))
        cell_list[cidx].append(i)

    # 3) For each cell, consider itself & neighbors
    neighbor_shifts = [-1, 0, 1]
    pair_i = []
    pair_j = []

    for cx in range(max_cell_x):
        for cy in range(max_cell_y):
            cidx = flatten_ix((cx, cy))
            current_particles = cell_list[cidx]

            for dx in neighbor_shifts:
                for dy in neighbor_shifts:
                    nx = (cx + dx) % max_cell_x
                    ny = (cy + dy) % max_cell_y
                    nidx = flatten_ix((nx, ny))
                    neighbor_particles = cell_list[nidx]

                    for i_idx in current_particles:
                        for j_idx in neighbor_particles:
                            if i_idx < j_idx:
                                pair_i.append(i_idx)
                                pair_j.append(j_idx)

    # Convert to tensors
    pair_i = torch.tensor(pair_i, dtype=torch.long).to(positions.device)
    pair_j = torch.tensor(pair_j, dtype=torch.long).to(positions.device)

    # 4) Minimum-image displacements & distances
    pos_i = positions[pair_i]
    pos_j = positions[pair_j]
    dr = wrap_displacement(pos_i - pos_j, box)  # Use wrap_displacement for PBC

    # Scale by radius
    dr_scaled = dr / radius
    dist = dr_scaled.norm(dim=-1)  # shape (E,)

    # Remove self-edges if needed
    if not add_self_edges:
        not_self = (pair_i != pair_j)
        dist = dist[not_self]
        dr_scaled = dr_scaled[not_self]
        pair_i = pair_i[not_self]
        pair_j = pair_j[not_self]

    # Filter edges by cutoff
    within_cutoff = (dist <= 1.0)
    pair_i = pair_i[within_cutoff]
    pair_j = pair_j[within_cutoff]
    dr_scaled = dr_scaled[within_cutoff]
    dist = dist[within_cutoff]

    # 5) Handle bonds if provided
    if bonds is not None:
        all_edges_non_unique = torch.cat([
            torch.stack([pair_i, pair_j], dim=1),  # E x 2
            bonds
        ], dim=0)

        all_edges, counts = torch.unique(all_edges_non_unique, dim=0, return_counts=True)
        bond_mask = (counts > 1)
        edge_types = bond_mask.int()

        final_pair_i = all_edges[:, 0]
        final_pair_j = all_edges[:, 1]
        pos_i_new = positions[final_pair_i]
        pos_j_new = positions[final_pair_j]
        dr_new = wrap_displacement(pos_i_new - pos_j_new, box)  # Use wrap_displacement
        dr_scaled_new = dr_new / radius
        dist_new = dr_scaled_new.norm(dim=-1)

        if not add_self_edges:
            not_self = (final_pair_i != final_pair_j)
            final_pair_i = final_pair_i[not_self]
            final_pair_j = final_pair_j[not_self]
            dr_scaled_new = dr_scaled_new[not_self]
            dist_new = dist_new[not_self]
            edge_types = edge_types[not_self]

        final_distances = dist_new.unsqueeze(1)

        return (
            final_pair_i,
            final_pair_j,
            dr_scaled_new,
            final_distances,
            edge_types
        )
    else:
        final_distances = dist.unsqueeze(1)
        return (
            pair_i,
            pair_j,
            dr_scaled,
            final_distances,
            None
        )


#TODO:"torchCompactRadius" PBC COMPATIBLE IMPLEMENTATION (seems to require the nvcc toolkit...) 
def _compute_connectivity(self, node_features, n_particles_per_trajectory, radius, pbc=True):
        # handle batches. Default is 2 examples per batch

        # Specify examples id for particles/points
        batch_ids = torch.cat([torch.LongTensor([i for _ in range(n)]) for i, n in enumerate(n_particles_per_trajectory)]).to(self._device)
        # radius = radius + 0.00001 # radius_graph takes r < radius not r <= radius
        i, j = radiusSearch(node_features, 
                                  support=radius, 
                                  domainMin=self._boundaries.min().item(),
                                  domainMax=self._boundaries.max().item(),
                                  periodicity=pbc,
                                  algorithm='compact', 
                                  )
        return i, j
    
#TODO: PyG with PBC
def _compute_connectivity_pbc_pyg(self, most_recent_position, n_particles_per_trajectory, radius, add_self_edges=True):
         # handle batches. Default is 2 examples per batch
        # batch_ids = torch.cat([torch.LongTensor([i for _ in range(n)]) for i, n in enumerate(n_particles_per_trajectory)]).to(self._device)
        # radius = radius + 0.00001 # radius_graph takes r < radius not r <= radius
        combined_positions = pbc_duplication(most_recent_position)
        edge_index = radius_graph(combined_positions, r=radius, loop=add_self_edges) # (2, n_edges)
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
    
#TODO: Molecular Dynamics (simmilar to Fairchem) 
def radius_graph_pbc(data, radius, max_num_neighbors_threshold, topk_per_pair=None):
    """Computes pbc graph edges under pbc.
    topk_per_pair: (num_atom_pairs,), select topk edges per atom pair
    Note: topk should take into account self-self edge for (i, i)
    """
    atom_pos = data.pos
    num_atoms = data.natoms
    lattice = data.cell
    batch_size = len(num_atoms)
    device = atom_pos.device
    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = num_atoms
    num_atoms_per_image_sqr = (num_atoms_per_image ** 2).long()

    # index offset between images
    index_offset = (
        torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    )

    index_offset_expand = torch.repeat_interleave(
        index_offset, num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = (
        torch.arange(num_atom_pairs, device=device) - index_sqr_offset
    )

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        (atom_count_sqr // num_atoms_per_image_expand)
    ).long() + index_offset_expand
    index2 = (
        atom_count_sqr % num_atoms_per_image_expand
    ).long() + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)
    
    unit_cell = torch.tensor(OFFSET_LIST, device=device).float()
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(index2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(lattice, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)

    if topk_per_pair is not None:
        assert topk_per_pair.size(0) == num_atom_pairs
        atom_distance_sqr_sort_index = torch.argsort(atom_distance_sqr, dim=1)
        assert atom_distance_sqr_sort_index.size() == (num_atom_pairs, num_cells)
        atom_distance_sqr_sort_index = (
            atom_distance_sqr_sort_index +
            torch.arange(num_atom_pairs, device=device)[:, None] * num_cells).view(-1)
        topk_mask = (torch.arange(num_cells, device=device)[None, :] <
                     topk_per_pair[:, None])
        topk_mask = topk_mask.view(-1)
        topk_indices = atom_distance_sqr_sort_index.masked_select(topk_mask)

        topk_mask = torch.zeros(num_atom_pairs * num_cells, device=device)
        topk_mask.scatter_(0, topk_indices, 1.)
        topk_mask = topk_mask.bool()

    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    if topk_per_pair is not None:
        topk_mask = torch.masked_select(topk_mask, mask)

    num_neighbors = torch.zeros(len(atom_pos), device=device)
    num_neighbors.index_add_(0, index1, torch.ones(len(index1), device=device))
    num_neighbors = num_neighbors.long()
    max_num_neighbors = torch.max(num_neighbors).long()

    # Compute neighbors per image
    _max_neighbors = copy.deepcopy(num_neighbors)
    _max_neighbors[
        _max_neighbors > max_num_neighbors_threshold
    ] = max_num_neighbors_threshold
    _num_neighbors = torch.zeros(len(atom_pos) + 1, device=device).long()
    _natoms = torch.zeros(num_atoms.shape[0] + 1, device=device).long()
    _num_neighbors[1:] = torch.cumsum(_max_neighbors, dim=0)
    _natoms[1:] = torch.cumsum(num_atoms, dim=0)
    num_neighbors_image = (
        _num_neighbors[_natoms[1:]] - _num_neighbors[_natoms[:-1]]
    )

    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)
    # return torch.stack((index2, index1)), unit_cell, atom_distance_sqr.sqrt(), num_neighbors_image    
    
    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        return torch.stack((index2, index1)), unit_cell, atom_distance_sqr.sqrt(), num_neighbors_image
    # atom_distance_sqr.sqrt() distance

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with values greater than radius*radius so we can easily remove unused distances later.
    distance_sort = torch.zeros(
        len(atom_pos) * max_num_neighbors, device=device
    ).fill_(radius * radius + 1.0)

    # Create an index map to map distances from atom_distance_sqr to distance_sort
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index1 * max_num_neighbors
        + torch.arange(len(index1), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance_sqr)
    distance_sort = distance_sort.view(len(atom_pos), max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]

    # Offset index_sort so that it indexes into index1
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors_threshold
    )
    # Remove "unused pairs" with distances greater than the radius
    mask_within_radius = torch.le(distance_sort, radius * radius)
    index_sort = torch.masked_select(index_sort, mask_within_radius)

    # At this point index_sort contains the index into index1 of the closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index1), device=device).bool()
    mask_num_neighbors.index_fill_(0, index_sort, True)

    # Finally mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
    index1 = torch.masked_select(index1, mask_num_neighbors)
    index2 = torch.masked_select(index2, mask_num_neighbors)
    unit_cell = torch.masked_select(
        unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)

    if topk_per_pair is not None:
        topk_mask = torch.masked_select(topk_mask, mask_num_neighbors)

    edge_index = torch.stack((index2, index1))   
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask_num_neighbors)
    
    return edge_index, unit_cell, atom_distance_sqr.sqrt(), num_neighbors_image
    # atom_distance_sqr.sqrt() distance
    
    
#Compute Connecitivity under PBC
EPSILON = 0.2    
def compute_connectivity_pbc(positions, lattices, n_node, radius, 
                             bonds=None, add_self_edges=True):
  """
  construct graph using radius cut-off under periodic boundary conditions.
  simplified computation of distances/displacements under PBC is possible
  due to the lattices being cubes.
  Inputs:
    positions: N x dim
    lattices: B x dim
    n_node: B x 1
  Optional Inputs:
    bonds: E x 2. include these bonds even if they don't fall into the radius cut-off.
  Outputs:
    senders, receivers, displacements, distances, edge_type (1 if is_bond)
  """
  n_node_per_image_sqr = (n_node ** 2).long()

  # index offset between images
  index_offset = (torch.cumsum(n_node, dim=0) - n_node)
  index_offset_expand = torch.repeat_interleave(index_offset, n_node_per_image_sqr)
  n_node_per_image_expand = torch.repeat_interleave(n_node, n_node_per_image_sqr)

  # Compute a tensor containing sequences of numbers that range 
  # from 0 to n_node_per_image_sqr for each image
  # that is used to compute indices for the pairs of atoms.
  num_atom_pairs = torch.sum(n_node_per_image_sqr)
  index_sqr_offset = (torch.cumsum(n_node_per_image_sqr, dim=0) - n_node_per_image_sqr)
  index_sqr_offset = torch.repeat_interleave(index_sqr_offset, n_node_per_image_sqr)
  atom_count_sqr = (torch.arange(num_atom_pairs, device=positions.device) - index_sqr_offset)

  # Compute the indices for the pairs of atoms (using division and mod)
  index1 = ((atom_count_sqr.div(n_node_per_image_expand, rounding_mode='floor'))).long() + index_offset_expand
  index2 = (atom_count_sqr % n_node_per_image_expand).long() + index_offset_expand
  # Get the positions for each atom
  pos1 = torch.index_select(positions, 0, index1)
  pos2 = torch.index_select(positions, 0, index2)

  lattices = lattices.repeat_interleave(n_node_per_image_sqr, dim=0)
  displacement = pos1 - pos2  
  displacement = torch.where(displacement > (0.5 * lattices),
                             displacement - lattices, displacement)
  displacement = torch.where(displacement < (-0.5 * lattices),
                             displacement + lattices, displacement)
  displacement = displacement / radius
  distance = displacement.norm(dim=-1)

  mask = torch.le(distance, 1)
  if not add_self_edges:
    mask_not_same = torch.gt(distance, EPSILON)
    mask = torch.logical_and(mask, mask_not_same)
  
  if bonds is not None:
    all_edges = torch.cat([index1[:, None], index2[:, None]], dim=1)
    all_edges_non_unique = torch.cat([all_edges, bonds], dim=0)
    all_edges, counts = torch.unique(all_edges_non_unique, dim=0, return_counts=True)
    bond_mask = (counts > 1)  # this is correct because <all_edges> enumerates all edges.
    mask = torch.logical_or(mask, bond_mask)  # include both raidus-edges and bonds.
    # get edge type.
    all_edge_indices, counts = torch.unique(
        torch.cat([mask.nonzero(), bond_mask.nonzero()], dim=0), return_counts=True)
    edge_types = (counts > 1).int()
  else:
    edge_types = None
  # senders, receivers, displacements, distances, edge_type (is_bond)
  return (index1[mask], index2[mask], displacement[mask], 
          distance[mask].unsqueeze(1), edge_types)