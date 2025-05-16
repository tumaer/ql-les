import warnings
import copy
import torch
import numpy as np
from torch_geometric.nn import radius_graph, radius, knn, knn_graph


def gen_grid_points(n: list, box_size: list) -> np.ndarray:
    """Generate cartesian grid points.

    Args:
        n (list): List of shape (D,) with number of points in each dimension.
        box_size (list): Size of the box in each dimension. Starting at 0.

    Returns:
        Cartesian grid points of shape (**n, D).
    """
    dim = len(n)
    dx = [box_size[i] / n[i] for i in range(dim)]
    import numpy as np

    tx = np.linspace(0, box_size[0], n[0], endpoint=False) + dx[0] / 2
    ty = np.linspace(0, box_size[1], n[1], endpoint=False) + dx[1] / 2
    if dim == 2:
        grid = np.meshgrid(tx, ty, indexing="ij")
    elif dim == 3:
        tz = np.linspace(0, box_size[2], n[2], endpoint=False) + dx[2] / 2
        grid = np.meshgrid(tx, ty, tz, indexing="ij")

    grid = np.stack(grid, axis=-1).astype(np.float32)
    return grid


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
    # floating point precision messes up the results
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
    """Shift position by a displacement vector."""
    if pbc:
        return wrap_position(r + dr, box)
    else:
        return r + dr


def displ_fn(r1, r2, box=None, pbc=False):
    """Compute the displacement vector between two positions."""
    if pbc:
        return wrap_displacement(r1 - r2, box)
    else:
        return r1 - r2


def _nearest_batch(
    x,
    n_particles_per_trajectory,
    condition="radius",
    cutoff=None,
    k=None,
    add_self_edges=True,
    query=None,
    n_pptr_query=None,
):
    """Compute the nearest neighbors for a batch of particles without PBC."""
    device = x.device
    batch_ids = torch.cat(
        [
            torch.full((n,), i, dtype=torch.long, device=device)
            for i, n in enumerate(n_particles_per_trajectory)
        ]
    )
    if query is not None:
        batch_y = torch.cat(
            [
                torch.full((n,), i, dtype=torch.long, device=device)
                for i, n in enumerate(n_pptr_query)
            ]
        )
        if condition == "radius":
            edge_index = radius(
                x=x, y=query, r=cutoff, batch_x=batch_ids, batch_y=batch_y, max_num_neighbors=300
            )
        elif condition == "knn":
            edge_index = knn(x=x, y=query, k=k, batch_x=batch_ids, batch_y=batch_y)
    else:
        # this implementation is eventually faster than setting query=x
        if condition == "radius":
            edge_index = radius_graph(
                x, r=cutoff, batch=batch_ids, loop=add_self_edges, max_num_neighbors=300
            )
        elif condition == "knn":
            edge_index = knn_graph(x, k=k, batch=batch_ids, loop=add_self_edges)
        # both `_graph` functions flip the order of senders/receivers. This matters in knn
        edge_index = torch.flip(edge_index, dims=[0])
    return edge_index


def _nearest_batch_pbc(
    x,
    n_particles_per_trajectory,
    box,
    condition="radius",
    cutoff=None,
    k=None,
    add_self_edges=True,
    query=None,
    n_pptr_query=None,
):
    """Compute the nearest neighbors for a batch of particles with PBC."""
    if not add_self_edges:
        raise NotImplementedError("Self edges are always considered for now")

    device = x.device
    combined_positions, num_copies = pbc_duplication(x, box, n_particles_per_trajectory)

    # Assign batch IDs for original and duplicated points
    batch_x = torch.cat(
        [
            torch.full((n * num_copies,), i, dtype=torch.long, device=device)
            for i, n in enumerate(n_particles_per_trajectory)
        ]
    )

    if query is not None:
        batch_y = torch.cat(
            [
                torch.full((n,), i, dtype=torch.long, device=device)
                for i, n in enumerate(n_pptr_query)
            ]
        )
    else:
        batch_y = torch.cat(
            [
                torch.full((n,), i, dtype=torch.long, device=device)
                for i, n in enumerate(n_particles_per_trajectory)
            ]
        )
        query = x
        n_pptr_query = n_particles_per_trajectory

    # Compute connectivity graph from duplicated (x) to original (y)
    if condition == "radius":
        edge_index = radius(
            x=combined_positions,
            y=query,
            r=cutoff,
            batch_x=batch_x,
            batch_y=batch_y,
            max_num_neighbors=300,
        )
    elif condition == "knn":
        edge_index = knn(
            x=combined_positions, y=query, k=k, batch_x=batch_x, batch_y=batch_y
        )  # first list is sorted
        if (n_particles_per_trajectory <= k).any():
            warnings.warn("If number of points < k, there will be duplicate edges.")

    # Correct edge_index[1] to wrap indices *within each batch*
    cum_counts = torch.cat([torch.tensor([0], device=device), torch.cumsum(n_pptr_query, dim=0)])
    for i in range(len(n_particles_per_trajectory)):
        start, end = cum_counts[i], cum_counts[i + 1]
        mask = (edge_index[0] >= start) & (edge_index[0] < end)  # edge_index[0] is sorted
        local_idx = edge_index[1][mask] - start
        edge_index[1][mask] = (local_idx % n_particles_per_trajectory[i]) + start

    #Message passing layers expect source_to_target
    edge_index = torch.flip(edge_index, dims=[0])

    return edge_index


def pbc_duplication(x, box, n_particles_per_trajectory):
    """Apply periodic boundary condition duplication.

    Args:
        x (torch.Tensor): Tensor of shape (N, D) containing concatenated
            positions for all batch elements.
        box (torch.Tensor): Tensor of shape (D,) representing the box size for
            periodic shifts.
        n_particles_per_trajectory (torch.Tensor): Tensor of shape (B,) containing
            the number of particles per batch item.

    Returns:
        torch.Tensor: Combined positions x of shape (N * num_copies, D).
        int: Number of copies created for periodic duplication.
    """
    device = x.device
    ndim = x.size(1)
    # Shifts in -1, 0, +1 for each axis
    shifts = torch.stack(
        torch.meshgrid(
            *[torch.tensor([-1, 0, 1], device=device) for _ in range(ndim)], indexing="ij"
        ),
        dim=-1,
    ).reshape(-1, ndim)  # (num_copies, D)
    num_copies = shifts.size(0)

    box = box.to(dtype=x.dtype, device=x.device)

    combined = []
    start = 0
    for n in n_particles_per_trajectory:
        # shift each trajectory independently
        combined.extend([x[start : start + n] + shift * box for shift in shifts])
        start += n
    combined_positions = torch.cat(combined, dim=0)
    return combined_positions, num_copies


def nearest(x, n_particles_per_trajectory, pbc=False, box=None, **kwargs):
    """Wrapper function for _nearest_batch or _nearest_batch_pbc.

    Args:
        x (torch.Tensor): Tensor of shape (N, D) containing
            concatenated positions for all batch elements.
        n_particles_per_trajectory (torch.Tensor): Tensor of shape (B,) containing
            the number of particles per batch item.
        pbc (bool, optional): Whether to use periodic boundary conditions.
        box (torch.Tensor, optional): Box size for periodic boundary conditions.
        **kwargs: Additional arguments for the _nearest_batch or _nearest_batch_pbc functions.

    kwargs:
        condition (str, optional): Condition for neighbor search, either "radius" or "knn".
            If "radius", the cutoff is used as the radius. If "knn", the cutoff is used as K.
        cutoff (float, optional): Cutoff distance for radius graph.
        k (int, optional): K in K-NN.
        add_self_edges (bool, optional): Whether to include self edges in the graph. Not used with query.
        query (torch.Tensor, optional): Tensor of shape (M, D) containing query points.
            If None, the search is performed on the x tensor itself.
        n_pptr_query (torch.tensor, optional): Tensor of shape (B,) containing the number
            of particle per batch of the query items.

    Returns:
        torch.Tensor: Edge indices of shape (2, E) with [sorted receivers, senders].
    """
    assert x.ndim == 2, "positions x must be a matrix"
    assert x.shape[1] in [2, 3], "positions x must be 2D or 3D"
    assert n_particles_per_trajectory.ndim == 1, "n_particles_per_trajectory must be a vector"
    if "query" in kwargs and kwargs["query"] is not None:
        assert "n_pptr_query" in kwargs, "n_pptr_query must be specified if query is not None"

    condition = kwargs.get("condition", "radius")  # by default, use radius
    assert condition in ["radius", "knn"], "condition must be either 'radius' or 'knn'"
    if condition == "knn":
        assert kwargs["k"] is not None, "k must be provided for knn"
    elif condition == "radius":
        assert kwargs["cutoff"] is not None, "cutoff must be provided for radius"

    if pbc:
        assert box is not None, "box_size must be provided for periodic boundary conditions"
        return _nearest_batch_pbc(x, n_particles_per_trajectory, box=box, **kwargs)
    else:
        return _nearest_batch(x, n_particles_per_trajectory, **kwargs)


if __name__ == "__main__":
    # Parameters
    Nx, L, dim = 4, 2 * np.pi, 2
    condition = "radius"  # "radius" or "knn"
    use_query = True  # False or True

    # Derived parameters
    dx = L / Nx
    box_size = np.ones(dim) * L
    k, cutoff = None, None
    if condition == "knn":
        k = 5  # 3 if dim == 2 else 4
        circle_radius = k ** (1 / dim)
    elif condition == "radius":
        cutoff = 1.25 * dx
        circle_radius = cutoff
    kwargs = {"box": box_size, "condition": condition, "cutoff": cutoff, "k": k}

    # Point cloud
    import random
    import matplotlib.pyplot as plt

    random.seed(0)
    np.random.seed(0)
    noise = np.random.normal(0, dx / 5, (Nx**dim, dim))
    pos_demo = shift_fn(pos_init_cartesian_2d(box_size, dx), noise - dx / 2, box_size, pbc=True)
    pos_demo = torch.tensor(pos_demo, dtype=torch.float32)
    n_part_per_traj = torch.tensor([pos_demo.shape[0]], dtype=torch.int64)
    if use_query:
        query = torch.tensor(pos_init_cartesian_2d(box_size, dx), dtype=torch.float32)
        n_pptr_query = torch.tensor([len(query)])
    else:
        query = n_pptr_query = None

    def plt_nbrs(ax, i, pos_demo, query, nbrs):
        """Helper function."""
        if use_query:
            ax.scatter(query[:, 0], query[:, 1], marker="x", alpha=0.5)
        # for each neighbor of particle i, draw a line to its neighbors
        pos_i = pos_demo[i] if query is None else query[i]
        neighbors = nbrs[1][nbrs[0] == i]
        for j in neighbors:
            ax.arrow(
                x=pos_i[0],
                y=pos_i[1],
                dx=pos_demo[j, 0] - pos_i[0],
                dy=pos_demo[j, 1] - pos_i[1],
                head_width=0.1,
                head_length=0.2,
                fc="red",
                ec="red",
                alpha=0.5,
                length_includes_head=True,
            )
        # draw a circle around particle i with cutoff radius
        for s in [[0, 0], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1]]:
            center = (pos_i[0] + s[0] * box_size[0], pos_i[1] + s[1] * box_size[1])
            circle = plt.Circle(center, circle_radius, color="r", fill=False)
            ax.add_artist(circle)
            if not pbc:
                break
        ax.set_xlim(0, L)
        ax.set_ylim(0, L)
        ax.set_aspect("equal", adjustable="box")
        title_str = f"cutoff={cutoff:.3f}" if condition == "radius" else f"k={k}"
        ax.set_title(f"Particle {i} has {len(neighbors)} neighbors with {title_str}")

    ### Test neighbor search without batching
    for pbc in [False, True]:
        nbrs = nearest(
            pos_demo, n_part_per_traj, pbc=pbc, query=query, n_pptr_query=n_pptr_query, **kwargs
        )

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(pos_demo[:, 0], pos_demo[:, 1])
        plt_nbrs(ax, 0, pos_demo, query, nbrs)
        plt.tight_layout()
        pbc_lbl = "pbc" if pbc else "nopbc"
        query_lbl = "_query" if query is not None else ""
        fig.savefig(f"nbrs{query_lbl}_{pbc_lbl}_{condition}.png")

    ### Test neighbor search with batching
    pos_demo_1 = copy.deepcopy(pos_demo)
    pos_demo_2 = torch.tensor(
        pos_init_cartesian_2d(box_size / 2, dx) - dx / 4, dtype=torch.float32
    )
    pos_s = [pos_demo_1, pos_demo_2]
    n_part_per_traj = torch.tensor([pos_demo_1.shape[0], pos_demo_2.shape[0]], dtype=torch.int64)
    pos_demo = torch.cat([pos_demo, pos_demo_2], dim=0)
    if use_query:
        query = query.repeat(2, 1)
        n_pptr_query = n_pptr_query.repeat(2)

    # Same as above, but two scatter plots for each point cloud in the batch
    for pbc in [False, True]:
        nbrs = nearest(
            pos_demo, n_part_per_traj, pbc=pbc, query=query, n_pptr_query=n_pptr_query, **kwargs
        )

        fig, axs = plt.subplots(1, 2, figsize=(10, 5.5))
        for ind, i in enumerate([0, len(pos_demo_1)]):  # first particle in each batch
            ax = axs[ind]
            ax.scatter(pos_s[ind][:, 0], pos_s[ind][:, 1])
            plt_nbrs(ax, i, pos_demo, query, nbrs)
        plt.tight_layout()
        pbc_lbl = "pbc" if pbc else "nopbc"
        query_lbl = "_query" if query is not None else ""
        fig.savefig(f"nbrs{query_lbl}_{pbc_lbl}_{condition}_batch.png")
