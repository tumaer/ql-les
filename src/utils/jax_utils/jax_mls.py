import numpy as np
import jax.numpy as jnp
from jax import ops, vmap
from numpy import array
from scipy.spatial import KDTree
from . import jax_md_space as space
from .jax_sph_kernel import QuinticKernel, M4PrimeKernel

EPS = jnp.finfo(float).eps


def mls_2nd_order(
    r,
    r_target,
    f,
    box_size,
    dx,
    dim,
    kernel_name="M4Prime",
    h_factor=None,
    regularization=1e-10,
):
    """2nd-order moving least squares interpolation for periodic flows in a
    rectangular box.

    Based on, "Analysis of interpolation schemes for the accurate estimation of
    energy spectrum in Lagrangian methods", Shi et al., 2013

    Args:
        r (np.ndarray): coordinates of N particles of shape (N, dim)
        r_target (np.ndarray): coordinates of target particles of shape (N, dim)
        f (np.ndarray): scalar field, e.g. velocity of shape (N, 1)
        box_size (np.ndarray): Domain box, e.g. np.array([1., 2., 3.])
        dx (float): average particle spacing
        dim (int): dimension of the vector field

    Returns:
        (np.ndarray): interpolated values of f on r_target
    """

    if dim not in (2, 3):
        raise ValueError(f"Only 2D and 3D are supported, got dim={dim}")

    # define kernel function
    if kernel_name == "M4Prime":
        h_factor = 0.85 if h_factor is None else h_factor
        kernel_fn = M4PrimeKernel(h=h_factor * dx, dim=dim)
        distance_p = np.inf
    elif kernel_name == "Quintic":
        h_factor = 2 / 3 if h_factor is None else h_factor
        kernel_fn = QuinticKernel(h=h_factor * dx, dim=dim)
        distance_p = 2
    else:
        raise NotImplementedError(f"Kernel {kernel_name} not implemented.")

    r = np.asarray(r)
    r_target = np.asarray(r_target)
    f = np.asarray(f)
    box_size = np.asarray(box_size, dtype=float)

    if r.ndim != 2 or r_target.ndim != 2:
        raise ValueError("r and r_target must be 2D arrays of shape (N, dim)")
    if r.shape[1] != dim or r_target.shape[1] != dim:
        raise ValueError(f"Expected last dimension to be {dim}")
    if f.ndim != 1 or f.shape[0] != r.shape[0]:
        raise ValueError("f must be a 1D scalar field with shape (N,)")
    if box_size.shape != (dim,):
        raise ValueError("box_size must be a scalar or sequence of length dim")

    # displacement function for neighbors list
    displacement_fn, _ = space.periodic(side=box_size)

    # number of target particles
    n_target = jnp.shape(r_target)[0]

    # enforce periodic boundary conditions
    r_pbc, f_pbc = pbc_copy_scalar(r, f, box_size, kernel_fn.cutoff, dim)

    # compute edge list
    tree = KDTree(r_pbc)
    senders = tree.query_ball_point(r_target, kernel_fn.cutoff, p=distance_p)
    sender_sizes = [len(x) for x in senders]
    if not any(sender_sizes):
        return jnp.zeros((n_target,), dtype=jnp.asarray(f).dtype)

    i_s = np.repeat(range(n_target), sender_sizes)
    j_s = np.concatenate(senders, axis=0)

    # r_np = np.asarray(r_target)
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.scatter(r_np[:,0], r_np[:,1], c='blue', s=1)
    # ax.scatter(r_pbc[:,0], r_pbc[:,1], c='red', s=1)
    # ax.set_aspect('equal')
    # fig.savefig("tmp.png")
    # plt.close()

    # precompute quantities
    r_ji = vmap(displacement_fn)(r_pbc[j_s], r_target[i_s])
    if kernel_name == "M4Prime":
        w_dist = vmap(kernel_fn.w)(r_ji)
    elif kernel_name == "Quintic":
        rel_distances = np.linalg.norm(r_ji, axis=1, ord=2)
        w_dist = kernel_fn.w(rel_distances)
    else:
        raise NotImplementedError(f"Kernel {kernel_name} not implemented.")

    # Basis size for quadratic polynomial in dim dimensions:
    # 1 + dim + dim*(dim+1)/2
    mat_size = 1 + dim + (dim * (dim + 1)) // 2

    # calculate indices
    ind_d = jnp.diag_indices(dim)
    ind_u = jnp.triu_indices(dim, 1)

    # mls matrix entries
    def matrix(w_dist, r_ji):
        tensor = jnp.tensordot(r_ji, r_ji, axes=0)

        row = jnp.ones(mat_size)
        row = row.at[1 : dim + 1].mul(r_ji)
        row = row.at[dim + 1 : 2 * dim + 1].mul(tensor[ind_d] * 0.5)
        row = row.at[2 * dim + 1 :].mul(tensor[ind_u])

        column = jnp.ones(mat_size)
        column = column.at[1 : dim + 1].mul(r_ji)
        column = column.at[dim + 1 : 2 * dim + 1].mul(tensor[ind_d])
        column = column.at[2 * dim + 1 :].mul(tensor[ind_u] * 2)

        return jnp.tensordot(column, row, axes=0) * w_dist

    # calculate matrix
    temp = vmap(matrix)(w_dist, r_ji)
    mat = ops.segment_sum(temp, i_s, n_target)
    mat = mat + regularization * jnp.eye(mat_size)[None, :, :]

    # define solution vector entries
    def vector(w_dist, r_ji, f_j):
        tensor = jnp.tensordot(r_ji, r_ji, axes=0)
        vector = jnp.ones(mat_size) * w_dist * f_j
        vector = vector.at[1 : dim + 1].mul(r_ji)
        vector = vector.at[dim + 1 : 2 * dim + 1].mul(tensor[ind_d])
        vector = vector.at[2 * dim + 1 :].mul(tensor[ind_u] * 2)
        return vector

    # calculate vector
    temp = vmap(vector)(w_dist, r_ji, f_pbc[j_s])
    vec = ops.segment_sum(temp, i_s, n_target)

    # function for solving the system
    def solve_lin(matrix, vector):
        temp = jnp.linalg.solve(matrix, vector)
        return temp[0]

    # calculate interpolated value
    f_target = vmap(solve_lin)(mat, vec)

    return f_target


def pbc_copy_scalar(
    r: array, f: array, box_size: array, halo: float, dim: int, unsorted: bool = True
):
    """Copy particles of a scalar field for PBCs on a rectangular domain

    Args:
        r (np.ndarray): coordinates of N particles of shape (N, dim)
        f (np.ndarray): vector field, e.g. velocity of shape (N, dim)
        box_size (np.ndarray): Domain box of form np.array([x_size, y_size, z_size])
        halo (float): Width of halo region, e.g. 3h for Quintic spline
        dim (int): dimension of the data
        unsorted (bool): whether indices remain the same

    Returns:
        (np.ndarray, np.ndarray): new positions and properties after copying
    """
    if dim == 1:
        # get right side indices
        right = np.where(r <= halo, True, False)

        # get left side indices
        left = np.where(r >= box_size - halo, True, False)

        # concatenate pbc values
        r = np.concatenate((r, r[right] + box_size, r[left] - box_size))
        f = np.concatenate((f, f[right], f[left]))

        # rid of possible overlap
        ind = np.unique(r, axis=0, return_index=True)[1]
        ind = sorted(ind) if unsorted else ind
        r_pbc = r[ind]
        f_pbc = f[ind]

    elif dim == 2:
        # left and right side first
        # get indices
        right = np.where(r[:, 0] <= halo, True, False)
        left = np.where(r[:, 0] >= box_size[0] - halo, True, False)

        # concatenate pbc values
        dr = np.array([box_size[0], 0.0])[None, :]
        r = np.concatenate((r, r[right, :] + dr, r[left, :] - dr), axis=0)
        f = np.concatenate((f, f[right], f[left]))

        # now top and bottom
        # get indices
        top = np.where(r[:, 1] <= halo, True, False)
        bottom = np.where(r[:, 1] >= box_size[1] - halo, True, False)

        # concatenate pbc values
        dr = np.array([0.0, box_size[1]])[None, :]
        r = np.concatenate((r, r[top, :] + dr, r[bottom, :] - dr), axis=0)
        f = np.concatenate((f, f[top], f[bottom]))

        # rid of possible overlap
        ind = np.unique(r, axis=0, return_index=True)[1]
        ind = sorted(ind) if unsorted else ind
        r_pbc = r[ind, :]
        f_pbc = f[ind]

    elif dim == 3:
        # left and right side first
        # get indices
        right = np.where(r[:, 0] <= halo, True, False)
        left = np.where(r[:, 0] >= box_size[0] - halo, True, False)

        # concatenate pbc values
        dr = np.array([box_size[0], 0.0, 0.0])[None, :]
        r = np.concatenate((r, r[right, :] + dr, r[left, :] - dr), axis=0)
        f = np.concatenate((f, f[right], f[left]))

        # now top and bottom
        # get indices
        top = np.where(r[:, 1] <= halo, True, False)
        bottom = np.where(r[:, 1] >= box_size[1] - halo, True, False)

        # concatenate pbc values
        dr = np.array([0.0, box_size[1], 0.0])[None, :]
        r = np.concatenate((r, r[top, :] + dr, r[bottom, :] - dr), axis=0)
        f = np.concatenate((f, f[top], f[bottom]))

        # now front and back
        # get indices
        back = np.where(r[:, 2] <= halo, True, False)
        front = np.where(r[:, 2] >= box_size[2] - halo, True, False)

        # concatenate pbc values
        dr = np.array([0.0, 0.0, box_size[2]])[None, :]
        r = np.concatenate((r, r[back, :] + dr, r[front, :] - dr), axis=0)
        f = np.concatenate((f, f[back], f[front]))

        # rid of possible overlap
        ind = np.unique(r, axis=0, return_index=True)[1]
        ind = sorted(ind) if unsorted else ind
        r_pbc = r[ind, :]
        f_pbc = f[ind]

    return r_pbc, f_pbc
