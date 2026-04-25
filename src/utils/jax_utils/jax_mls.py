import itertools

import numpy as np
import jax.numpy as jnp
from jax import ops, vmap
from numpy import array
from scipy.spatial import KDTree

from . import jax_md_space as space
from .jax_sph_kernel import QuinticKernel, M4PrimeKernel


def mls(
    r,
    r_target,
    f,
    box_size,
    dx,
    dim,
    order=2,
    kernel_name="M4Prime",
    h_factor=None,
    regularization=1e-10,
):
    """Moving least squares interpolation for periodic flows (orders 0/1/2)."""
    if dim not in (2, 3):
        raise ValueError(f"Only 2D and 3D are supported, got dim={dim}")
    if order not in (0, 1, 2):
        raise ValueError(f"Only MLS orders 0, 1, 2 are supported, got order={order}")

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

    if kernel_name == "M4Prime":
        h = (0.85 if h_factor is None else h_factor) * dx
        kernel_fn = M4PrimeKernel(h=h, dim=dim)
        distance_p = np.inf
    elif kernel_name == "Quintic":
        h = (2 / 3 if h_factor is None else h_factor) * dx
        kernel_fn = QuinticKernel(h=h, dim=dim)
        distance_p = 2
    else:
        raise NotImplementedError(f"Kernel {kernel_name} not implemented.")

    displacement_fn, _ = space.periodic(side=box_size)
    n_target = int(r_target.shape[0])

    r_pbc, f_pbc = pbc_copy_scalar(r, f, box_size, kernel_fn.cutoff, dim)
    tree = KDTree(r_pbc)
    senders = tree.query_ball_point(r_target, kernel_fn.cutoff, p=distance_p)
    sender_sizes = [len(x) for x in senders]
    if not any(sender_sizes):
        return jnp.zeros((n_target,), dtype=jnp.asarray(f).dtype)

    i_s = np.repeat(range(n_target), sender_sizes)
    j_s = np.concatenate(senders, axis=0)

    r_ji = vmap(displacement_fn)(r_pbc[j_s], r_target[i_s])
    if kernel_name == "M4Prime":
        w_dist = vmap(kernel_fn.w)(r_ji)
    else:
        w_dist = kernel_fn.w(np.linalg.norm(r_ji, axis=1, ord=2))

    if order == 0:
        p_vals = jnp.ones((r_ji.shape[0], 1), dtype=r_ji.dtype)
    elif order == 1:
        p_vals = jnp.concatenate([jnp.ones((r_ji.shape[0], 1), dtype=r_ji.dtype), r_ji], axis=1)
    else:
        if dim == 2:
            quad = jnp.stack(
                [
                    r_ji[:, 0] ** 2,
                    r_ji[:, 1] ** 2,
                    2.0 * r_ji[:, 0] * r_ji[:, 1],
                ],
                axis=1,
            )
        else:
            quad = jnp.stack(
                [
                    r_ji[:, 0] ** 2,
                    r_ji[:, 1] ** 2,
                    r_ji[:, 2] ** 2,
                    2.0 * r_ji[:, 0] * r_ji[:, 1],
                    2.0 * r_ji[:, 0] * r_ji[:, 2],
                    2.0 * r_ji[:, 1] * r_ji[:, 2],
                ],
                axis=1,
            )
        p_vals = jnp.concatenate(
            [jnp.ones((r_ji.shape[0], 1), dtype=r_ji.dtype), r_ji, quad], axis=1
        )

    mat_size = p_vals.shape[1]
    mat_contrib = w_dist[:, None, None] * (p_vals[:, :, None] * p_vals[:, None, :])
    mat = ops.segment_sum(mat_contrib, i_s, n_target)
    mat = mat + regularization * jnp.eye(mat_size)[None, :, :]

    vec_contrib = w_dist[:, None] * p_vals * f_pbc[j_s][:, None]
    vec = ops.segment_sum(vec_contrib, i_s, n_target)

    coeff = jnp.linalg.solve(mat, vec[..., None]).squeeze(-1)
    return coeff[:, 0]


def pbc_copy_scalar(
    r: array, f: array, box_size: array, halo: float, dim: int, unsorted: bool = True
):
    """Copy particles of a scalar field for PBCs on a rectangular domain."""
    shifts = list(itertools.product([-1, 0, 1], repeat=dim))
    shifts.remove((0,) * dim)

    r_aug = [r]
    f_aug = [f]
    for shift in shifts:
        shift = np.asarray(shift, dtype=float)
        mask = np.ones(r.shape[0], dtype=bool)
        for axis, s in enumerate(shift):
            if s > 0:
                mask &= r[:, axis] <= halo
            elif s < 0:
                mask &= r[:, axis] >= box_size[axis] - halo
        if np.any(mask):
            dr = shift[None, :] * box_size[None, :]
            r_aug.append(r[mask] + dr)
            f_aug.append(f[mask])

    r_all = np.concatenate(r_aug, axis=0)
    f_all = np.concatenate(f_aug, axis=0)
    ind = np.unique(r_all, axis=0, return_index=True)[1]
    ind = sorted(ind) if unsorted else ind
    return r_all[ind], f_all[ind]
