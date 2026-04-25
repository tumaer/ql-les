"""Minimal tracer advection in a 2D Taylor-Green vortex on [0, 1]^2.

This script initializes 50^2 passive tracers at Cartesian cell centers and
advects them up to t=1.0 with periodic boundaries.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from src.utils.interpolate import QuinticKernel


def taylor_green_velocity(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Analytical 2D Taylor-Green velocity field on [0, 1]^2."""
    two_pi = 2.0 * np.pi
    u = -np.cos(two_pi * x) * np.sin(two_pi * y)
    v = np.sin(two_pi * x) * np.cos(two_pi * y)
    return u, v


def naive_neighbor_list(
    pos: np.ndarray, cutoff: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Naive O(N^2) neighbor list with periodic minimum-image convention."""
    dr = pos[:, None, :] - pos[None, :, :]
    dr -= np.round(dr)
    dist = np.linalg.norm(dr, axis=-1)
    mask = dist <= cutoff
    i_idx, j_idx = np.where(mask)
    return i_idx, j_idx, dist[i_idx, j_idx]


def sph_density_sum(pos: np.ndarray, dx: float) -> np.ndarray:
    """SPH density summation with rho0=1, m=dx^2, and quintic kernel h=dx."""
    kernel = QuinticKernel(h=dx, dim=2)
    i_idx, _j_idx, dist_ij = naive_neighbor_list(pos, cutoff=kernel.cutoff)
    w_ij = kernel.w(torch.from_numpy(dist_ij).float()).numpy()
    mass = dx**2
    rho = np.zeros(pos.shape[0], dtype=np.float64)
    np.add.at(rho, i_idx, mass * w_ij)
    return rho


def simulate_tracers(
    n: int = 50, t_end: float = 1.0, n_steps: int = 10_000
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advect n^2 tracers from cell centers to time t_end using forward Euler."""
    centers = (np.arange(n) + 0.5) / n
    x0, y0 = np.meshgrid(centers, centers, indexing="xy")
    x, y = x0.copy(), y0.copy()

    dt = t_end / n_steps

    for _ in range(n_steps):
        u, v = taylor_green_velocity(x, y)
        x = (x + dt * u) % 1.0
        y = (y + dt * v) % 1.0

    return x0, y0, x, y


if __name__ == "__main__":
    n = 50
    x0, y0, x1, y1 = simulate_tracers(n=n, t_end=1.0, n_steps=10_000)

    pos0 = np.column_stack([x0.ravel(), y0.ravel()])
    pos1 = np.column_stack([x1.ravel(), y1.ravel()])
    dx = 1.0 / n

    u0, v0 = taylor_green_velocity(pos0[:, 0], pos0[:, 1])
    u1, v1 = taylor_green_velocity(pos1[:, 0], pos1[:, 1])
    velmag0 = np.sqrt(u0**2 + v0**2)
    velmag1 = np.sqrt(u1**2 + v1**2)

    rho0 = sph_density_sum(pos0, dx=dx)
    rho1 = sph_density_sum(pos1, dx=dx)

    plt.rcParams.update({"font.size": 16, "axes.titlesize": 18, "axes.labelsize": 18})
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 7.6), layout="constrained")

    plots = [
        (axes[0, 0], pos0, velmag0, (0.0, 1.0)),
        (axes[0, 1], pos1, velmag1, (0.0, 1.0)),
        (axes[1, 0], pos0, rho0, (0.9, 1.1)),
        (axes[1, 1], pos1, rho1, (0.9, 1.1)),
    ]
    scatters = []
    for ax, pos, values, (vmin, vmax) in plots:
        s = ax.scatter(pos[:, 0], pos[:, 1], c=values, s=12, cmap="viridis", vmin=vmin, vmax=vmax)
        scatters.append(s)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].set_title("t=0")
    axes[0, 1].set_title("t=1")
    axes[0, 0].set_ylabel("Velocity magnitude")
    axes[1, 0].set_ylabel("Density")

    fig.subplots_adjust(left=0.10, right=0.88, top=0.90, bottom=0.08, wspace=0.15, hspace=0.15)
    fig.colorbar(scatters[1], ax=[axes[0, 0], axes[0, 1]], fraction=0.04, pad=0.02)
    fig.colorbar(scatters[3], ax=[axes[1, 0], axes[1, 1]], fraction=0.04, pad=0.02)

    fig.savefig("tgv_tracer.png", dpi=300)
