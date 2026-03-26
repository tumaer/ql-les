"""Shared visualization helpers for trajectory comparisons."""

from __future__ import annotations

import numpy as np
import torch

from src.utils.interpolate import GridInterpolator
from src.utils.jax_utils.jax_spectral import energy_spectrum


BOX_SIZE = 2 * np.pi
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_INTERPOLATOR_CACHE: dict[tuple[int, int, float, str], GridInterpolator] = {}


class ZeroNeighborsInterpolationError(RuntimeError):
    """Raised when MLS2 interpolation has target points with zero neighbors."""


def stats(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Extract min/max/mean along axis=0."""
    return {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "med": np.mean(arr, axis=0),
    }


def infer_n_per_dim(n_particles: int, dim: int) -> int:
    """Infer grid resolution from number of particles for square/cubic domains."""
    return int(round(n_particles ** (1.0 / dim)))


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation for flattened arrays."""
    a_flat = np.asarray(a).reshape(-1)
    b_flat = np.asarray(b).reshape(-1)
    a_centered = a_flat - np.mean(a_flat)
    b_centered = b_flat - np.mean(b_flat)
    denom = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_centered, b_centered) / denom)


def get_mls2_interpolator(
    nx: int,
    dim: int,
    box_size: float = BOX_SIZE,
    device: torch.device | None = None,
) -> GridInterpolator:
    """Get cached MLS2 grid interpolator with quintic kernel."""
    if device is None:
        device = _DEVICE
    key = (nx, dim, float(box_size), str(device))
    if key not in _INTERPOLATOR_CACHE:
        dx = box_size / nx
        domain_size = (box_size,) * dim
        interp = GridInterpolator(
            is_periodic=True,
            domain_size=domain_size,
            dim=dim,
            dx=dx,
            condition="radius",
            cutoff_factor=2,
            kernel="quintic",
            mls_order=2,
        ).to(device)
        _INTERPOLATOR_CACHE[key] = interp
    return _INTERPOLATOR_CACHE[key]


def interpolate_velocity_to_grid_mls2(
    r: np.ndarray,
    u: np.ndarray,
    nx: int,
    dim: int,
    box_size: float = BOX_SIZE,
    device: torch.device | None = None,
) -> np.ndarray:
    """Interpolate particle velocity to grid using MLS2. Raises on zero-neighbor targets."""
    interpolator = get_mls2_interpolator(nx=nx, dim=dim, box_size=box_size, device=device)
    device_i = interpolator.grid.device

    r_src = torch.tensor(r[:, :dim], dtype=torch.float32, device=device_i)
    u_src = torch.tensor(u[:, :dim], dtype=torch.float32, device=device_i)

    try:
        with torch.no_grad():
            u_grid = interpolator(r=r_src, f=u_src)
    except AssertionError as err:
        if "zero weights" in str(err):
            raise ZeroNeighborsInterpolationError(str(err)) from err
        raise

    n_per_dim = [nx] * dim
    u_grid_np = u_grid.detach().cpu().numpy()
    out = np.zeros((dim, *n_per_dim), dtype=np.float32)
    for d in range(dim):
        out[d] = u_grid_np[:, d].reshape(n_per_dim)
    return out


def spectrum_from_grid(u_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute isotropic energy spectrum from a grid velocity field."""
    nx = u_grid.shape[1]
    spectrum_full = np.asarray(energy_spectrum(u_grid))
    k = np.arange(1, nx // 2 + 1)
    return k, spectrum_full[1 : nx // 2 + 1]
