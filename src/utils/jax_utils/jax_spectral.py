import numpy as np
import jax.numpy as jnp
from jax import Array

EPS = jnp.finfo(float).eps


def get_real_wavenumber_grid(n, dim):
    """Get the real wavenumber grid for a given dimension and max wavenumber."""
    Nf = n // 2 + 1
    k = np.fft.fftfreq(n, 1.0 / n)  # for other dimensions
    kx = k[:Nf].copy()
    kx[-1] *= -1
    if dim == 2:
        k_field = np.array(np.meshgrid(kx, k, indexing="ij"), dtype=int)
    elif dim == 3:
        k_field = np.array(np.meshgrid(kx, k, k, indexing="ij"), dtype=int)
    return k_field, k


def energy_spectrum(vel: Array, mul_fac: float = 1.0, is_scalar_field: bool = False):
    """JAX implemented energy spectrum computation on a grid.

    Code based on JAX-FLUIDS implementation."""

    dim = vel.shape[0]
    ns = vel.shape[1:]

    # check for square box with equal side length
    assert jnp.array_equal(ns, jnp.ones(dim) * ns[0])

    # common resolution
    n = ns[0]

    # Fourier transform
    if dim == 1:
        # TODO: check whether 1D is working
        vel_hat = jnp.fft.rfftn(vel)
    elif dim == 2:
        vel_hat = jnp.fft.rfftn(vel, axes=(2, 1))
    elif dim == 3:
        vel_hat = jnp.fft.rfftn(vel, axes=(3, 2, 1))

    # initialize wavenumber grid
    k_field, k = get_real_wavenumber_grid(n, dim)

    # compute prefactor
    fact = (
        2 * (k_field[0] > 0) * (k_field[0] < n // 2)
        + 1 * (k_field[0] == 0)
        + 1 * (k_field[0] == n // 2)
    )

    # calculate wavenumber vector norms
    k_field_norm = jnp.linalg.norm(k_field, axis=0, ord=2)

    # calculate integration shell
    shell = (k_field_norm + 0.5).astype(int).flatten()

    # fourier transform prefactor
    vel_hat /= n**dim

    # calculate energy
    abs_energy = jnp.sum(jnp.abs(vel_hat**2), axis=0)
    abs_energy *= fact * mul_fac

    # number of samples
    n_samples = jnp.zeros(n)
    n_samples = n_samples.at[shell].add(fact.flatten())

    # compute energy spectrum
    ek = jnp.zeros(n)
    ek = ek.at[shell].add(abs_energy.flatten())
    ek *= 4 * jnp.pi * k**2 / (n_samples + EPS)

    return ek


def spectral_filtering(u, ckp_N):
    """Spectral filtering for LES reference data.

    Args:
        u (jnp.ndarray): Flow field with shape (3, N, N, N) or (2, N, N).
        ckp_N (int): How many spatial modes to keep (after spectral filtering).

    Returns:
        jnp.ndarray: Flow field of shape (3, ckp_N, ckp_N, ckp_N) or (2, ckp_N, ckp_N).
    """

    spatial_dim = u.ndim - 1
    if spatial_dim not in (2, 3):
        raise ValueError(f"Only 2D and 3D are supported, got spatial_dim={spatial_dim}")

    N_u = u.shape[1]
    if any(s != N_u for s in u.shape[1:]):
        raise ValueError("Only isotropic square/cubic grids are supported")
    if ckp_N > N_u:
        raise ValueError("ckp_N cannot be larger than the input resolution")

    fft_axes = tuple(range(1, u.ndim))

    # FFT
    y_fft = jnp.fft.fftn(u, axes=fft_axes)  # (3, N, N, N)  TODO: this was (3,2,1)

    # Cut high frequencies. (3, N, N, N) -> (3, ckp_N, ckp_N, ckp_N)
    sl = slice(N_u // 2 - ckp_N // 2, N_u // 2 + ckp_N // 2)
    slices = tuple([slice(None)] + [sl] * spatial_dim)
    y_fft_sub = jnp.fft.fftshift(y_fft, axes=fft_axes)[slices]
    y_fft_sub = jnp.fft.ifftshift(y_fft_sub, axes=fft_axes)

    # IFFT
    # TODO: this was (3,2,1)
    y_ifft = jnp.fft.ifftn(y_fft_sub, axes=fft_axes) / (N_u / ckp_N) ** spatial_dim
    y_ifft = y_ifft.real  # (3, ckp_N, ckp_N, ckp_N)

    # # Orientation changes during spectral filtering from DNS to LES grid
    # import matplotlib.pyplot as plt
    # _, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].imshow(u[0])
    # axs[0].set_title(f"[{u[0].min():.2f}, {u[0].max():.2f}]")
    # axs[1].imshow(y_ifft[0])
    # axs[1].set_title(f"[{y_ifft[0].min():.2f}, {y_ifft[0].max():.2f}]")
    # plt.savefig("orientation_check.png")

    return y_ifft
