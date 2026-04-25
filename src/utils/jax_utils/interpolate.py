import importlib.util
from typing import Sequence

import numpy as np
import jax.numpy as jnp
from numpy import array
from scipy.sparse.linalg import LinearOperator, cg
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator, griddata

from src.utils.jax_utils.jax_mls import mls


def pos_init_cartesian(box_size: array, n_per_dim: Sequence[int]):
    """Create particle coordinates at cell centers for a Cartesian grid."""
    box_size = np.asarray(box_size, dtype=float)
    n_per_dim = np.asarray(n_per_dim, dtype=int)
    if box_size.shape[0] != n_per_dim.shape[0]:
        raise ValueError("box_size and n_per_dim must have the same length")

    axes = [
        (np.arange(n_i, dtype=float) + 0.5) * (box_size[i] / n_i)
        for i, n_i in enumerate(n_per_dim)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([m.reshape(-1) for m in mesh], axis=1)
    return jnp.asarray(points)


class ParticleGridInterpolator:
    """Unified interpolation between particle sets and Cartesian grids in 2D/3D.

    Grid representation:
        - `u` has shape `(dim, *n_per_dim)`.

    Particle representation:
        - positions `r` have shape `(Np, dim)` where each coordinate is in `[0, L_i)`.
        - vector values `u_r` have shape `(Np, dim)`.

    Methods (preferred short names):
        - MLS: `p2g_mls`, `g2p_mls`
        - Spline (RBF + regular grid): `p2g_spline`, `g2p_spline`
        - SciPy standard interpolators: `p2g_scipy`, `g2p_scipy`
        - DFT (direct non-uniform Fourier sums): `p2g_dft`, `g2p_dft`
        - NUFFT (`finufft` when available, else DFT fallback): `p2g_nufft`, `g2p_nufft`

    Complexity notation:
        - `D`: spatial dimension (`2` or `3`)
        - `Np`: number of particles
        - `Ng`: number of grid points (`prod(n_per_dim)`)
        - `k`: average local neighbors (for MLS)
        - `I`: number of CG iterations (for inverse spectral methods)

    Complexity style:
        Method docstrings report asymptotic runtime and memory using the notation above.
        Constants and low-order terms are omitted.

    Key init args:
        - `n_per_dim`, `box_size`, `dim`: grid/domain definition
        - `mls_*`: control MLS kernel and regularization
        - `spline_*`: control RBF spline interpolation
        - `scipy_method`, `scipy_fill_value`: control standard SciPy interpolation behavior
        - `nufft_backend`: `"auto"`, `"finufft"`, or `"dft"`
        - `nufft_eps`: requested NUFFT tolerance for `finufft`
    """

    def __init__(
        self,
        n_per_dim,
        box_size,
        dim=3,
        mls_kernel="Quintic",
        mls_order=2,
        mls_h_factor=0.8,
        mls_regularization=1e-10,
        nufft_splits=64,
        spline_kernel="thin_plate_spline",
        spline_smoothing=0.0,
        spline_method="cubic",
        scipy_method="linear",
        scipy_fill_value=np.nan,
        nufft_backend="auto",
        nufft_eps=1e-10,
    ):
        if dim not in (2, 3):
            raise ValueError(f"Only 2D and 3D are supported, got dim={dim}")

        if np.isscalar(n_per_dim):
            n_per_dim = (int(n_per_dim),) * dim
        if np.isscalar(box_size):
            box_size = (float(box_size),) * dim

        self.dim = dim
        self.n_per_dim = tuple(int(n_i) for n_i in n_per_dim)
        self.box_size = np.asarray(box_size, dtype=float)

        if len(self.n_per_dim) != dim:
            raise ValueError("n_per_dim length must match dim")
        if self.box_size.shape != (dim,):
            raise ValueError("box_size must be a scalar or a sequence of length dim")
        if any(n_i <= 1 for n_i in self.n_per_dim):
            raise ValueError("Each grid dimension must be > 1")

        self.n_total = int(np.prod(self.n_per_dim))
        self.dx_vec = self.box_size / np.asarray(self.n_per_dim, dtype=float)
        self.dx = float(np.mean(self.dx_vec))

        self.mls_kernel = mls_kernel
        if mls_order not in (0, 1, 2):
            raise ValueError("mls_order must be 0, 1 or 2")
        self.mls_order = int(mls_order)
        self.mls_h_factor = mls_h_factor
        self.mls_regularization = mls_regularization

        self.nufft_splits = int(nufft_splits)
        if self.nufft_splits <= 0:
            raise ValueError("nufft_splits must be positive")

        self.spline_kernel = spline_kernel
        self.spline_smoothing = float(spline_smoothing)
        self.spline_method = spline_method

        self.scipy_method = scipy_method
        self.scipy_fill_value = scipy_fill_value

        if nufft_backend not in ("auto", "finufft", "dft"):
            raise ValueError("nufft_backend must be one of: 'auto', 'finufft', 'dft'")
        self.nufft_backend = nufft_backend
        self.nufft_eps = float(nufft_eps)
        self._finufft_available = importlib.util.find_spec("finufft") is not None
        if self.nufft_backend == "finufft" and not self._finufft_available:
            raise ImportError(
                "nufft_backend='finufft' requested but package 'finufft' is not installed"
            )

        self.grid_points = pos_init_cartesian(self.box_size, self.n_per_dim)
        self._grid_axes = [
            (np.arange(n_i, dtype=float) + 0.5) * (self.box_size[i] / n_i)
            for i, n_i in enumerate(self.n_per_dim)
        ]
        self._k_axes = [
            np.fft.fftshift(np.fft.fftfreq(n_i, d=1.0 / n_i)) for n_i in self.n_per_dim
        ]
        self._k_mesh = np.array(np.meshgrid(*self._k_axes, indexing="ij"), dtype=float)
        self._k_flat_np = self._k_mesh.reshape(self.dim, -1)
        self._k_flat = jnp.asarray(self._k_mesh.reshape(self.dim, -1))
        self._fft_axes = tuple(range(1, self.dim + 1))
        self._cell_center_offset = 0.5 * self.dx_vec
        self._periodic_shifts = (
            np.array(
                np.meshgrid(*[(-L, 0.0, L) for L in self.box_size], indexing="ij"),
                dtype=float,
            )
            .reshape(self.dim, -1)
            .T
        )

    def _wrap_positions_np(self, r):
        r = np.asarray(r, dtype=float)
        return np.mod(r, self.box_size)

    def _periodic_tile_scalar(self, points, values):
        tiled_points = np.concatenate([points + shift for shift in self._periodic_shifts], axis=0)
        tiled_values = np.tile(values, len(self._periodic_shifts))
        return tiled_points, tiled_values

    def _validate_particle_positions(self, r, name="r"):
        r = jnp.asarray(self._wrap_positions_np(r))
        if r.ndim != 2 or r.shape[1] != self.dim:
            raise ValueError(f"Expected {name} shape (Np, {self.dim}), got {r.shape}")
        return r

    def _validate_particle_field(self, r, u_r):
        r = self._validate_particle_positions(r, name="r")
        u_r = jnp.asarray(u_r)
        if u_r.ndim != 2 or u_r.shape != (r.shape[0], self.dim):
            raise ValueError(f"Expected u_r shape {(r.shape[0], self.dim)}, got {u_r.shape}")
        return r, u_r

    def _validate_grid_field(self, u):
        u = jnp.asarray(u)
        expected = (self.dim, *self.n_per_dim)
        if u.shape != expected:
            raise ValueError(f"Expected u shape {expected}, got {u.shape}")
        return u

    def p2g_mls(self, r_src, u_r):
        """Particles -> grid using MLS per vector component.

        Time complexity:
            `O(D * (Np log Np + Ng log Np + Ng * k))`

        Space complexity:
            `O(Np + Ng * k)`
        """
        r_src, u_r = self._validate_particle_field(r_src, u_r)
        components = [
            mls(
                r=r_src,
                r_target=self.grid_points,
                f=u_r[:, i],
                box_size=self.box_size,
                dx=self.dx,
                dim=self.dim,
                order=self.mls_order,
                kernel_name=self.mls_kernel,
                h_factor=self.mls_h_factor,
                regularization=self.mls_regularization,
            )
            for i in range(self.dim)
        ]
        u_grid = jnp.stack(components, axis=0).reshape((self.dim, *self.n_per_dim))
        return u_grid

    def g2p_mls(self, r_target, u):
        """Grid -> particles using MLS per vector component.

        Time complexity:
            `O(D * (Ng log Ng + Np log Ng + Np * k))`

        Space complexity:
            `O(Ng + Np * k)`
        """
        r_target = self._validate_particle_positions(r_target, name="r_target")
        u = self._validate_grid_field(u)

        u_flat = u.reshape(self.dim, -1)
        components = [
            mls(
                r=self.grid_points,
                r_target=r_target,
                f=u_flat[i],
                box_size=self.box_size,
                dx=self.dx,
                dim=self.dim,
                order=self.mls_order,
                kernel_name=self.mls_kernel,
                h_factor=self.mls_h_factor,
                regularization=self.mls_regularization,
            )
            for i in range(self.dim)
        ]
        return jnp.stack(components, axis=1)

    def g2p_spline(self, r_target, u):
        """Grid -> particles via `RegularGridInterpolator`.

        Time complexity:
            `O(D * Np)`

        Space complexity:
            `O(D * Np)` for outputs (plus interpolator internals).
        """
        r_target = self._wrap_positions_np(
            self._validate_particle_positions(r_target, name="r_target")
        )
        u = np.asarray(self._validate_grid_field(u), dtype=float)

        method = self.spline_method
        if method in ("cubic", "quintic") and any(n_i < 4 for n_i in self.n_per_dim):
            method = "linear"

        components = []
        for i in range(self.dim):
            interp = RegularGridInterpolator(
                self._grid_axes,
                u[i],
                method=method,
                bounds_error=False,
                fill_value=None,
            )
            components.append(interp(r_target))
        return jnp.asarray(np.stack(components, axis=1))

    def p2g_spline(self, r_src, u_r):
        """Particles -> grid via global RBF interpolation.

        Time complexity:
            `O(D * (Np^3 + Ng * Np))` (dense RBF solve + evaluation).

        Space complexity:
            `O(Np^2 + D * Ng)`
        """
        r_src, u_r = self._validate_particle_field(r_src, u_r)
        r_src = self._wrap_positions_np(r_src)
        u_r = np.asarray(u_r, dtype=float)

        components = []
        for i in range(self.dim):
            tiled_points, tiled_values = self._periodic_tile_scalar(r_src, u_r[:, i])
            interp = RBFInterpolator(
                tiled_points,
                tiled_values,
                kernel=self.spline_kernel,
                smoothing=self.spline_smoothing,
            )
            components.append(interp(np.asarray(self.grid_points)))

        return jnp.asarray(np.stack(components, axis=0).reshape((self.dim, *self.n_per_dim)))

    def g2p_scipy(self, r_target, u):
        """Grid -> particles using SciPy regular-grid interpolation.

        Time complexity:
            `O(D * Np)`; nearest-neighbor fallback adds another `O(D * Np)` pass.

        Space complexity:
            `O(D * Np)`
        """
        r_target = self._wrap_positions_np(
            self._validate_particle_positions(r_target, name="r_target")
        )
        u = np.asarray(self._validate_grid_field(u), dtype=float)

        method = self.scipy_method
        if method in ("cubic", "quintic") and any(n_i < 4 for n_i in self.n_per_dim):
            method = "linear"

        components = []
        for i in range(self.dim):
            interp = RegularGridInterpolator(
                self._grid_axes,
                u[i],
                method=method,
                bounds_error=False,
                fill_value=self.scipy_fill_value,
            )
            values = interp(r_target)
            if np.isnan(values).any() and method != "nearest":
                nearest = RegularGridInterpolator(
                    self._grid_axes,
                    u[i],
                    method="nearest",
                    bounds_error=False,
                    fill_value=None,
                )
                values = np.where(np.isnan(values), nearest(r_target), values)
            components.append(values)

        return jnp.asarray(np.stack(components, axis=1))

    def p2g_scipy(self, r_src, u_r):
        """Particles -> grid using SciPy `griddata`.

        Time complexity:
            Typical 2D/3D behavior is `O(D * (Np log Np + Ng log Np))`.
            Nearest fallback adds `O(D * Ng log Np)`.

        Space complexity:
            `O(Np + Ng)` (plus triangulation internals).
        """
        r_src, u_r = self._validate_particle_field(r_src, u_r)
        r_src = self._wrap_positions_np(r_src)
        u_r = np.asarray(u_r, dtype=float)

        method = (
            self.scipy_method if self.scipy_method in ("nearest", "linear", "cubic") else "linear"
        )
        xi = np.asarray(self.grid_points)
        components = []
        for i in range(self.dim):
            tiled_points, tiled_values = self._periodic_tile_scalar(r_src, u_r[:, i])
            vals = griddata(
                points=tiled_points, values=tiled_values, xi=xi, method=method, fill_value=np.nan
            )
            if np.isnan(vals).any() and method != "nearest":
                vals_nearest = griddata(
                    points=tiled_points, values=tiled_values, xi=xi, method="nearest"
                )
                vals = np.where(np.isnan(vals), vals_nearest, vals)
            components.append(vals)

        return jnp.asarray(np.stack(components, axis=0).reshape((self.dim, *self.n_per_dim)))

    def _phase_matrix(self, x_points):
        shifted_points = jnp.asarray(x_points) - jnp.asarray(self._cell_center_offset)
        theta = 2.0 * np.pi * shifted_points / jnp.asarray(self.box_size)
        return theta @ self._k_flat

    def _phase_matrix_np(self, x_points):
        shifted_points = np.asarray(x_points, dtype=np.float64) - np.asarray(
            self._cell_center_offset, dtype=np.float64
        )
        theta = 2.0 * np.pi * shifted_points / np.asarray(self.box_size, dtype=np.float64)
        return theta @ self._k_flat_np

    def _apply_fourier_series_dft(self, coeff, r_target):
        coeff_flat = np.asarray(coeff, dtype=np.complex128).reshape(-1)
        chunks = np.array_split(np.arange(r_target.shape[0]), self.nufft_splits)
        out = []
        for idx in chunks:
            if len(idx) == 0:
                continue
            phase = self._phase_matrix_np(r_target[idx])
            expo = np.exp(1j * phase)
            out.append(expo @ coeff_flat)
        return np.concatenate(out, axis=0)

    def _adjoint_spectral_sum_dft(self, r_src, values):
        values = np.asarray(values, dtype=np.complex128)
        chunks = np.array_split(np.arange(r_src.shape[0]), self.nufft_splits)
        coeff = np.zeros((self.dim, self.n_total), dtype=np.complex128)
        for idx in chunks:
            if len(idx) == 0:
                continue
            phase = self._phase_matrix_np(r_src[idx])
            expo = np.exp(-1j * phase)
            coeff += values[idx].T @ expo
        return coeff.reshape((self.dim, *self.n_per_dim))

    def _adjoint_scalar_spectral_sum_dft(self, r_src, values):
        values = np.asarray(values, dtype=np.complex128).reshape(-1)
        chunks = np.array_split(np.arange(r_src.shape[0]), self.nufft_splits)
        coeff = np.zeros((self.n_total,), dtype=np.complex128)
        for idx in chunks:
            if len(idx) == 0:
                continue
            phase = self._phase_matrix_np(r_src[idx])
            expo = np.exp(-1j * phase)
            coeff += values[idx] @ expo
        return coeff.reshape(self.n_per_dim)

    def g2p_dft(self, r_target, u):
        """Grid -> particles via direct non-uniform inverse DFT.

        Time complexity:
            `O(D * (Ng log Ng + Np * Ng))`

        Space complexity:
            `O(Ng + D * Np)`
        """
        r_target = self._validate_particle_positions(r_target, name="r_target")
        u = self._validate_grid_field(u)

        u_hat = np.fft.fftshift(
            np.fft.fftn(np.asarray(u, dtype=np.float64), axes=self._fft_axes), axes=self._fft_axes
        )
        values = np.zeros((r_target.shape[0], self.dim), dtype=np.float64)
        for i in range(self.dim):
            coeff = u_hat[i] / self.n_total
            values[:, i] = np.real(self._apply_fourier_series_dft(coeff, np.asarray(r_target)))
        return jnp.asarray(values)

    def p2g_dft(self, r_src, u_r):
        """Particles -> grid via direct least-squares inversion of non-uniform DFT.

        Time complexity:
            `O(D * I * Np * Ng + D * Ng log Ng)`

        Space complexity:
            `O(Ng + Np)`
        """
        r_src, u_r = self._validate_particle_field(r_src, u_r)
        r_np = np.asarray(r_src, dtype=np.float64)
        u_np = np.asarray(u_r, dtype=np.float64)
        n_modes = self.n_total
        regularization = max(self.mls_regularization, 1e-10)

        def matvec(coeff_flat):
            coeff = coeff_flat.reshape(self.n_per_dim)
            values = self._apply_fourier_series_dft(coeff, r_np)
            adjoint = self._adjoint_scalar_spectral_sum_dft(r_np, values)
            return adjoint.reshape(-1) + regularization * coeff_flat

        operator = LinearOperator(
            shape=(n_modes, n_modes),
            matvec=matvec,
            dtype=np.complex128,
        )

        rhs = self._adjoint_spectral_sum_dft(r_np, u_np)
        coeff = np.zeros((self.dim, *self.n_per_dim), dtype=np.complex128)
        for i in range(self.dim):
            solution, info = cg(
                operator, rhs[i].reshape(-1), rtol=1e-8, maxiter=min(8 * n_modes, 2000)
            )
            if info != 0:
                raise RuntimeError(
                    f"DFT least-squares solve failed for component {i} with info={info}"
                )
            coeff[i] = solution.reshape(self.n_per_dim)

        u_grid = np.zeros((self.dim, *self.n_per_dim), dtype=np.float64)
        for i in range(self.dim):
            coeff_unshift = np.fft.ifftshift(coeff[i])
            u_grid[i] = np.real(np.fft.ifftn(coeff_unshift) * self.n_total)
        return jnp.asarray(u_grid)

    def _should_use_finufft(self):
        if self.nufft_backend == "dft":
            return False
        if self.nufft_backend == "finufft":
            return True
        return self._finufft_available

    def _to_finufft_angles(self, r):
        shifted_points = np.asarray(r, dtype=np.float64) - np.asarray(
            self._cell_center_offset, dtype=np.float64
        )
        angles = 2.0 * np.pi * (shifted_points / np.asarray(self.box_size, dtype=np.float64))
        return np.ascontiguousarray(angles)

    def _adjoint_spectral_sum_finufft(self, r_src, values):
        coeff = np.zeros((self.dim, *self.n_per_dim), dtype=np.complex128)
        for i in range(self.dim):
            coeff[i] = self._finufft_type1(r_src, values[:, i])
        return coeff

    def _finufft_type2(self, fk, r_target):
        import finufft

        x = self._to_finufft_angles(r_target)
        fk = np.ascontiguousarray(np.asarray(fk, dtype=np.complex128))
        x0 = np.ascontiguousarray(x[:, 0])
        x1 = np.ascontiguousarray(x[:, 1])
        if self.dim == 2:
            return finufft.nufft2d2(x0, x1, fk, isign=1, eps=self.nufft_eps)
        x2 = np.ascontiguousarray(x[:, 2])
        return finufft.nufft3d2(x0, x1, x2, fk, isign=1, eps=self.nufft_eps)

    def _finufft_type1(self, r_src, c):
        import finufft

        x = self._to_finufft_angles(r_src)
        c = np.ascontiguousarray(np.asarray(c, dtype=np.complex128))
        x0 = np.ascontiguousarray(x[:, 0])
        x1 = np.ascontiguousarray(x[:, 1])
        if self.dim == 2:
            return finufft.nufft2d1(x0, x1, c, self.n_per_dim, isign=-1, eps=self.nufft_eps)
        x2 = np.ascontiguousarray(x[:, 2])
        return finufft.nufft3d1(
            x0,
            x1,
            x2,
            c,
            self.n_per_dim,
            isign=-1,
            eps=self.nufft_eps,
        )

    def g2p_nufft(self, r_target, u):
        """Grid -> particles via NUFFT (`finufft`) or DFT fallback.

        Time complexity:
            - with `finufft`: `O(D * (Ng log Ng + Np))`
            - fallback (`dft`): `O(D * (Ng log Ng + Np * Ng))`

        Space complexity:
            `O(Ng + D * Np)`
        """
        r_target = self._validate_particle_positions(r_target, name="r_target")
        u = self._validate_grid_field(u)

        if not self._should_use_finufft():
            return self.g2p_dft(r_target=r_target, u=u)

        u_np = np.asarray(u, dtype=np.float64)
        out = np.zeros((r_target.shape[0], self.dim), dtype=np.float64)
        for i in range(self.dim):
            fk = np.fft.fftshift(np.fft.fftn(u_np[i])) / self.n_total
            out[:, i] = np.real(self._finufft_type2(fk, np.asarray(r_target)))
        return jnp.asarray(out)

    def p2g_nufft(self, r_src, u_r):
        """Particles -> grid via NUFFT-based least-squares or DFT fallback.

        Time complexity:
            - with `finufft`: `O(D * I * (Ng log Ng + Np) + D * Ng log Ng)`
            - fallback (`dft`): `O(D * I * Np * Ng + D * Ng log Ng)`

        Space complexity:
            `O(Ng + Np)`
        """
        r_src, u_r = self._validate_particle_field(r_src, u_r)
        if not self._should_use_finufft():
            return self.p2g_dft(r_src=r_src, u_r=u_r)

        r_np = np.asarray(r_src, dtype=np.float64)
        u_np = np.asarray(u_r, dtype=np.float64)
        n_modes = self.n_total
        regularization = max(self.mls_regularization, 1e-10)

        def matvec(coeff_flat):
            coeff = coeff_flat.reshape(self.n_per_dim)
            values = self._finufft_type2(coeff, r_np)
            adjoint = self._finufft_type1(r_np, values)
            return (adjoint + regularization * coeff).reshape(-1)

        operator = LinearOperator(
            shape=(n_modes, n_modes),
            matvec=matvec,
            dtype=np.complex128,
        )

        rhs = self._adjoint_spectral_sum_finufft(r_np, u_np)
        coeff = np.zeros((self.dim, *self.n_per_dim), dtype=np.complex128)
        for i in range(self.dim):
            solution, info = cg(
                operator, rhs[i].reshape(-1), rtol=1e-8, maxiter=min(8 * n_modes, 2000)
            )
            if info != 0:
                raise RuntimeError(
                    f"NUFFT least-squares solve failed for component {i} with info={info}"
                )
            coeff[i] = solution.reshape(self.n_per_dim)

        u_grid = np.zeros((self.dim, *self.n_per_dim), dtype=np.float64)
        for i in range(self.dim):
            coeff_unshift = np.fft.ifftshift(coeff[i])
            u_grid[i] = np.real(np.fft.ifftn(coeff_unshift) * self.n_total)
        return jnp.asarray(u_grid)


if __name__ == "__main__":
    np.random.seed(0)
    import matplotlib.pyplot as plt
    import time

    def analytic_velocity(points, dim):
        """Example 2D/3D field."""
        out = np.zeros((points.shape[0], dim), dtype=float)
        out[:, 0] = np.sin(points[:, 0])
        # out[5, 0] = 1.0  # test boundary condition handling
        out[:, 1] = np.cos(points[:, 1])
        if dim == 3:
            out[:, 2] = np.sin(points[:, 2])
        return out

    def to_plot_field(u, dim):
        """Helper to handle 2D/3D."""
        if dim == 2:
            return np.asarray(u[0]).T
        z_mid = u.shape[-1] // 2
        return np.asarray(u[0, :, :, z_mid]).T

    for dim in [2, 3]:
        nx = 8
        n_per_dim = (nx,) * dim
        box = (2 * np.pi,) * dim

        interpolator = ParticleGridInterpolator(
            n_per_dim=n_per_dim,
            box_size=box,
            dim=dim,
            nufft_splits=16,
            spline_method="cubic",
            scipy_method="linear",
            nufft_backend="auto",
        )

        r_grid = np.asarray(interpolator.grid_points)
        r_eval = (
            r_grid + 0.15 * np.asarray(interpolator.dx_vec) * np.random.randn(*r_grid.shape)
        ) % np.asarray(box)
        u_ref_grid = analytic_velocity(r_grid, dim).T.reshape((dim, *n_per_dim))

        methods = {  # sorted by increasing accuracy in g2p mode
            "scipy": (
                lambda uu, rr: interpolator.g2p_scipy(rr, uu),
                lambda rr, uur: interpolator.p2g_scipy(rr, uur),
            ),
            "spline": (
                lambda uu, rr: interpolator.g2p_spline(rr, uu),
                lambda rr, uur: interpolator.p2g_spline(rr, uur),
            ),
            "mls": (
                lambda uu, rr: interpolator.g2p_mls(rr, uu),
                lambda rr, uur: interpolator.p2g_mls(rr, uur),
            ),
            "nufft": (
                lambda uu, rr: interpolator.g2p_nufft(rr, uu),
                lambda rr, uur: interpolator.p2g_nufft(rr, uur),
            ),
            "dft": (
                lambda uu, rr: interpolator.g2p_dft(rr, uu),
                lambda rr, uur: interpolator.p2g_dft(rr, uur),
            ),
        }

        print(f"[{dim}D] relative reconstruction errors and times (g2p/p2g)")
        for method_name, (to_particles, to_grid) in methods.items():
            t1 = time.time()
            u_eval = np.asarray(to_particles(u_ref_grid, r_eval))
            t2 = time.time()
            u_back = np.asarray(to_grid(r_eval, u_eval))
            t3 = time.time()

            err = float(np.linalg.norm(u_back - u_ref_grid) / (np.linalg.norm(u_ref_grid) + 1e-12))
            print(f"  {method_name.upper():6s}: {err:.3e} ({t2 - t1:.3f}s / {t3 - t2:.3f}s)")

            fig, axs = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
            axs[0].set_title("u_before (grid)")
            axs[0].imshow(
                to_plot_field(u_ref_grid, dim), origin="lower", extent=(0, box[0], 0, box[1])
            )
            axs[1].set_title("u_particles")
            axs[1].scatter(r_eval[:, 0], r_eval[:, 1], c=u_eval[:, 0], s=10)
            axs[1].set_xlim(0, box[0])
            axs[1].set_ylim(0, box[1])
            axs[2].set_title("u_after (grid)")
            axs[2].imshow(
                to_plot_field(u_back, dim), origin="lower", extent=(0, box[0], 0, box[1])
            )

            fig.savefig(f"interp_demo_{dim}d_{method_name}.png", dpi=140)
            plt.close(fig)

    # [2D] relative reconstruction errors and times (g2p/p2g)
    # SCIPY : 7.733e-02 (0.001s / 0.011s)
    # SPLINE: 4.668e-03 (0.004s / 0.017s)
    # MLS   : 6.179e-03 (1.601s / 0.133s)
    # NUFFT : 2.837e-08 (0.031s / 0.828s)
    # DFT   : 2.837e-08 (0.002s / 0.053s)
    # [3D] relative reconstruction errors and times (g2p/p2g)
    # SCIPY : 7.343e-02 (0.002s / 1.870s)
    # SPLINE: 5.602e-03 (0.011s / 117.047s)
    # MLS   : 4.706e-03 (1.825s / 0.235s)
    # NUFFT : 3.975e-08 (0.031s / 2.356s)
    # DFT   : 3.963e-08 (0.042s / 4.344s)
