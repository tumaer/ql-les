"""Utilities for SPH particle relaxation. See Neural SPH (Toshev et al., 2024)."""

import jax.numpy as jnp
import numpy as np
import torch
from torch_scatter import scatter_add
from torch_geometric.nn import radius_graph

from src.utils.nbrs_utils import (
    pos_init_cartesian_2d, shift_fn, displ_fn, radius_graph_pbc
)


EPS = torch.finfo(torch.float32).eps


class TaitEoS:
    """Tait equation of state.

    From: "A generalized wall boundary condition for smoothed particle
    hydrodynamics", Adami et al 2012
    """

    def __init__(self, p_ref, rho_ref, p_background, gamma):
        self.p_ref = p_ref
        self.rho_ref = rho_ref
        self.p_bg = p_background
        self.gamma = gamma

    def p_fn(self, rho):
        return self.p_ref * ((rho / self.rho_ref) ** self.gamma - 1) + self.p_bg

    def rho_fn(self, p):
        p_temp = p + self.p_ref - self.p_bg
        return self.rho_ref * (p_temp / self.p_ref) ** (1 / self.gamma)


class QuinticKernel:
    """The quintic kernel function of Morris."""

    def __init__(self, h, dim=3):
        self._one_over_h = 1.0 / h

        self._normalized_cutoff = 3.0
        self.cutoff = self._normalized_cutoff * h
        if dim == 1:
            self._sigma = 1.0 / 120.0 * self._one_over_h
        elif dim == 2:
            self._sigma = 7.0 / 478.0 / np.pi * self._one_over_h**2
        elif dim == 3:
            self._sigma = 3.0 / 359.0 / np.pi * self._one_over_h**3

    def w(self, r):
        q = r * self._one_over_h
        q1 = torch.clamp(1.0 - q, min=0.0)
        q2 = torch.clamp(2.0 - q, min=0.0)
        q3 = torch.clamp(3.0 - q, min=0.0)

        return self._sigma * (q3**5 - 6.0 * q2**5 + 15.0 * q1**5)

    def grad_w(self, r):
        """Evaluates the 1D kernel gradient at the radial distance r."""
        q = r * self._one_over_h
        q1 = torch.clamp(1.0 - q, min=0.0)
        q2 = torch.clamp(2.0 - q, min=0.0)
        q3 = torch.clamp(3.0 - q, min=0.0)

        grad_w_q = self._sigma * (-5.0 * (q3**4 - 6.0 * q2**4 + 15.0 * q1**4))
        grad_w_r = grad_w_q * self._one_over_h

        return grad_w_r


def relax_wrapper(
    Nx, dim=2, L=2 * np.pi, is_physical=False, u_ref=None, is_tvf=True, nu=0.0, box=None
):
    """Compute pressure gradient term from NSE.

    Args:
        Nx (int): Number of particles per dimension.
        dim (int): Dimension.
        L (float): Box size.
        is_physical (bool): Whether to use physical units internally. Doesn't affect
            the output.
        u_ref (float): Reference velocity for equation of state.

    Returns:
        Callable which takes coordinates `r` and returns density `rho`.
    """

    N_tot = Nx**dim
    dx_phys = L / Nx
    l_ref = 1.0 if is_physical else dx_phys
    dx = dx_phys / l_ref  # non-dimensionalize

    rho_ref = 1.0  # reference density
    mass = dx**dim * rho_ref

    box = box if box is not None else np.ones(dim) * L / l_ref 
    pbc = True
    
    kernel_fn = QuinticKernel(h=dx, dim=dim)

    u_eos = u_ref / l_ref
    c_eos = 10 * u_eos  # speed of sound
    p_eos = c_eos**2 * rho_ref  # reference pressure
    eos = TaitEoS(p_ref=p_eos, rho_ref=rho_ref, p_background=0.0, gamma=1.0)

    def normalize_length(r):
        return r / l_ref

    def denormalize_length(r):
        return r * l_ref

    def loop_body(r, n_part_per_traj, u=None, verbose=False):
        if nu != 0.0:
            assert (u is not None) and (r.shape==u.shape), "If nu!=0, u needed."

        r = normalize_length(r)

        if pbc:
            edge_index = radius_graph_pbc(r, n_part_per_traj, kernel_fn.cutoff, box)
        else:
            edge_index = radius_graph(r, r=kernel_fn.cutoff, loop=False)
        i_s, j_s = edge_index
        # print(i_s[j_s==0].sort()[0])
        # print(sum(i_s!=len(r)))
        r_i, r_j = r[i_s], r[j_s]
        dr_ij = displ_fn(r_i, r_j, box, pbc)
        dist = torch.norm(dr_ij, dim=-1)
        w_dist = kernel_fn.w(dist)

        rho = mass * scatter_add(w_dist, i_s, dim=0, dim_size=N_tot)
        p = eos.p_fn(rho)
        if verbose:
            print(f"Density min/max/std: {rho.min().item():.4f}, {rho.max().item():.4f}, {rho.std().item():.4f}")
        
        def acceleration_fn(r_ij, d_ij, rho_i, rho_j, p_i, p_j, u_i=None, u_j=None):
            # Compute unit vector, above eq. (6), Zhang (2017). Sign flipped here.
            e_ij = r_ij / (d_ij[:, None] + EPS)

            # Compute kernel gradient
            kernel_der = kernel_fn.grad_w(d_ij)
            kernel_grad = kernel_der[:,None] * e_ij

            # Compute density-weighted pressure (weighted arithmetic mean)
            p_ij = (rho_j * p_i + rho_i * p_j) / (rho_i + rho_j)

            # Eq. (8), Adami (2012) with constant `mass`
            prefactor = mass * ((1 / rho_i) ** 2 + (1 / rho_j) ** 2)
            acc = (-prefactor * p_ij)[:,None] * kernel_grad

            if nu != 0.0:
                u_ij = u_i - u_j
                # Inter-particle-averaged shear viscosity (harmonic mean) eq. (6), Adami (2013)
                eta_ij = 1.0
                temp = eta_ij * u_ij / (d_ij[:,None] + EPS) * kernel_der[:,None]
                # Eq. (10), Adami (2012)
                acc += nu * prefactor[:,None] * temp

            if is_tvf:
                # Add transport velocity acceleration term on top (Eq. 13)
                acc += 0.5 * (prefactor * kernel_der / (d_ij + EPS) * (-p_eos))[:,None] * r_ij

            return acc

        if nu != 0.0:
            u_i, u_j = u[i_s], u[j_s]
        else:
            u_i, u_j = None, None
        out = acceleration_fn(dr_ij, dist, rho[i_s], rho[j_s], p[i_s], p[j_s], u_i, u_j)
        acc = scatter_add(out, i_s, dim=0, dim_size=N_tot)
        acc = denormalize_length(acc)
        return acc

    return loop_body


if __name__ == "__main__":
    pass

    # Nx, L, dim = 16, 2 * np.pi, 2
    # dx = L / Nx
    # box_size = np.ones(dim) * L
    # u_ref = 7.0
    # dt = 0.0005
    # torch.set_printoptions(precision=8)

    # import random
    # random.seed(0)
    # np.random.seed(0)
    # noise = np.random.normal(0, dx/10, (Nx**dim, dim)) 
    # pos_demo = pos_init_cartesian_2d(box_size, dx) + noise
    # pos_demo = torch.tensor(pos_demo, dtype=torch.float32)
    # # print(pos_demo[:10])

    # is_tvf = False
    # nu = 0.0
    # relax_fn = relax_wrapper(
    #     Nx=Nx, dim=dim, L=L, is_physical=True, u_ref=u_ref, is_tvf=is_tvf, nu=nu
    # )
    # n_part_per_traj = torch.tensor([pos_demo.shape[0]], dtype=torch.int64)
        
    # for _ in range(10):
    #     if nu != 0.0:
    #         torch.manual_seed(0)
    #         u = torch.randn_like(pos_demo)/5 * u_ref
    #         acc = relax_fn(pos_demo, n_part_per_traj, u=u)
    #     else:
    #         acc = relax_fn(pos_demo, n_part_per_traj)
    #     pos_demo = shift_fn(pos_demo, dt**2 * (2*acc), box=box_size, pbc=True)
    
    
    ### Validate QuinticKernel
    # a = torch.linspace(0,0.4,100)
    # w = QuinticKernel(h=0.1).w(a)
    # g = QuinticKernel(h=0.1).grad_w(a)
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, 2)
    # axs[0].plot(a,w)
    # axs[1].plot(a,g)
    # fig.savefig("quintic_torch.png")