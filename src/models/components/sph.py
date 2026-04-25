"""Utilities for SPH particle relaxation. See Neural SPH (Toshev et al., 2024)."""

import numpy as np
import torch
from torch_scatter import scatter_add

from src.utils.nbrs_utils import pos_init_cartesian_2d, shift_fn, displ_fn, nearest
from src.utils.interpolate import QuinticKernel

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
        """Compute pressure from density."""
        return self.p_ref * ((rho / self.rho_ref) ** self.gamma - 1) + self.p_bg

    def rho_fn(self, p):
        """Compute density from pressure."""
        p_temp = p + self.p_ref - self.p_bg
        return self.rho_ref * (p_temp / self.p_ref) ** (1 / self.gamma)


def relax_wrapper(
    Nx,
    dim=2,
    L=2 * np.pi,
    is_physical=False,
    u_ref=None,
    is_tvf=True,
    nu=0.0,
    box=None,
    tvf_factor=1.0,
    separate_tvf=False,
    is_clip_rho=True,
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

    def loop_body(r, n_part_per_traj, u=None, verbose=False, return_all=False):
        if nu != 0.0:
            assert (u is not None) and (r.shape == u.shape), "If nu!=0, u needed."

        r = normalize_length(r)

        edge_index = nearest(r, n_part_per_traj, pbc, box, cutoff=kernel_fn.cutoff)
        i_s, j_s = edge_index
        # print(i_s[j_s==0].sort()[0])
        # print(sum(i_s!=len(r)))
        r_i, r_j = r[i_s], r[j_s]
        dr_ij = displ_fn(r_i, r_j, box, pbc)
        dist = torch.norm(dr_ij, dim=-1)
        w_dist = kernel_fn.w(dist)

        rho = mass * scatter_add(w_dist, i_s, dim=0, dim_size=N_tot)
        if is_clip_rho:
            # Following NeuralSPH (https://arxiv.org/abs/2402.06275), we clip density
            rho = torch.clamp(rho, max=1.02 * rho_ref)
            rho = torch.where(rho < 0.98 * rho_ref, rho_ref, rho)
        p = eos.p_fn(rho)
        if verbose:
            print(
                f"Density min/max/std: {rho.min().item():.4f}, {rho.max().item():.4f}, {rho.std().item():.4f}"
            )

        def acceleration_fn(r_ij, d_ij, rho_i, rho_j, p_i, p_j, u_i=None, u_j=None):
            # Compute unit vector, above eq. (6), Zhang (2017). Sign flipped here.
            e_ij = r_ij / (d_ij[:, None] + EPS)

            # Compute kernel gradient
            kernel_der = kernel_fn.grad_w(d_ij)
            kernel_grad = kernel_der[:, None] * e_ij

            # Compute density-weighted pressure (weighted arithmetic mean)
            p_ij = (rho_j * p_i + rho_i * p_j) / (rho_i + rho_j)

            # Eq. (8), Adami (2012) with constant `mass`
            prefactor = mass * ((1 / rho_i) ** 2 + (1 / rho_j) ** 2)
            acc = (-prefactor * p_ij)[:, None] * kernel_grad

            if nu != 0.0:
                u_ij = u_i - u_j
                # Inter-particle-averaged shear viscosity (harmonic mean) eq. (6), Adami (2013)
                eta_ij = 1.0
                temp = eta_ij * u_ij / (d_ij[:, None] + EPS) * kernel_der[:, None]
                # Eq. (10), Adami (2012)
                acc += nu * prefactor[:, None] * temp
                # if only_visc:
                #     acc = nu * prefactor[:, None] * temp

            if is_tvf:
                # Add transport velocity acceleration term on top (Eq. 13)
                dvdt = (prefactor * kernel_der / (d_ij + EPS) * (-p_eos))[:, None] * r_ij
                acc_tvf = 0.5 * tvf_factor * dvdt  # 0.5 is from integrator.
                if separate_tvf:
                    return {"acc": acc, "acc_tvf": acc_tvf}
                else:
                    return {"acc": acc + acc_tvf}
            return {"acc": acc}

        if nu != 0.0:
            u_i, u_j = u[i_s], u[j_s]
        else:
            u_i, u_j = None, None
        out = acceleration_fn(dr_ij, dist, rho[i_s], rho[j_s], p[i_s], p[j_s], u_i, u_j)
        if is_tvf and separate_tvf:
            acc_tvf = scatter_add(out["acc_tvf"], i_s, dim=0, dim_size=N_tot)
            acc_tvf = denormalize_length(acc_tvf)
        acc = scatter_add(out["acc"], i_s, dim=0, dim_size=N_tot)
        acc = denormalize_length(acc)

        if not return_all:
            # return only the acceleration
            return acc
        else:
            tvf_res = {"acc_tvf": acc_tvf} if (is_tvf and separate_tvf) else {}
            return {"acc": acc, "rho": rho, "p": p, "edge_index": edge_index} | tvf_res

    return loop_body


if __name__ == "__main__":
    torch.set_printoptions(precision=8)
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import os

    os.makedirs("figs_0", exist_ok=True)  # run from Cartesian grid
    os.makedirs("figs_1", exist_ok=True)  # run from relaxed positions
    fig_dir = "figs_1" if os.path.exists("figs_0/tgv_2d_final_tvf.pt") else "figs_0"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def plt_frame(x, u, t, suffix=None):
        """Plot a single frame of the TGV velocity field."""
        fig, ax = plt.subplots(layout="constrained")
        sc = ax.scatter(x[:, 0], x[:, 1], c=u, s=5, vmin=0.98, vmax=1.02)
        ax.set_title(f"t={t:.3f}")
        ax.set_aspect("equal")
        cbar = plt.colorbar(sc, ax=ax, orientation="vertical")
        cbar.set_label("Density")
        plt.savefig(f"{fig_dir}/tgv_2d_{t:.3f}{suffix}.png")

    def plt_evolution(ts, val, label, ref=None):
        """Plot a scalar quantity over time and optionally overlay a reference curve."""
        fig, ax = plt.subplots(layout="constrained")
        ax.plot(ts, val)
        ax.set_xlabel("Time")
        ax.set_ylabel(label)
        if ref is not None:
            ax.plot(ts, ref, "k--")
        ax.set_yscale("log")
        ax.grid()
        fig.savefig(f"{fig_dir}/tgv_2d_{label}.png")

    #### Run TGV 2D to validate the SPH implementation.
    # based on: https://github.com/tumaer/jax-sph/blob/main/cases/tgv.py
    Nx, Lx, dim = 50, 1.0, 2
    dx = Lx / Nx
    box_size = torch.ones(dim).to(device) * Lx
    u_ref = 1.0
    nu = 0.01
    dt = 0.0005  # CFL * dx / (11 * u_ref) = 0.25 * 0.02 / (11 * 1) = 0.000454545...
    t_end = 3
    for is_tvf in [False, True]:
        suffix = "_tvf" if is_tvf else ""
        if os.path.exists("figs_0/tgv_2d_final_tvf.pt"):
            print("Loading existing x ...")
            x = torch.load("figs_0/tgv_2d_final_tvf.pt", map_location=device)["x"]
        else:
            x = pos_init_cartesian_2d(box_size.cpu(), dx)
            x = torch.tensor(x, dtype=torch.float32).to(device)
        ux = -1.0 * torch.cos(2.0 * np.pi * x[:, 0]) * torch.sin(2.0 * np.pi * x[:, 1])
        uy = +1.0 * torch.sin(2.0 * np.pi * x[:, 0]) * torch.cos(2.0 * np.pi * x[:, 1])
        u = torch.stack([ux, uy], dim=1).to(device)
        v = u.clone()
        kwargs = {"Nx": Nx, "dim": dim, "L": Lx, "is_physical": True, "u_ref": u_ref, "nu": nu}
        kwargs["box"] = box_size
        sph_fn = relax_wrapper(**kwargs, is_tvf=is_tvf, separate_tvf=True)
        n_part_per_traj = torch.tensor([x.shape[0]], dtype=torch.int64).to(device)

        umax, rho_max = [], []
        ts = [i * dt for i in range(round(t_end / dt) + 1)]
        u_max_ref = np.exp(-8 * np.pi**2 * nu * np.array(ts)) * u_ref
        for t in tqdm(ts):
            outputs = sph_fn(x, n_part_per_traj, u=u, return_all=True)
            u = u + dt * outputs["acc"]
            v = u + dt * (outputs["acc_tvf"] if is_tvf else 0)  # factor 0.5 in acc_tvf
            x = shift_fn(x, dt * v, box=box_size, pbc=True)
            umax.append(u.abs().max().item())
            rho_max.append(outputs["rho"].max().item())
            if t in [0.0, 0.2, 0.4, 0.6]:
                plt_frame(x.cpu(), outputs["rho"].cpu(), t, suffix)
        torch.save({"x": x.cpu(), "u": u.cpu()}, f"{fig_dir}/tgv_2d_final{suffix}.pt")
        plt_evolution(ts, umax, "u_max" + suffix, u_max_ref)
        plt_evolution(ts, rho_max, "rho_max" + suffix)

    #### Plot QuinticKernel
    a = torch.linspace(0, 0.4, 100)
    w = QuinticKernel(h=0.1).w(a)
    g = QuinticKernel(h=0.1).grad_w(a)

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(a, w)
    axs[1].plot(a, g)
    for ax in axs:
        ax.grid()
    fig.savefig(f"{fig_dir}/quintic_torch.png")
