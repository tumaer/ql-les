"""Minimal JAX-SPH version of TGV 2D simulation. Requires installed JAX-SPH."""

import os
import warnings
import time
from typing import Callable, Dict

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import config, grad, jit, ops, vmap
from omegaconf import DictConfig, OmegaConf

config.update("jax_enable_x64", True)

from jax_sph import partition  # noqa: E402
from jax_sph.io_state import io_setup, read_h5, write_state  # noqa: E402
from jax_sph.jax_md import space  # noqa: E402
from jax_sph.jax_md.partition import Sparse  # noqa: E402
from jax_sph.utils import Logger, get_array_stats, pos_init_cartesian_2d, pos_init_cartesian_3d  # noqa: E402

EPS = jnp.finfo(float).eps

warnings.filterwarnings("ignore")


def get_ekin(state: Dict, dx: float):
    """Compute the kinetic energy of the fluid from `state["v"]`."""
    v = state["v"]
    ekin = jnp.square(v).sum().item()
    return 0.5 * ekin * dx ** v.shape[1]


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
        """Pressure from density."""
        return self.p_ref * ((rho / self.rho_ref) ** self.gamma - 1) + self.p_bg

    def rho_fn(self, p):
        """Density from pressure."""
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
            self._sigma = 7.0 / 478.0 / jnp.pi * self._one_over_h**2
        elif dim == 3:
            self._sigma = 3.0 / 359.0 / jnp.pi * self._one_over_h**3

    def w(self, r):
        """Evaluate the kernel function at distance r."""
        q = r * self._one_over_h
        q1 = jnp.maximum(0.0, 1.0 - q)
        q2 = jnp.maximum(0.0, 2.0 - q)
        q3 = jnp.maximum(0.0, 3.0 - q)

        return self._sigma * (q3**5 - 6.0 * q2**5 + 15.0 * q1**5)

    def grad_w(self, r):
        """Evaluate the kernel gradient at distance r."""
        return grad(self.w)(r)


class TGV:
    """Taylor-Green Vortex"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.case = cfg.case

        # get the config file name, e.g. "cases/db.yaml" -> "db"
        cfg.case.name = os.path.splitext(os.path.basename(cfg.config))[0]

        # relaxation configurations
        if self.case.mode == "rlx":
            self._set_default_rlx()

        if self.case.r0_type == "relaxed":
            self._load_only_fluid = False
            self._init_pos2D = self._get_relaxed_r0
            self._init_pos3D = self._get_relaxed_r0

    def _box_size2D(self):
        """The domain size in 2D."""
        return np.array([1.0, 1.0])

    def _box_size3D(self):
        """The domain size in 3D."""
        return 2 * np.pi * np.array([1.0, 1.0, 1.0])

    def _init_pos2D(self, box_size, dx):
        """Initialize positions on 2D Cartesian grid."""
        r = pos_init_cartesian_2d(box_size, dx)
        return r

    def _init_pos3D(self, box_size, dx):
        """Initialize positions on 3D Cartesian grid."""
        r = pos_init_cartesian_3d(box_size, dx)
        return r

    def _init_velocity2D(self, r):
        """Initialize velocity field in 2D according to the TGV definition."""
        x, y = r
        # from Transport Veocity paper by Adami et al. 2013
        u = -1.0 * jnp.cos(2.0 * jnp.pi * x) * jnp.sin(2.0 * jnp.pi * y)
        v = +1.0 * jnp.sin(2.0 * jnp.pi * x) * jnp.cos(2.0 * jnp.pi * y)

        return jnp.array([u, v])

    def _init_velocity3D(self, r):
        """Initialize velocity field in 3D according to the TGV definition."""
        x, y, z = r
        z_term = jnp.cos(z)
        u = +jnp.sin(x) * jnp.cos(y) * z_term
        v = -jnp.cos(x) * jnp.sin(y) * z_term
        w = 0.0
        return jnp.array([u, v, w])

    def initialize(self):
        """Case setup function."""
        cfg = self.cfg
        dx = cfg.case.dx
        dim = cfg.case.dim
        rho_ref = cfg.case.rho_ref
        viscosity = cfg.case.viscosity
        u_ref = cfg.case.u_ref
        cfl = cfg.solver.cfl

        key_prng = jax.random.PRNGKey(cfg.seed)

        # Primal: reference density, dynamic viscosity, and velocity
        # Derived: reference speed of sound, pressure
        c_ref = cfg.case.c_ref_factor * u_ref
        p_ref = rho_ref * c_ref**2 / cfg.eos.gamma
        # for free surface simulation p_background in the EoS has to be 0.0
        p_bg = cfg.eos.p_bg_factor * p_ref

        # calculate volume and mass
        h = dx
        volume_ref = h**dim
        mass_ref = volume_ref * rho_ref

        # time integration step dt
        dt_convective = cfl * h / (c_ref + u_ref)
        dt_viscous = cfl * h**2 * rho_ref / (viscosity + EPS)
        dt = np.amin([dt_convective, dt_viscous]).item()

        print("dt_convective :", dt_convective)
        print("dt_viscous    :", dt_viscous)
        print("dt_max_allowed:", dt)

        if cfg.solver.dt is not None:
            if cfg.solver.dt > dt:
                warnings.warn("Explicit dt should comply with CFL.", UserWarning)
            dt = cfg.solver.dt

        print("dt_final      :", dt)

        if cfg.case.mode == "rlx":
            # run a relaxation of randomly initialized state for 500 steps
            sequence_length = 5000
            cfg.solver.t_end = dt * sequence_length
            # turn background pressure on for homogeneous particle distribution
        else:
            sequence_length = int(cfg.solver.t_end / dt)

        # Equation of state
        eos = TaitEoS(p_ref, rho_ref, p_bg, cfg.eos.gamma)

        # initialize box and positions of particles
        if dim == 2:
            box_size = self._box_size2D()
            r = self._init_pos2D(box_size, dx)
        elif dim == 3:
            box_size = self._box_size3D()
            r = self._init_pos3D(box_size, dx)
        displacement_fn, shift_fn = space.periodic(side=box_size)

        num_particles = len(r)
        print("Total number of particles = ", num_particles)

        key, subkey = jax.random.split(key_prng)

        # initialize the velocity given the coordinates r with the noise
        if dim == 2:
            v = vmap(self._init_velocity2D)(r)
        elif dim == 3:
            v = vmap(self._init_velocity3D)(r)

        # initialize all other field values
        rho = jnp.ones(num_particles) * cfg.case.rho_ref
        mass = jnp.ones(num_particles) * mass_ref
        eta = jnp.ones(num_particles) * cfg.case.viscosity

        # initialize the state dictionary
        state = {
            "r": r,
            "u": v,
            "v": v,
            "dudt": jnp.zeros_like(v),
            "dvdt": jnp.zeros_like(v),
            "rho": rho,
            "p": eos.p_fn(rho),
            "mass": mass,
            "eta": eta,
        }

        # the following arguments are needed for dataset generation
        cfg.case.c_ref, cfg.case.p_ref, cfg.case.p_bg = c_ref, p_ref, p_bg
        cfg.solver.dt, cfg.solver.sequence_length = dt, sequence_length
        cfg.case.num_particles_max = num_particles
        cfg.case.pbc = [True, True, True]
        cfg.case.bounds = np.array([np.zeros_like(box_size), box_size]).T.tolist()

        return (
            cfg,
            box_size,
            state,
            eos,
            key,
            displacement_fn,
            shift_fn,
        )


def rho_summation_fn(mass, i_s, w_dist, N):
    """Density summation."""
    return mass * ops.segment_sum(w_dist, i_s, N)


def acceleration_tvf_fn_wrapper(kernel_fn):
    """Transport velocity formulation acceleration according to Adami et al. 2013."""

    def acceleration_tvf_fn(r_ij, d_ij, rho_i, rho_j, m_i, m_j, p_bg_i):
        # compute the common prefactor `_c`
        _weighted_volume = ((m_i / rho_i) ** 2 + (m_j / rho_j) ** 2) / m_i
        _kernel_grad = kernel_fn.grad_w(d_ij)
        _c = _weighted_volume * _kernel_grad / (d_ij + EPS)

        # (Eq. 13) - or at least the acceleration term
        a_eq_13 = _c * 1.0 * p_bg_i * r_ij

        return a_eq_13

    return acceleration_tvf_fn


def tvf_stress_fn(rho: float, u, v):
    """Transport velocity stress tensor. See 'A' under (Eq. 4) in Adami et al. 2013."""
    return jnp.outer(rho * u, v - u)


def acceleration_standard_fn_wrapper(kernel_fn):
    """Standard SPH acceleration according to Adami et al. 2012."""

    def acceleration_standard_fn(
        r_ij,
        d_ij,
        rho_i,
        rho_j,
        u_i,
        u_j,
        v_i,
        v_j,
        m_i,
        m_j,
        eta_i,
        eta_j,
        p_i,
        p_j,
    ):
        # (Eq. 6) - inter-particle-averaged shear viscosity (harmonic mean)
        eta_ij = 2 * eta_i * eta_j / (eta_i + eta_j + EPS)
        # (Eq. 7) - density-weighted pressure (weighted arithmetic mean)
        p_ij = (rho_j * p_i + rho_i * p_j) / (rho_i + rho_j)

        # compute the common prefactor `_c`
        _weighted_volume = ((m_i / rho_i) ** 2 + (m_j / rho_j) ** 2) / m_i
        _kernel_grad = kernel_fn.grad_w(d_ij)
        _c = _weighted_volume * _kernel_grad / (d_ij + EPS)

        # (Eq. 8): \boldsymbol{e}_{ij} is computed as r_ij/d_ij here.
        _A = (tvf_stress_fn(rho_i, u_i, v_i) + tvf_stress_fn(rho_j, u_j, v_j)) / 2
        _u_ij = u_i - u_j
        a_eq_8 = _c * (-p_ij * r_ij + jnp.dot(_A, r_ij) + eta_ij * _u_ij)
        return a_eq_8

    return acceleration_standard_fn


class WCSPH:
    """Weakly compressible SPH solver with transport velocity formulation."""

    def __init__(
        self,
        displacement_fn: Callable,
        eos: TaitEoS,
        dx: float,
        dim: int,
        dt: float,
        c_ref: float,
    ):
        self.displacement_fn = displacement_fn
        self.dt = dt
        self.eos = eos
        self.c_ref = c_ref

        self._kernel_fn = QuinticKernel(h=dx, dim=dim)
        self._acceleration_tvf_fn = acceleration_tvf_fn_wrapper(self._kernel_fn)
        self._acceleration_fn = acceleration_standard_fn_wrapper(self._kernel_fn)

    def forward_wrapper(self):
        """Wrapper of update step of SPH."""

        def forward(state, neighbors):
            """Update step of SPH solver.

            Args:
                state (dict): Flow fields and particle properties.
                neighbors (_type_): Neighbors object.
            """

            r, mass, eta = state["r"], state["mass"], state["eta"]
            u, v, dudt, dvdt = state["u"], state["v"], state["dudt"], state["dvdt"]
            rho, p = state["rho"], state["p"]
            N = len(r)

            # precompute displacements `dr` and distances `dist`
            # the second vector is sorted
            i_s, j_s = neighbors.idx
            r_i_s, r_j_s = r[i_s], r[j_s]
            dr_i_j = vmap(self.displacement_fn)(r_i_s, r_j_s)
            dist = space.distance(dr_i_j)
            w_dist = vmap(self._kernel_fn.w)(dist)

            ##### Density summation or evolution
            rho = rho_summation_fn(mass, i_s, w_dist, N)

            ##### Compute primitives
            # pressure, and background pressure
            p = vmap(self.eos.p_fn)(rho)
            background_pressure_tvf = vmap(self.eos.p_fn)(jnp.zeros_like(p))

            ##### Compute RHS

            out = vmap(self._acceleration_fn)(
                dr_i_j,
                dist,
                rho[i_s],
                rho[j_s],
                u[i_s],
                u[j_s],
                v[i_s],
                v[j_s],
                mass[i_s],
                mass[j_s],
                eta[i_s],
                eta[j_s],
                p[i_s],
                p[j_s],
            )

            dudt = ops.segment_sum(out, i_s, N)

            out_tv = vmap(self._acceleration_tvf_fn)(
                dr_i_j,
                dist,
                rho[i_s],
                rho[j_s],
                mass[i_s],
                mass[j_s],
                background_pressure_tvf[i_s],
            )
            dvdt = ops.segment_sum(out_tv, i_s, N)

            state = {
                "r": r,
                "u": u,
                "v": v,
                "dudt": dudt,
                "dvdt": dvdt,
                "rho": rho,
                "p": p,
                "mass": mass,
                "eta": eta,
            }

            return state

        return forward


def si_euler(tvf: float, model: Callable, shift_fn: Callable):
    """Semi-implicit Euler integrator for transport velocity formulation. See Adami et al. 2013."""

    def advance(dt: float, state: Dict, neighbors):
        """Call to integrator."""

        # 1. Twice 1/2dt integration of u and v
        state["u"] += 1.0 * dt * state["dudt"]
        state["v"] = state["u"] + tvf * 0.5 * dt * state["dvdt"]

        # 2. Integrate position with velocity v
        state["r"] = shift_fn(state["r"], 1.0 * dt * state["v"])

        # 3. Update neighbor list
        num_particles = len(state["r"])
        neighbors = neighbors.update(state["r"], num_particles=num_particles)

        # 4. Compute accelerations
        state = model(state, neighbors)

        return state, neighbors

    return advance


def simulate(cfg):
    case = TGV(cfg)
    (
        cfg,
        box_size,
        state,
        eos_fn,
        key,
        displacement_fn,
        shift_fn,
    ) = case.initialize()

    solver = WCSPH(
        displacement_fn,
        eos_fn,
        cfg.case.dx,
        cfg.case.dim,
        cfg.solver.dt,
        cfg.case.c_ref,
    )
    forward = solver.forward_wrapper()

    # Initialize a neighbors list for looping through the local neighborhood
    # cell_size = r_cutoff + dr_threshold
    # capacity_multiplier is used for preallocating the (2, NN) neighbors.idx
    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box_size,
        r_cutoff=solver._kernel_fn.cutoff,
        backend=cfg.nl.backend,
        capacity_multiplier=1.25,
        mask_self=False,
        format=Sparse,
        num_particles_max=state["r"].shape[0],
        num_partitions=cfg.nl.num_partitions,
        pbc=np.array(cfg.case.pbc),
    )
    num_particles = len(state["r"])
    neighbors = neighbor_fn.allocate(state["r"], num_particles=num_particles)

    # Instantiate advance function for our use case
    advance = si_euler(cfg.solver.tvf, forward, shift_fn)

    advance = advance if cfg.no_jit else jit(advance)

    # compile kernel and initialize accelerations
    _state, _neighbors = advance(0.0, state, neighbors)
    _state["v"].block_until_ready()

    # create data directory and dump config.yaml
    dir = io_setup(cfg)
    # set up progress bar logging
    logger = Logger(
        dt=cfg.solver.dt,
        dx=cfg.case.dx,
        print_props=cfg.io.print_props,
        sequence_length=cfg.solver.sequence_length,
    )

    start = time.time()
    for step in range(cfg.solver.sequence_length + 2):
        write_state(step - 1, state, dir, cfg)

        state_, neighbors_ = advance(cfg.solver.dt, state, neighbors)

        # Check whether the edge list is too small and if so, create longer one
        if neighbors_.did_buffer_overflow:
            edges_ = neighbors.idx.shape
            print(f"Reallocate neighbors list {edges_} at step {step}")
            neighbors = neighbor_fn.allocate(state["r"], num_particles=num_particles)
            print(f"To list {neighbors.idx.shape}")

            # To run the loop N times even if sometimes did_buffer_overflow > 0
            # we directly rerun the advance step here
            state, neighbors = advance(cfg.solver.dt, state, neighbors)
        else:
            state, neighbors = state_, neighbors_

        # update the progress bar
        if step % cfg.io.write_every == 0:
            logger.print_stats(state, step)

    print(f"time: {time.time() - start:.2f} s")


def val_TGV(val_root, dim=2, nxs=[50, 100], save_fig=False):
    """Validate the SPH implementation of the Taylor-Green Vortex

    Args:
        dim (int, optional): Dimension. Defaults to 2.
        nxs (list, optional): Set of 1/dx values. Defaults to [10, 20, 50].
        save_fig (bool, optional): Whether to save output. Defaults to False.
    """

    # get all existing directories with relevant statistics
    if dim == 2:
        tvf = "tgv2d_tvf/"

    dirs_tvf = os.listdir(os.path.join(val_root, tvf))
    dirs_tvf = [d for d in dirs_tvf if ("2D_TGV_SPH" in d)]
    dirs_tvf = sorted(dirs_tvf)

    # to each directory read the medatada file and store nx values
    nx_found_tvf = {}
    for dir in dirs_tvf:
        cfg = OmegaConf.load(os.path.join(val_root, tvf, dir, "config.yaml"))
        side_length = cfg.case.bounds[0][1] - cfg.case.bounds[0][0]
        nx_found_tvf[round(side_length / cfg.case.dx)] = [dir, cfg]

    # verify that all requested nx values are available
    for nx in nxs:
        assert nx in nx_found_tvf, FileNotFoundError(f"Simulation nx={nx} missing")

    # plots
    fig1, axs1 = plt.subplots(1, 2, figsize=(10, 5))
    temp = 0

    for nx in nxs:
        temp = temp + 1

        cfg_tvf = nx_found_tvf[nx][1]

        dir_path_tvf = nx_found_tvf[nx][0]
        files_tvf = os.listdir(os.path.join(val_root, tvf, dir_path_tvf))
        files_h5_tvf = [f for f in files_tvf if (".h5" in f)]
        files_h5_tvf = sorted(files_h5_tvf)

        u_max_vec_tvf = np.zeros((len(files_h5_tvf)))
        e_kin_vec_tvf = np.zeros((len(files_h5_tvf)))

        for i, filename in enumerate(files_h5_tvf):
            state = read_h5(os.path.join(val_root, tvf, dir_path_tvf, filename))
            u_max_vec_tvf[i] = get_array_stats(state, "v")
            e_kin_vec_tvf[i] = get_ekin(state, cfg_tvf.case.dx)
        e_kin_vec_tvf /= side_length**dim

        # plot
        t_tvf = np.linspace(0.0, cfg_tvf.solver.t_end, len(u_max_vec_tvf))
        if dim == 2:
            lbl1 = f"SPH + tvf, dx={cfg_tvf.case.dx}"
            axs1[0].plot(t_tvf, u_max_vec_tvf, label=lbl1)
            axs1[1].plot(t_tvf, e_kin_vec_tvf, label=lbl1)

    # reference solutions in 2D and 3D
    if dim == 2:
        # x-axis
        t = np.linspace(0.0, cfg_tvf.solver.t_end, 100)

        rho = 1.0
        u_ref = 1.0
        L = 1.0
        eta = 0.01
        Re = rho * u_ref * L / eta
        slope_u_max = -8 * np.pi**2 / Re
        u_max_theory = np.exp(slope_u_max * t)
        e_kin_theory = 0.25 * np.exp(2 * slope_u_max * t)

        axs1[0].plot(t, u_max_theory, "k", label="Theory")
        axs1[1].plot(t, e_kin_theory, "k", label="Theory")

    # plot layout

    axs1[0].set_yscale("log")
    axs1[0].set_xlabel(r"$t$ [-]")
    axs1[0].set_ylabel(r"$u_{max}$ [-]" if dim == 2 else r"$dE/dt$ [-]")
    axs1[0].set_xlim([0, cfg.solver.t_end])
    axs1[0].legend()
    axs1[0].grid()

    axs1[1].set_yscale("log")
    axs1[1].set_xlabel(r"$t$ [-]")
    axs1[1].set_ylabel(r"$E_{kin}$ [-]")
    axs1[1].set_xlim([0, cfg.solver.t_end])
    axs1[1].legend()
    axs1[1].grid()

    fig1.suptitle(f"{str(dim)}D Taylor-Green Vortex")
    fig1.tight_layout()

    ###### save or visualize
    if save_fig:
        os.makedirs(f"{val_root}", exist_ok=True)
        nxs_str = "_".join([str(i) for i in nxs])
        plt.savefig(f"{val_root}/{str(dim)}D_TGV_{nxs_str}.pdf")


cfg_tgv = OmegaConf.create(
    {
        "config": "cases/tgv.yaml",
        "seed": 123,
        "no_jit": False,
        "gpu": 0,
        "dtype": "float64",
        "xla_mem_fraction": 0.8,
        "case": {
            "source": "tgv.py",
            "mode": "sim",
            "dim": 2,
            "dx": 0.02,
            "r0_type": "cartesian",
            "viscosity": 0.01,
            "u_ref": 1.0,
            "c_ref_factor": 10.0,
            "rho_ref": 1.0,
            "name": "tgv",
            "c_ref": 10.0,
            "p_ref": 100.0,
            "p_bg": 0.0,
            "num_particles_max": 2500,
            "pbc": [True, True, True],
            "bounds": [[0.0, 1.0], [0.0, 1.0]],
        },
        "solver": {
            "name": "SPH",
            "tvf": 1.0,
            "cfl": 0.275,
            "dt": 0.0005,
            "t_end": 6,
            "sequence_length": 12000,
        },
        "eos": {
            "name": "Tait",
            "gamma": 1.0,
            "p_bg_factor": 0.0,
        },
        "nl": {
            "backend": "jaxmd_vmap",
            "num_partitions": 1,
        },
        "io": {
            "write_type": ["h5"],
            "write_every": 25,
            "data_path": "data_valid/tgv2d_tvf/",
            "print_props": ["u_max"],
        },
    }
)


if __name__ == "__main__":
    simulate(cfg_tgv)
    val_TGV("data_valid/", dim=2, nxs=[50], save_fig=True)
