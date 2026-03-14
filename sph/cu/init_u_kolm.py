"""Re-initialise particle velocities."""

from __future__ import annotations

from pathlib import Path
import argparse
import time

import numpy as np

from utils import load_state, write_state
from analyse import plt_field_2d
from src.utils.jax_utils.interpolate import ParticleGridInterpolator
from src.utils.jax_utils.jax_spectral import spectral_filtering


def get_ur_from_npy(u_path, x, Nx):
    """x.shape: (N, 3)"""
    u = np.load(u_path)  # (2, 512, 512)
    N_hres = u.shape[1]
    L = 2 * np.pi

    u_lres = u if N_hres == Nx else spectral_filtering(u, Nx)  # (2, Nx, Nx)
    interpolator = ParticleGridInterpolator(
        n_per_dim=(Nx, Nx),
        box_size=(L, L),
        dim=2,
        nufft_splits=2,
        nufft_backend="auto",
    )

    start = time.time()
    u_x = interpolator.g2p_nufft(r_target=x[:, :2], u=u_lres)
    print("NUFFT interpolation t=", time.time() - start)

    ##########
    # u_back = interpolator.p2g_scipy(r_src=x[:, :2], u_r=u_x)
    # print(f"{u_lres.shape=}, {u_x.shape=}, {u_back.shape=}")
    # import matplotlib.pyplot as plt
    # vmag = 4
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    # for ax, title, field in zip(
    #     axs, ["u_before", "u_r", "u_after"], [u_lres[0], u_x[:,0], u_back[0]]
    # ):
    #     ax.set_title(title + f" [{field.min():.4f}, {field.max():.4f}]")
    # axs[0].imshow(u_lres[0], origin="lower", vmin=-vmag, vmax=vmag)
    # axs[1].scatter(x[:, 0], x[:, 1], c=u_x[:, 0], s=1, vmin=-vmag, vmax=vmag)
    # axs[1].set_xlim(0, L); axs[1].set_ylim(0, L)
    # axs[2].imshow(u_back[0], origin="lower", vmin=-vmag, vmax=vmag)
    # fig.savefig("debug_interp.png")
    # plt.close()
    #############

    u_new = np.zeros_like(x)
    u_new[:, 0] = u_x[:, 0]
    u_new[:, 1] = u_x[:, 1]
    return u_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write analytical velocity field")
    parser.add_argument("--src_pos", type=Path, help="Path to a state_step file")
    parser.add_argument("--src_u", type=Path, help="Path to a .npy file containing u")
    args = parser.parse_args()

    src_pos = args.src_pos

    state = load_state(src_pos)  # positions
    Nx = int(len(state["x"]) ** 0.5)  # number of particles along x

    # Load velocity field from npy
    # cp /local/disk/atoshev/dataset_kolm/raw/2D_KOLM_4096_140kevery1/traj_19/u_512_04500_burnin.npy traj_19
    trajs = sorted(args.src_u.glob("traj_*/"))
    assert args.src_u.is_dir() and len(trajs) == 5, (
        f"Expected 5 trajectories in {args.src_u}, found {len(trajs)}"
    )
    for traj in trajs:
        u_path = traj / "u_512_04500_burnin.npy"
        u = get_ur_from_npy(u_path, state["x"], Nx)
        dst = traj / "relaxed_state.bin"
        write_state(dst, nx=state["nx"], n=state["n"], t=state["t"], x=state["x"], u=u)

    print(f"Loaded: {args.src_pos=}, {args.src_u=}")
    state["u"] = np.asarray(u, dtype="<f8").reshape(len(state["x"]), 3)
    print(f"{state['nx']=}, {state['n']=}, {state['t']=}")
    print(f"{state['x'].shape=}, {state['x'].max()=}")
    print(f"{state['u'].shape=}, {state['u'].max()=}")
    plt_field_2d(state, fig_dir=traj, vref=4)
