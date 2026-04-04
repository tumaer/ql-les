"""Utilities to inspect tgv_cuda state snapshots."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from utils import load_states, plt_evolution


def plt_field_2d(state: dict, fig_dir: Path = Path("fig"), vref=1) -> None:
    """Scatter-plot the x-velocity component of a snapshot."""
    x = state["x"]
    fig, ax = plt.subplots(figsize=(5, 4.5), layout="constrained")
    vmag = (state["u"] ** 2).sum(-1) ** 0.5
    scatter = ax.scatter(x[:, 0], x[:, 1], c=vmag, s=10, vmin=0, vmax=vref)
    ax.set_title("|u|")
    ax.set_aspect("equal")
    plt.colorbar(scatter, ax=ax)
    fig.savefig(fig_dir / f"ux_{state['t']:.4f}.png")
    plt.close(fig)


def plt_epsilon_tgv3d(df_diag: pd.DataFrame, fig_dir):
    """Plot kinetic energy dissipation rate.
    Compare SPH against reference from ref_tgv3d_Re100.csv [time, epsilon] every dt=0.1
    """
    df_ref = pd.read_csv("ref_tgv3d_Re100.csv")

    stride = max(len(df_diag) // 50, 1)
    df_sample = df_diag.iloc[::stride]
    t = df_sample["time"].to_numpy()
    ekin = df_sample["ekin"].to_numpy()
    dt = np.diff(t)
    epsilon = -np.diff(ekin) / dt / (2 * np.pi) ** 3  # normalize by domain volume
    t_mid = t[:-1] + dt / 2

    label = "epsilon"
    # increase default font size
    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(figsize=(4.5, 3.5), layout="constrained")
    ax.plot(t_mid, epsilon, "ok", fillstyle="none", label="SPH", markersize=8, lw=0.1)
    # ax.plot(t, epsilon, ".k", label="SPH", markersize=2)
    ax.plot(df_ref["time"].values, df_ref["epsilon"].values, "k", lw=2, label="DNS")
    ax.set_xlabel("Time")
    ax.set_ylabel(r"Dissipation rate $\epsilon$")
    ax.legend()
    # ax.set_ylim(0.004, 0.015)
    ax.set_xlim(-0.2, 10.2)
    ax.set_yticks([0.005, 0.010, 0.015])
    fig.savefig(fig_dir / f"tgv3d_{label}.png", dpi=300)
    fig.savefig(fig_dir / f"tgv3d_{label}.pdf", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse tgv_cuda output directory")
    parser.add_argument("--case", type=str, help="Type of field, e.g., tgv2d/tgv3d")
    parser.add_argument("--path", type=Path, help="Path to a save directory")
    args = parser.parse_args()

    assert args.path.is_dir()

    states = load_states(args.path)
    df = pd.read_csv(args.path / "diagnostics.csv")

    fig_dir = args.path / "fig"
    fig_dir.mkdir(exist_ok=True, parents=True)
    t = df["time"].values

    if args.case == "tgv2d" or args.case == "kolm2d":
        # uref column is only present for 2-D runs
        uref = df["uref"].values if "uref" in df.columns else None
        plt_evolution(t, df["umax"].values, "umax", uref, fig_dir, yscale="log")
        plt_evolution(t, df["rho_max"].values, "rho_max", None, fig_dir)
        plt_evolution(t, df["ekin"].values, "ekin", None, fig_dir, yscale="log")

        vref = {"tgv2d": 1, "kolm2d": 4}[args.case]
        if len(states) > 0:
            plt_field_2d(states[0], fig_dir, vref=vref)
        if len(states) > 2:
            plt_field_2d(states[1], fig_dir, vref=vref)
        if len(states) > 3:
            plt_field_2d(states[2], fig_dir, vref=vref)

    elif args.case == "tgv3d" or args.case == "hit3d":
        plt_evolution(t, df["umax"].values, "umax", None, fig_dir)
        plt_evolution(t, df["ekin"].values, "ekin", None, fig_dir, "log")
        plt_evolution(t, df["rho_max"].values, "rho_max", None, fig_dir)
        if args.case == "tgv3d":
            plt_epsilon_tgv3d(df, fig_dir)
    else:
        raise NotImplementedError
