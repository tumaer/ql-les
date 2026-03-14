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


def plt_epsilon_tgv2d(states, fig_dir):
    """Plot kinetic energy dissipation rate.
    Compare SPH against reference from ref_tgv3d_Re100.csv [time, epsilon] every dt=0.1
    """
    df_ref = pd.read_csv("ref_tgv3d_Re100.csv")

    dt_sim = states[1]["t"] - states[0]["t"]

    def ekin_fn(u):
        return 0.5 * (u**2).sum(-1).mean()

    ekin = np.array([ekin_fn(state["u"]) for state in states])
    epsilon = -(ekin[1:] - ekin[:-1]) / dt_sim
    t = np.array([state["t"] for state in states[1:]])

    label = "epsilon"
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(t, epsilon, label="SPH")
    ax.plot(df_ref["time"].values, df_ref["epsilon"].values, "k--", label="ref")
    ax.set_xlabel("Time")
    ax.set_ylabel(label)
    ax.set_yscale("log")
    ax.grid()
    fig.savefig(fig_dir / f"tgv_{label}.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse tgv_cuda output directory")
    parser.add_argument("--type", type=str, help="Type of field, e.g., tgv2d/tgv3d")
    parser.add_argument("--path", type=Path, help="Path to a save directory")
    args = parser.parse_args()

    assert args.path.is_dir()

    if args.type == "tgv2d" or args.type == "kolm2d":
        states = load_states(args.path)
        df = pd.read_csv(args.path / "diagnostics.csv")

        fig_dir = args.path / "fig"
        fig_dir.mkdir(exist_ok=True, parents=True)

        # uref column is only present for 2-D runs
        uref = df["uref"].values if "uref" in df.columns else None
        t = df["time"].values
        plt_evolution(t, df["umax"].values, "umax", uref, fig_dir, yscale="log")
        plt_evolution(t, df["rho_max"].values, "rho_max", None, fig_dir)
        plt_evolution(t, df["ekin"].values, "ekin", None, fig_dir, yscale="log")

        vref = {"tgv2d": 1, "kolm2d": 4}[args.type]
        if len(states) > 0:
            plt_field_2d(states[0], fig_dir, vref=vref)
        if len(states) > 2:
            plt_field_2d(states[2], fig_dir, vref=vref)
        if len(states) > 3:
            plt_field_2d(states[-1], fig_dir, vref=vref)

    elif args.type == "tgv3d":
        states = load_states(args.path)
        df = pd.read_csv(args.path / "diagnostics.csv")

        fig_dir = args.path / "fig"
        fig_dir.mkdir(exist_ok=True, parents=True)

        plt_evolution(df["time"].values, df["umax"].values, "umax", None, fig_dir)
        plt_evolution(df["time"].values, df["rho_max"].values, "rho_max", None, fig_dir)
        plt_epsilon_tgv2d(states, fig_dir)
    else:
        raise NotImplementedError
