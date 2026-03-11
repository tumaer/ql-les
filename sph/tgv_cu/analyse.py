"""Utilities to inspect tgv_cuda state snapshots."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from utils import load_state


def _step_from_name(path: Path) -> int:
    m = re.search(r"state_step_(\d+)\.bin$", path.name)
    return int(m.group(1)) if m else -1


def load_states(save_dir: str | Path) -> list[dict]:
    """Load and return all state files in *save_dir*, sorted by step index."""
    save_dir = Path(save_dir)
    files = sorted(save_dir.glob("state_step_*.bin"), key=_step_from_name)
    return [load_state(f) for f in files]


def plt_evolution(ts, val, label: str, ref=None, fig_dir: Path = Path("fig")) -> None:
    """Plot a scalar quantity over time and optionally overlay a reference curve."""
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(ts, val, label=label)
    if ref is not None:
        ax.plot(ts, ref, "k--", label="ref")
    ax.set_xlabel("Time")
    ax.set_ylabel(label)
    ax.set_yscale("log")
    ax.grid()
    fig.savefig(fig_dir / f"tgv_{label}.png")
    plt.close(fig)


def plt_field(state: dict, suffix: str, fig_dir: Path = Path("fig")) -> None:
    """Scatter-plot the x-velocity component of a snapshot."""
    x = state["x"]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x[:, 0], x[:, 1], c=state["u"][:, 0], s=5, vmin=-1, vmax=1)
    fig.savefig(fig_dir / f"ux{suffix}.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse tgv_cuda output directory")
    parser.add_argument("path", type=Path, help="Path to a save directory")
    args = parser.parse_args()

    if args.path.is_dir():
        states = load_states(args.path)
        df = pd.read_csv(args.path / "diagnostics.csv")

        fig_dir = args.path / "fig"
        fig_dir.mkdir(exist_ok=True, parents=True)

        # uref column is only present for 2-D runs
        uref = df["uref"].values if "uref" in df.columns else None
        plt_evolution(df["time"].values, df["umax"].values, "umax", uref, fig_dir)
        plt_evolution(df["time"].values, df["rho_max"].values, "rho_max", None, fig_dir)

        if len(states) > 0:
            plt_field(states[0], "_t0", fig_dir)
        if len(states) > 2:
            plt_field(states[2], "_t02", fig_dir)
