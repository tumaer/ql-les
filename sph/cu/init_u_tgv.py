"""Re-initialise particle velocities from the analytic 2-D TGV field.

After running the init relaxation (tgv_cuda_init.conf), particle positions
have been relaxed to a near-uniform lattice.  This script loads the final
snapshot, resets the velocities to the exact TGV field, and saves the result
as ``relaxed_state.bin`` for use as the starting point of production runs.
"""

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np

from utils import load_state, write_state


def tgv2d_u(x: np.ndarray, k: int = 2) -> np.ndarray:
    """Return analytic 2-D TGV velocity at positions *x* (L=1 domain)."""
    u = np.zeros_like(x)
    u[:, 0] = -np.cos(k * np.pi * x[:, 0]) * np.sin(k * np.pi * x[:, 1])
    u[:, 1] = +np.sin(k * np.pi * x[:, 0]) * np.cos(k * np.pi * x[:, 1])
    return u


def tgv3d_u(x: np.ndarray) -> np.ndarray:
    """Return analytic 3-D TGV velocity at positions x (L=1 domain)."""
    u = np.zeros_like(x)
    u[:, 0] = +np.sin(x[:, 0]) * np.cos(x[:, 1]) * np.cos(x[:, 2])
    u[:, 1] = -np.cos(x[:, 0]) * np.sin(x[:, 1]) * np.cos(x[:, 2])
    u[:, 2] = 0.0
    return u


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write analytical velocity field")
    parser.add_argument("--case", type=str, help="Type of field, e.g., tgv2d/tgv3d")
    parser.add_argument("--src", type=Path, help="Path to a state_step file")
    args = parser.parse_args()

    src = args.src
    dst = args.src.parent / "relaxed_state.bin"

    state = load_state(src)
    if args.case == "tgv2d":
        u_new = tgv2d_u(state["x"])
    elif args.case == "tgv3d":
        u_new = tgv3d_u(state["x"])
    else:
        raise NotImplementedError
    write_state(dst, nx=state["nx"], n=state["n"], t=state["t"], x=state["x"], u=u_new)

    print(f"Loaded: {src}")
    print(f"Wrote:  {dst}")
    speed = np.linalg.norm(u_new, axis=1).max()
    print(f"nx={state['nx']}, n={state['n']}, t={state['t']:.6f}, umax={speed:.6e}")
