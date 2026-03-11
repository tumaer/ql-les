"""Re-initialise particle velocities from the analytic 2-D TGV field.

After running the init relaxation (tgv_cuda_init.conf), particle positions
have been relaxed to a near-uniform lattice.  This script loads the final
snapshot, resets the velocities to the exact TGV field, and saves the result
as ``relaxed_state.bin`` for use as the starting point of production runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from utils import load_state, write_state


def tgv_velocity(x: np.ndarray) -> np.ndarray:
    """Return analytic 2-D TGV velocity at positions *x* (L=1 domain).

    Accepts arrays of shape (n, 2) or (n, 3); the z-column is left as zero.
    """
    u = np.zeros_like(x)
    u[:, 0] = -np.cos(2.0 * np.pi * x[:, 0]) * np.sin(2.0 * np.pi * x[:, 1])
    u[:, 1] = +np.sin(2.0 * np.pi * x[:, 0]) * np.cos(2.0 * np.pi * x[:, 1])
    return u


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    src = root / "res_tgv2d_init/state_step_00006000.bin"
    dst = root / "res_tgv2d_init/relaxed_state.bin"

    state = load_state(src)
    u_new = tgv_velocity(state["x"])
    write_state(dst, nx=state["nx"], n=state["n"], t=state["t"], x=state["x"], u=u_new)

    print(f"Loaded: {src}")
    print(f"Wrote:  {dst}")
    speed = np.linalg.norm(u_new, axis=1).max()
    print(f"nx={state['nx']}, n={state['n']}, t={state['t']:.6f}, umax={speed:.6e}")
