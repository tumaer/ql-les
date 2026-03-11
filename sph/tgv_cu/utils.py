"""Shared utilities for reading and writing tgv_cuda binary state files."""

from __future__ import annotations

from pathlib import Path
import struct

import numpy as np


def load_state(path: str | Path) -> dict[str, np.ndarray | int | float]:
    """Load one state file produced by tgv_cuda.

    Binary layout (little-endian):
    - int32 nx
    - int32 n
    - float64 t
    - x: n x 3 float64  (Vec3; z=0 for 2-D runs)
    - u: n x 3 float64  (Vec3; z=0 for 2-D runs)
    """
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 16:
        raise ValueError(f"File too small: {path}")

    nx, n = struct.unpack_from("<ii", raw, 0)
    t = struct.unpack_from("<d", raw, 8)[0]

    vec_bytes = n * 3 * 8
    expected = 16 + vec_bytes + vec_bytes
    if len(raw) != expected:
        raise ValueError(
            f"Unexpected file size for {path}. Expected {expected} bytes, got {len(raw)} bytes."
        )

    x = np.frombuffer(raw, dtype="<f8", count=n * 3, offset=16).reshape(n, 3).copy()
    u = np.frombuffer(raw, dtype="<f8", count=n * 3, offset=16 + vec_bytes).reshape(n, 3).copy()

    return {"nx": int(nx), "n": int(n), "t": float(t), "x": x, "u": u}


def write_state(
    path: str | Path,
    nx: int,
    n: int,
    t: float,
    x: np.ndarray,
    u: np.ndarray,
) -> None:
    """Write a state file compatible with tgv_cuda (Vec3 layout)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x_arr = np.asarray(x, dtype="<f8").reshape(n, 3)
    u_arr = np.asarray(u, dtype="<f8").reshape(n, 3)

    with path.open("wb") as f:
        f.write(struct.pack("<ii", int(nx), int(n)))
        f.write(struct.pack("<d", float(t)))
        f.write(x_arr.tobytes(order="C"))
        f.write(u_arr.tobytes(order="C"))
