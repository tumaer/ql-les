"""Small utility to compare Kolm2d runs against the spectral reference."""

from __future__ import annotations

import argparse
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import _step_from_name, load_state
from src.utils.jax_utils.interpolate import ParticleGridInterpolator
from src.utils.jax_utils.jax_spectral import energy_spectrum

REF_PATH = Path("/local/disk/atoshev/dataset_kolm/raw/2D_KOLM_4096_140kevery1")
REF_TRAJ_IDS = range(15, 20)
REF_BURNIN = 4500
BOX_SIZE = 2 * np.pi


def _stats(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Extract min/max/median along axis=0."""
    return {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "med": np.median(arr, axis=0),
    }


def _step_from_h5_name(path: Path) -> int:
    """Simulation step from step_*.h5 l3es h5 file."""
    return int(path.stem.split("_")[1])


def _step_from_spectrum_name(path: Path) -> int:
    """Simulation step from spectrum_*.csv file from SPH CUDA solver."""
    return int(path.stem.split("_")[1])


def _latest_file(traj: Path, pattern: str, key: Callable[[Path], int]) -> Path | None:
    """Return file path of last file according to the pattern and key."""
    files = sorted(traj.glob(pattern), key=key)
    return files[-1] if files else None


@lru_cache(maxsize=None)
def _get_interpolator(nx: int) -> ParticleGridInterpolator:
    """Interpolator convenience wrapper."""
    return ParticleGridInterpolator(
        n_per_dim=(nx, nx),
        box_size=(BOX_SIZE, BOX_SIZE),
        dim=2,
        nufft_splits=2,
        nufft_backend="auto",
    )


def compute_spectrum_from_particles(r: np.ndarray, u: np.ndarray, nx: int) -> pd.DataFrame:
    """Get spectrum after converting particles to given grid."""
    interpolator = _get_interpolator(nx)
    r_src = np.asarray(r[:, :2])
    u_src = np.asarray(u[:, :2])
    start = time.time()
    # kolm2d_64:  MLS: 1.0s, Scipy: 0.8s, Spline: inf, NUFFT: 55s, DFT: inf
    # kolm2d_128: MLS: 1.3s, Scipy: 3.2s
    # kolm2d_256: MLS: 3.0s, Scipy: 14s
    # kolm2d_512: MLS: 10s
    u_grid = interpolator.p2g_mls(r_src=r_src, u_r=u_src)
    print("P2G interpolation t=", time.time() - start)
    spectrum_full = np.asarray(energy_spectrum(u_grid))
    k = np.arange(1, nx // 2 + 1)
    return pd.DataFrame({"k": k, "energy": spectrum_full[1 : nx // 2 + 1]})


def get_ref_ekin() -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Extract Ekin from diagnostics.csv files."""
    ekin = []
    last_df = None
    for idx in REF_TRAJ_IDS:
        df = pd.read_csv(REF_PATH / f"traj_{idx}/diagnostics.csv")
        ekin.append(df["ekin"].to_numpy())
        last_df = df

    if last_df is None:
        raise FileNotFoundError(f"No reference diagnostics found in {REF_PATH}")

    time = last_df["time"].to_numpy()[REF_BURNIN:] - 4.5
    ekin = np.array(ekin)[:, REF_BURNIN:]
    return _stats(ekin), time


def get_spectra(
    trajs: list[Path],
    source_pattern: str,
    source_step: Callable[[Path], int],
    loader: Callable[[Path], pd.DataFrame],
    recompute: bool = False,
) -> list[Path]:
    """Load or (if needed) compute spectra."""
    out_files: list[Path] = []
    for traj in trajs:
        cached = (
            None if recompute else _latest_file(traj, "spectrum_*.csv", _step_from_spectrum_name)
        )
        if cached is not None:
            out_files.append(cached)
            continue

        source_file = _latest_file(traj, source_pattern, source_step)
        if source_file is None:
            continue

        step = source_step(source_file)
        spectrum_path = traj / f"spectrum_{step}.csv"
        if not spectrum_path.exists() or recompute:
            df = loader(source_file)
            df.to_csv(spectrum_path, index=False)
        out_files.append(spectrum_path)
    return out_files


def _state_spectrum(file: Path) -> pd.DataFrame:
    """Load .bin state and compute its spectrum."""
    state = load_state(file)
    return compute_spectrum_from_particles(r=state["x"], u=state["u"], nx=int(state["nx"]))


def _ref_spectrum(file: Path) -> pd.DataFrame:
    """Load .h5 file and compute its spectrum."""
    with h5py.File(file, "r") as data:
        r = np.asarray(data["r"])
        u = np.asarray(data["u"])
    nx = int(round(np.sqrt(len(r))))
    return compute_spectrum_from_particles(r=r, u=u, nx=nx)


def load_spectra_stats(spectrum_files: list[Path]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load kolm2d_{nx}/spectrum_*.csv files."""
    spectra = []
    k = None
    for file in spectrum_files:
        df = pd.read_csv(file)
        k_i = df["k"].to_numpy()
        e_i = df["energy"].to_numpy()
        if k is None:
            k = k_i
        elif not np.array_equal(k, k_i):
            raise ValueError(f"Incompatible k-axis in {file}")
        spectra.append(e_i)

    if not spectra or k is None:
        raise ValueError("No spectra found")

    spectra_arr = np.array(spectra)
    return k, _stats(spectra_arr)


def plot_ekin(save_dir: Path, trajs: list[Path]) -> None:
    """Plot Ekin min/max/median over the test trajs."""
    trajs_ekin = []
    time = None
    for traj in trajs:
        df = pd.read_csv(traj / "diagnostics.csv")
        trajs_ekin.append(df["ekin"].to_numpy())
        if time is None:
            time = df["time"].to_numpy()

    if not trajs_ekin or time is None:
        raise FileNotFoundError(f"No diagnostics found under {save_dir}")

    ekin_stats = _stats(np.array(trajs_ekin))
    ekin_ref, t_ref = get_ref_ekin()

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(t_ref, ekin_ref["med"], "k", label="Reference")
    ax.fill_between(t_ref, ekin_ref["min"], ekin_ref["max"], color="k", alpha=0.2)
    ax.plot(time, ekin_stats["med"], label="Run")
    ax.fill_between(time, ekin_stats["min"], ekin_stats["max"], alpha=0.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Kinetic energy")
    ax.legend()
    fig.savefig(save_dir / "ekin.png")
    plt.close(fig)


def plot_spectra(save_dir: Path, trajs: list[Path], recompute: bool = False) -> None:
    """Plot spectral min/max/median over the test trajs."""
    run_spectra_files = get_spectra(
        trajs,
        "state_step_*.bin",
        _step_from_name,
        _state_spectrum,
        recompute=recompute,
    )
    ref_trajs = [REF_PATH / f"traj_{idx}" for idx in REF_TRAJ_IDS]
    ref_spectra_files = get_spectra(
        ref_trajs,
        "com/step_*.h5",
        _step_from_h5_name,
        _ref_spectrum,
        recompute=recompute,
    )

    k_ref, ref_stats = load_spectra_stats(ref_spectra_files)
    k_run, run_stats = load_spectra_stats(run_spectra_files)

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(k_ref, ref_stats["med"], "k", label="Reference")
    ax.fill_between(k_ref, ref_stats["min"], ref_stats["max"], color="k", alpha=0.2)
    ax.plot(k_run, run_stats["med"], label="Run")
    ax.fill_between(k_run, run_stats["min"], run_stats["max"], alpha=0.2)
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("Energy spectrum")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    fig.savefig(save_dir / "spectrum.png")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse Kolm2d output directory")
    parser.add_argument("--path", type=Path, required=True, help="Path to a save directory")
    parser.add_argument("--recompute-spectra", action="store_true", help="New spectrum_*.csv.")
    args = parser.parse_args()

    trajs = sorted(args.path.glob("traj_*/"))
    if not trajs:
        raise FileNotFoundError(f"No trajectories found in {args.path}")

    plot_ekin(args.path, trajs)
    plot_spectra(args.path, trajs, recompute=args.recompute_spectra)
