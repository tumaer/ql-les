"""Utilities to analyse Kolm2d and HIT3d runs against DNS references."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
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

BOX_SIZE = 2 * np.pi
REF_TRAJ_IDS = range(15, 20)


@dataclass(frozen=True)
class CaseConfig:
    """Helper for case-specific parameters and patterns."""

    name: str
    dim: int
    run_pattern: str
    ref_pattern: str


CASE_CONFIGS = {
    "kolm2d": CaseConfig(
        name="kolm2d", dim=2, run_pattern="state_step_*.bin", ref_pattern="com/step_*.h5"
    ),
    "hit3d": CaseConfig(
        name="hit3d", dim=3, run_pattern="state_step_*.bin", ref_pattern="com/step_*.h5"
    ),
}


def _stats(arr: np.ndarray) -> dict[str, np.ndarray]:
    """Extract min/max/median along axis=0."""
    return {
        "min": np.min(arr, axis=0),
        "max": np.max(arr, axis=0),
        "med": np.median(arr, axis=0),
    }


def _step_from_filename(file):
    """e.g.: "step_00000.h5" or "u_64_00000.npy"""
    return int(file.stem.split("_")[-1])


def _step_from_spectrum_name(path: Path) -> int:
    """Simulation step from spectrum_*.csv file from SPH CUDA solver."""
    return int(path.stem.split("_")[1])


def _infer_n_per_dim(n_particles: int, dim: int) -> int:
    return int(round(n_particles ** (1.0 / dim)))


def _latest_file(traj: Path, pattern: str, key: Callable[[Path], int]) -> Path | None:
    """Return file path of last file according to the pattern and key."""
    files = sorted(traj.glob(pattern), key=key)
    return files[-1] if files else None


@lru_cache(maxsize=None)
def _get_interpolator(nx: int, dim: int) -> ParticleGridInterpolator:
    """Interpolator convenience wrapper."""
    return ParticleGridInterpolator(
        n_per_dim=(nx,) * dim,
        box_size=(BOX_SIZE,) * dim,
        dim=dim,
        nufft_splits=2,
        nufft_backend="auto",
    )


def compute_spectrum_from_particles(
    r: np.ndarray, u: np.ndarray, nx: int, dim: int
) -> pd.DataFrame:
    """Get spectrum after converting particles to given grid."""
    interpolator = _get_interpolator(nx, dim=dim)
    r_src = np.asarray(r[:, :dim])
    u_src = np.asarray(u[:, :dim])
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


def get_ref_ekin(
    ref_path: Path,
    burnin_steps: int,
    dim: int,
    ref_traj_ids=REF_TRAJ_IDS,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Extract Ekin from diagnostics.csv files."""
    ekin = []
    last_df = None
    for idx in ref_traj_ids:
        df = pd.read_csv(ref_path / f"traj_{idx}/diagnostics.csv")
        ekin.append(df["ekin"].to_numpy())
        last_df = df

    time = last_df["time"].to_numpy()[burnin_steps:] - last_df["time"].to_numpy()[burnin_steps]
    ekin = np.array(ekin)[:, burnin_steps:]
    ekin *= BOX_SIZE**dim  # during dataset generation, Ekin is not scaled by volume
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


def list_cached_spectra(trajs: list[Path]) -> list[Path | None]:
    return [_latest_file(traj, "spectrum_*.csv", _step_from_spectrum_name) for traj in trajs]


def _state_spectrum(file: Path, dim: int) -> pd.DataFrame:
    """Load .bin state and compute its spectrum."""
    state = load_state(file)
    return compute_spectrum_from_particles(
        r=state["x"], u=state["u"], nx=int(state["nx"]), dim=dim
    )


def _ref_spectrum(file: Path, dim: int) -> pd.DataFrame:
    """Load .h5 file and compute its spectrum."""
    with h5py.File(file, "r") as data:
        r = np.asarray(data["r"])
        u = np.asarray(data["u"])
    nx = _infer_n_per_dim(len(r), dim=dim)
    return compute_spectrum_from_particles(r=r, u=u, nx=nx, dim=dim)


def load_spectra_stats(
    spectrum_files: list[Path | None],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
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

    spectra_arr = np.array(spectra)
    return k, _stats(spectra_arr)


def plot_ekin(
    save_dir: Path, trajs: list[Path], ref_path: Path, dim: int, burnin_steps_dns: int
) -> None:
    """Plot Ekin min/max/median over the test trajs."""
    trajs_ekin = []
    time = None
    for traj in trajs:
        df = pd.read_csv(traj / "diagnostics.csv")
        trajs_ekin.append(df["ekin"].to_numpy())
        if time is None:
            time = df["time"].to_numpy()

    ekin_stats = _stats(np.array(trajs_ekin))
    ekin_ref, t_ref = get_ref_ekin(ref_path, burnin_steps=burnin_steps_dns, dim=dim)

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(t_ref, ekin_ref["med"], "k", label="Reference")
    ax.fill_between(t_ref, ekin_ref["min"], ekin_ref["max"], color="k", alpha=0.2)
    ax.plot(time, ekin_stats["med"], label="Run")
    ax.fill_between(time, ekin_stats["min"], ekin_stats["max"], alpha=0.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Kinetic energy")
    if dim == 3:
        ax.set_yscale("log")
    ax.legend()
    fig.savefig(save_dir / "ekin.png")
    plt.close(fig)


def compute_spectra(
    trajs: list[Path],
    dim: int,
    run_pattern: str,
    ref_path: Path,
    ref_pattern: str,
    ref_traj_ids=REF_TRAJ_IDS,
) -> None:
    """Compute and cache `spectrum_*.csv` for run and reference trajectories."""
    get_spectra(
        trajs,
        run_pattern,
        _step_from_name,
        lambda file: _state_spectrum(file, dim=dim),
        recompute=True,
    )
    ref_trajs = [ref_path / f"traj_{idx}" for idx in ref_traj_ids]
    get_spectra(
        ref_trajs,
        ref_pattern,
        _step_from_filename,
        lambda file: _ref_spectrum(file, dim=dim),
        recompute=True,
    )


def plot_spectra(
    save_dir: Path,
    trajs: list[Path],
    ref_path: Path,
    ref_traj_ids=REF_TRAJ_IDS,
) -> None:
    """Plot spectral min/max/median over the test trajs from cached spectrum CSV files."""
    run_spectra_files = list_cached_spectra(trajs)
    ref_trajs = [ref_path / f"traj_{idx}" for idx in ref_traj_ids]
    ref_spectra_files = list_cached_spectra(ref_trajs)

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
    parser = argparse.ArgumentParser(description="Analyse Kolm2d/HIT3d output directory")
    parser.add_argument("--path", type=Path, required=True, help="Path to a save directory")
    parser.add_argument("--case", type=str, choices=tuple(CASE_CONFIGS.keys()), required=True)
    parser.add_argument(
        "--ref-path", type=Path, required=True, help="Path to reference DNS dataset"
    )
    parser.add_argument(
        "--burnin-steps-dns", type=int, default=0, help="Burn-in steps for reference diagnostics"
    )
    parser.add_argument(
        "--recompute-spectra",
        action="store_true",
        help="Only recompute spectrum_*.csv for run and reference; skip plotting.",
    )
    args = parser.parse_args()

    case = CASE_CONFIGS[args.case]
    trajs = sorted(args.path.glob("traj_*/"))

    if args.recompute_spectra:
        compute_spectra(
            trajs=trajs,
            dim=case.dim,
            run_pattern=case.run_pattern,
            ref_path=args.ref_path,
            ref_pattern=case.ref_pattern,
        )
    plot_ekin(args.path, trajs, args.ref_path, case.dim, args.burnin_steps_dns)
    plot_spectra(args.path, trajs, args.ref_path)
