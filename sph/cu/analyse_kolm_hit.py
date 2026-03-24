"""Utilities to analyse Kolm2d and HIT3d runs against DNS references."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from utils import _step_from_name, load_state
from src.utils.interpolate import GridInterpolator
from src.utils.jax_utils.jax_spectral import energy_spectrum

BOX_SIZE = 2 * np.pi
REF_TRAJ_IDS = range(15, 20)
SPECTRUM_CSV_NAME = "spectrum.csv"
CORR_CSV_NAME = "corr.csv"


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


def _infer_n_per_dim(n_particles: int, dim: int) -> int:
    """Infer n_per_dim for a given number of particles and dimension, assuming a cubic grid."""
    return int(round(n_particles ** (1.0 / dim)))


def _latest_file(traj: Path, pattern: str, key: Callable[[Path], int]) -> Path | None:
    """Return file path of last file according to the pattern and key."""
    files = sorted(traj.glob(pattern), key=key)
    return files[-1] if files else None


_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_corr_interpolator_cache = {}


def _get_interpolator(nx: int, dim: int) -> GridInterpolator:
    """Interpolator wrapper with GPU support."""
    key = (nx, dim)
    if key not in _corr_interpolator_cache:
        dx = BOX_SIZE / nx
        domain_size = (BOX_SIZE,) * dim
        interp = GridInterpolator(
            is_periodic=True,
            domain_size=domain_size,
            dim=dim,
            dx=dx,
            condition="radius",
            cutoff_factor=2,
            kernel="quintic",
            mls_order=2,
        ).to(_device)
        _corr_interpolator_cache[key] = interp
    return _corr_interpolator_cache[key]


def _interpolate_velocity_to_grid_mls2(
    r: np.ndarray,
    u: np.ndarray,
    nx: int,
    dim: int,
) -> np.ndarray:
    """Interpolate particle velocity field to Cartesian grid using torch MLS 2nd-order."""
    interpolator = _get_interpolator(nx, dim=dim)
    device = interpolator.grid.device

    # Convert numpy to torch on device
    r_src = torch.tensor(r[:, :dim], dtype=torch.float32, device=device)
    u_src = torch.tensor(u[:, :dim], dtype=torch.float32, device=device)

    # Interpolate to grid using quintic kernel on GPU
    with torch.no_grad():
        u_grid = interpolator(r=r_src, f=u_src)

    # Reshape result from (grid_size, dim) to (dim, *n_per_dim)
    n_per_dim = [nx] * dim
    u_grid_np = u_grid.cpu().numpy()
    u_grid_shaped = np.zeros((dim, *n_per_dim), dtype=np.float32)
    for d in range(dim):
        u_grid_shaped[d] = u_grid_np[:, d].reshape(n_per_dim)
    return u_grid_shaped


def _pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation for flattened fields."""
    a_flat = np.asarray(a).reshape(-1)
    b_flat = np.asarray(b).reshape(-1)
    a_centered = a_flat - np.mean(a_flat)
    b_centered = b_flat - np.mean(b_flat)
    denom = np.linalg.norm(a_centered) * np.linalg.norm(b_centered)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a_centered, b_centered) / denom)


def compute_spectrum_from_particles(
    r: np.ndarray, u: np.ndarray, nx: int, dim: int
) -> pd.DataFrame:
    """Get spectrum after converting particles to given grid."""
    start = time.time()
    u_grid = _interpolate_velocity_to_grid_mls2(r=r, u=u, nx=nx, dim=dim)
    print("P2G interpolation t=", time.time() - start)
    spectrum_full = np.asarray(energy_spectrum(u_grid))
    if not (nx & (nx - 1) == 0):  # If nx is not a power of two divide by 1.5
        nx = round(nx / 1.5)  # divide by 1.5 to undo the better spectrum trick
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


def _list_cached_metric(trajs: list[Path], csv_name: str) -> list[Path | None]:
    """List paths to cached metric CSV files for a list of trajectories, or None if missing."""
    return [traj / csv_name if (traj / csv_name).exists() else None for traj in trajs]


def _state_spectrum(file: Path, dim: int, nx: int) -> pd.DataFrame:
    """Load .bin state and compute its spectrum."""
    state = load_state(file)
    return compute_spectrum_from_particles(r=state["x"], u=state["u"], nx=nx, dim=dim)


def _ref_spectrum(file: Path, dim: int, nx: int) -> pd.DataFrame:
    """Load .h5 file and compute its spectrum."""
    with h5py.File(file, "r") as data:
        r = np.asarray(data["r"])
        u = np.asarray(data["u"])
    return compute_spectrum_from_particles(r=r, u=u, nx=nx, dim=dim)


def _load_metric_stats(
    files: list[Path | None],
    x_col: str,
    y_col: str,
    missing_msg: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values = []
    x_axis = None
    for file in files:
        if file is None:
            continue
        df = pd.read_csv(file)
        x_i = df[x_col].to_numpy()
        y_i = df[y_col].to_numpy()
        if x_axis is None:
            x_axis = x_i
        elif not np.array_equal(x_axis, x_i):
            raise ValueError(f"Incompatible {x_col} axis in {file}")
        values.append(y_i)

    if len(values) == 0 or x_axis is None:
        raise FileNotFoundError(missing_msg)

    values_arr = np.array(values)
    return x_axis, _stats(values_arr)


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
    """Compute and cache per-trajectory spectrum CSV for run and reference trajectories."""
    ref_trajs = [ref_path / f"traj_{idx}" for idx in ref_traj_ids]
    n_pairs = min(len(trajs), len(ref_trajs))
    for run_traj, ref_traj in zip(trajs[:n_pairs], ref_trajs[:n_pairs]):
        run_file = _latest_file(run_traj, run_pattern, _step_from_name)
        ref_file = _latest_file(ref_traj, ref_pattern, _step_from_filename)
        if run_file is None or ref_file is None:
            continue

        with h5py.File(ref_file, "r") as data:
            nx_ref = _infer_n_per_dim(len(data["r"]), dim=dim)

        # Improve high wavenumber spectrum without using the full nx_run
        nx_run = _infer_n_per_dim(len(load_state(run_file)["x"]), dim=dim)
        nx = max(nx_ref, min(round(nx_ref * 1.5), nx_run))

        run_spectrum_path = run_traj / SPECTRUM_CSV_NAME
        _state_spectrum(run_file, dim=dim, nx=nx).to_csv(run_spectrum_path, index=False)

        ref_spectrum_path = ref_traj / SPECTRUM_CSV_NAME
        _ref_spectrum(ref_file, dim=dim, nx=nx).to_csv(ref_spectrum_path, index=False)


def plot_spectra(
    save_dir: Path,
    trajs: list[Path],
    ref_path: Path,
    ref_traj_ids=REF_TRAJ_IDS,
) -> None:
    """Plot spectral min/max/median over the test trajs from cached spectrum CSV files."""
    run_spectra_files = _list_cached_metric(trajs, SPECTRUM_CSV_NAME)
    ref_trajs = [ref_path / f"traj_{idx}" for idx in ref_traj_ids]
    ref_spectra_files = _list_cached_metric(ref_trajs, SPECTRUM_CSV_NAME)

    k_ref, ref_stats = _load_metric_stats(
        ref_spectra_files,
        x_col="k",
        y_col="energy",
        missing_msg="Missing trajectory spectrum.csv files. Run with --recompute-spectra before plotting.",
    )
    k_run, run_stats = _load_metric_stats(
        run_spectra_files,
        x_col="k",
        y_col="energy",
        missing_msg="Missing trajectory spectrum.csv files. Run with --recompute-spectra before plotting.",
    )

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(k_ref, ref_stats["med"], "k", label="Reference")
    ax.fill_between(k_ref, ref_stats["min"], ref_stats["max"], color="k", alpha=0.2)
    ax.plot(k_run, run_stats["med"], label="Run")
    ax.fill_between(k_run, run_stats["min"], run_stats["max"], alpha=0.2)
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("Energy spectrum")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid()
    ax.legend()
    fig.savefig(save_dir / "spectrum.png")
    plt.close(fig)


def compute_velocity_corr_csv(
    burnin_steps_dns: int,
    trajs: list[Path],
    ref_path: Path,
    dim: int,
    run_pattern: str,
    ref_pattern: str,
    n_frames: int,
    ref_traj_ids=REF_TRAJ_IDS,
) -> list[Path]:
    """Compute Pearson correlation and cache per-trajectory CSV files."""
    ref_trajs = [ref_path / f"traj_{idx}" for idx in ref_traj_ids]
    n_pairs = min(len(trajs), len(ref_trajs))
    if n_pairs == 0:
        print("No run/reference trajectory pairs available for velocity correlation.")
        return []

    paired_data: list[tuple[Path, list[Path], list[Path], np.ndarray]] = []
    min_available_frames = None
    for run_traj, ref_traj in zip(trajs[:n_pairs], ref_trajs[:n_pairs]):
        run_files = sorted(run_traj.glob(run_pattern), key=_step_from_name)
        ref_files = sorted(ref_traj.glob(ref_pattern), key=_step_from_filename)

        ref_diag = pd.read_csv(ref_traj / "diagnostics.csv")
        ref_time = ref_diag["time"].to_numpy()
        time_axis = ref_time[burnin_steps_dns:]

        paired_data.append((run_traj, run_files, ref_files, time_axis))
        available_frames = min(len(run_files), len(ref_files))
        if min_available_frames is None:
            min_available_frames = available_frames
        else:
            min_available_frames = min(min_available_frames, available_frames)

    if not paired_data or min_available_frames is None:
        print("No run/reference trajectory files available after burn-in for correlation.")
        return []

    n_frames = max(1, min(n_frames, min_available_frames))
    out_files: list[Path] = []

    for run_traj, run_files, ref_files, time_axis in paired_data:
        run_indices = np.linspace(0, len(run_files) - 1, num=n_frames, dtype=int)
        ref_indices = np.linspace(0, len(ref_files) - 1, num=n_frames, dtype=int)
        time_indices = np.linspace(0, len(time_axis) - 1, num=n_frames, dtype=int)
        sampled_times = time_axis[time_indices] - time_axis[0]

        corr_values = []
        for run_idx, ref_idx in zip(run_indices, ref_indices):
            run_state = load_state(run_files[run_idx])
            with h5py.File(ref_files[ref_idx], "r") as data:
                r_ref = np.asarray(data["r"])
                u_ref = np.asarray(data["u"])

            nx_ref = _infer_n_per_dim(len(r_ref), dim=dim)
            nx_run = _infer_n_per_dim(len(run_state["x"]), dim=dim)

            t1 = time.time()
            u_run_grid = _interpolate_velocity_to_grid_mls2(
                r=(run_state["x"] - BOX_SIZE / (nx_run * 2))
                % BOX_SIZE,  # Shift run grid to align with reference
                u=run_state["u"],
                nx=nx_ref,
                dim=dim,
            )
            t2 = time.time()
            u_ref_grid = _interpolate_velocity_to_grid_mls2(
                r=r_ref,
                u=u_ref,
                nx=nx_ref,
                dim=dim,
            )
            t3 = time.time()
            corr = _pearson_corr(u_run_grid, u_ref_grid)
            print(
                f"P2G interp t_run={t2 - t1:.2f}s, t_ref={t3 - t2:.2f}s, corr={corr:.4f} "
                f"for run {run_files[run_idx].name} vs ref {ref_files[ref_idx].name}"
            )
            corr_values.append(corr)

        corr_df = pd.DataFrame(
            {
                "time": np.asarray(sampled_times, dtype=float),
                "corr": np.asarray(corr_values, dtype=float),
            }
        )
        corr_path = run_traj / CORR_CSV_NAME
        corr_df.to_csv(corr_path, index=False)
        out_files.append(corr_path)

    if len(out_files) == 0:
        print("Velocity correlation could not be computed for any trajectory pair.")
        return []

    return out_files


def plot_velocity_corr(save_dir: Path, trajs: list[Path]) -> None:
    """Plot Pearson correlation statistics from cached per-trajectory CSV files."""
    t_corr, corr_stats = _load_metric_stats(
        _list_cached_metric(trajs, CORR_CSV_NAME),
        x_col="time",
        y_col="corr",
        missing_msg="Missing trajectory corr.csv files. Run with --recompute-corr before plotting.",
    )

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(t_corr, corr_stats["med"], label="Run vs Reference")
    ax.fill_between(t_corr, corr_stats["min"], corr_stats["max"], alpha=0.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Pearson correlation")
    ax.set_ylim(-0.02, 1.02)
    ax.grid()
    ax.legend()
    fig.savefig(save_dir / "velocity_corr.png")
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
        help="Only recompute trajectory spectrum.csv for run and reference; skip plotting.",
    )
    parser.add_argument(
        "--corr-frames",
        type=int,
        default=26,
        help="Number of equidistant frames used for velocity-field correlation.",
    )
    parser.add_argument(
        "--recompute-corr",
        action="store_true",
        help="Recompute and cache velocity correlation CSV; run before plotting correlation.",
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
        print("Finished recomputing spectra.")

    if args.recompute_corr:
        compute_velocity_corr_csv(
            burnin_steps_dns=args.burnin_steps_dns,
            trajs=trajs,
            ref_path=args.ref_path,
            dim=case.dim,
            run_pattern=case.run_pattern,
            ref_pattern=case.ref_pattern,
            n_frames=args.corr_frames,
        )
        print("Finished recomputing corr.")

    plot_ekin(args.path, trajs, args.ref_path, case.dim, args.burnin_steps_dns)
    try:
        plot_spectra(args.path, trajs, args.ref_path)
    except FileNotFoundError as err:
        print(f"Skipping spectrum plot: {err}")
    try:
        plot_velocity_corr(save_dir=args.path, trajs=trajs)
    except FileNotFoundError as err:
        print(f"Skipping correlation plot: {err}")
