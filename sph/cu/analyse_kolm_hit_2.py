from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import _step_from_name

from sph.cu.analyse_kolm_hit import (
    CASE_CONFIGS,
    CORR_CSV_NAME,
    REF_TRAJ_IDS,
    SPECTRUM_CSV_NAME,
    _latest_file,
    _list_cached_metric,
    _ref_spectrum,
    _step_from_filename,
    _state_spectrum,
    infer_n_per_dim,
    stats,
)


def _available_run_dirs(data_root: Path, case_name: str) -> list[tuple[int, Path]]:
    """List available run directories for a given case, returning sorted list of (nx, path)."""
    run_dirs = []
    for save_dir in data_root.glob(f"{case_name}_*"):
        if (not save_dir.is_dir()) or save_dir.name.endswith("_init"):
            continue
        print(f"{save_dir=}")
        nx = int(save_dir.name.split("_")[-1])
        run_dirs.append((nx, save_dir))
    return sorted(run_dirs, key=lambda pair: pair[0])


def _load_ekin_values(trajs: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Load raw Ekin values for a list of trajectories."""
    trajs_ekin = []
    time = None
    for traj in trajs:
        df = pd.read_csv(traj / "diagnostics.csv")
        trajs_ekin.append(df["ekin"].to_numpy())
        print(f"{trajs_ekin[-1].shape=}")
        if time is None:
            time = df["time"].to_numpy()

    return np.array(trajs_ekin), time


def _load_ref_ekin_values(
    ref_path: Path,
    burnin_steps: int,
    dim: int,
    ref_traj_ids=REF_TRAJ_IDS,
) -> tuple[np.ndarray, np.ndarray]:
    """Load raw reference Ekin values with the same scaling as get_ref_ekin."""
    values = []
    last_df = None
    for idx in ref_traj_ids:
        df = pd.read_csv(ref_path / f"traj_{idx}/diagnostics.csv")
        values.append(df["ekin"].to_numpy())
        last_df = df
    time = last_df["time"].to_numpy()[burnin_steps:] - last_df["time"].to_numpy()[burnin_steps]
    arr = np.array(values)[:, burnin_steps:]
    arr *= (2 * np.pi) ** dim
    return arr, time


def _load_raw_metric(
    files: list[Path | None], x_col: str, y_col: str
) -> tuple[np.ndarray, np.ndarray]:
    """Load raw per-trajectory metric values with a shared x-axis."""
    raw_values = []
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
        raw_values.append(y_i)
    if len(raw_values) == 0 or x_axis is None:
        raise FileNotFoundError(f"Missing metric files with columns {x_col}/{y_col}")
    return x_axis, np.array(raw_values)


def _to_serializable(value):
    """Recursively convert a value to a JSON-serializable form (e.g. np.ndarray to list)."""
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def compute_spectra_for_all(
    data_root: Path,
    case_name: str,
    ref_path: Path,
    ref_traj_ids=REF_TRAJ_IDS,
) -> None:
    """Compute spectra for all trajectories and save as CSV in each traj directory."""
    case = CASE_CONFIGS[case_name]
    run_dirs = _available_run_dirs(data_root, case_name)

    ref_trajs = [ref_path / f"traj_{idx}" for idx in ref_traj_ids]

    for _, save_dir in run_dirs:
        trajs = sorted(save_dir.glob("traj_*"))
        n_pairs = min(len(trajs), len(ref_trajs))
        for run_traj, ref_traj in zip(trajs[:n_pairs], ref_trajs[:n_pairs]):
            run_file = _latest_file(run_traj, case.run_pattern, _step_from_name)
            ref_file = _latest_file(ref_traj, case.ref_pattern, _step_from_filename)
            if run_file is None or ref_file is None:
                continue
            with h5py.File(ref_file, "r") as data:
                nx_ref = infer_n_per_dim(len(data["r"]), dim=case.dim)

            _state_spectrum(run_file, dim=case.dim, nx=nx_ref).to_csv(
                run_traj / SPECTRUM_CSV_NAME, index=False
            )
            _ref_spectrum(ref_file, dim=case.dim, nx=nx_ref).to_csv(
                ref_traj / SPECTRUM_CSV_NAME, index=False
            )


def plot_ekin_and_spectra(
    data_root: Path,
    case_name: str,
    ref_path: Path,
    burnin_steps_dns: int,
    ref_traj_ids=REF_TRAJ_IDS,
) -> None:
    """Plot velocity correlation, Ekin, and spectra across resolutions, comparing to DNS reference."""
    case = CASE_CONFIGS[case_name]
    run_dirs = _available_run_dirs(data_root, case_name)

    plot_data = {
        "corr": {"ref": {}, "pred": {}},
        "ekin": {"ref": {}, "pred": {}},
        "spectrum": {"ref": {}, "pred": {}},
    }

    plt.rcParams.update({"font.size": 13})
    fig, axs = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    ax_corr, ax_ekin, ax_spectrum = axs

    # Reference velocity correlation (raw for JSON; stats for plotting)
    ref_trajs = [ref_path / f"traj_{idx}" for idx in ref_traj_ids]

    # Reference Ekin (raw for JSON; stats for plotting)
    ekin_ref_values, t_ref = _load_ref_ekin_values(
        ref_path=ref_path, burnin_steps=burnin_steps_dns, dim=case.dim, ref_traj_ids=ref_traj_ids
    )
    ekin_ref = stats(ekin_ref_values)
    plot_data["ekin"]["ref"] = {"x": t_ref, "values": ekin_ref_values}
    ax_ekin.plot(t_ref, ekin_ref["med"], "k", label="DNS")
    ax_ekin.fill_between(t_ref, ekin_ref["min"], ekin_ref["max"], color="k", alpha=0.2)

    # Reference spectrum (raw for JSON; stats for plotting)
    ref_spectrum_files = _list_cached_metric(ref_trajs, SPECTRUM_CSV_NAME)
    k_ref, ref_values = _load_raw_metric(ref_spectrum_files, x_col="k", y_col="energy")
    ref_stats = stats(ref_values)
    plot_data["spectrum"]["ref"] = {"x": k_ref, "values": ref_values}
    ax_spectrum.plot(k_ref, ref_stats["med"], "k", label="DNS")
    ax_spectrum.fill_between(k_ref, ref_stats["min"], ref_stats["max"], color="k", alpha=0.2)

    for nx, save_dir in run_dirs:
        trajs = sorted(save_dir.glob("traj_*"))

        # Plot velocity correlation (raw for JSON; stats for plotting)
        corr_files = _list_cached_metric(trajs, CORR_CSV_NAME)
        t_corr, corr_values = _load_raw_metric(corr_files, x_col="time", y_col="corr")
        # Shift so that t=0 value is 1 on average. Diff due to particle interpolation
        # corr_values = corr_values + 1 - corr_values[:, 0].mean()
        corr_stats = stats(corr_values)
        plot_data["corr"]["pred"][nx] = {"x": t_corr, "values": corr_values}
        ax_corr.plot(t_corr, corr_stats["med"], label=rf"SPH, {nx}$^{case.dim}$")
        ax_corr.fill_between(t_corr, corr_stats["min"], corr_stats["max"], alpha=0.2)

        # Plot Ekin
        ekin_values, time = _load_ekin_values(trajs)
        ekin_stats = stats(ekin_values)
        plot_data["ekin"]["pred"][nx] = {"x": time, "values": ekin_values}
        ax_ekin.plot(time, ekin_stats["med"], label=rf"SPH, {nx}$^{case.dim}$")
        ax_ekin.fill_between(time, ekin_stats["min"], ekin_stats["max"], alpha=0.2)

        # Plot spectrum (raw for JSON; stats for plotting)
        spectrum_files = _list_cached_metric(trajs, SPECTRUM_CSV_NAME)
        k, spectra_values = _load_raw_metric(spectrum_files, x_col="k", y_col="energy")
        spectra_stats = stats(spectra_values)
        plot_data["spectrum"]["pred"][nx] = {"x": k, "values": spectra_values}
        ax_spectrum.plot(
            k_ref, spectra_stats["med"][: len(k_ref)], label=rf"SPH, {nx}$^{case.dim}$"
        )
        ax_spectrum.fill_between(
            k_ref,
            spectra_stats["min"][: len(k_ref)],
            spectra_stats["max"][: len(k_ref)],
            alpha=0.2,
        )

    ax_corr.set_xlabel("Time")
    ax_corr.set_ylabel("Velocity correlation")
    ax_corr.set_xlim(time[0], time[-1])
    ax_corr.set_ylim(-0.02, 1.02)

    ax_ekin.set_xlabel("Time")
    ax_ekin.set_ylabel("Kinetic energy")
    if case.dim == 3:
        ax_ekin.set_yscale("log")
    ax_ekin.set_xlim(time[0], time[-1])

    ax_spectrum.set_xlabel("Wavenumber k")
    ax_spectrum.set_ylabel("Energy spectrum")
    ax_spectrum.set_xscale("log")
    ax_spectrum.set_yscale("log")
    ax_spectrum.set_xlim(0.8, k_ref[-1] * 1.1)
    ax_spectrum.legend()

    for ax in axs:
        ax.grid()

    fig.savefig(data_root / f"{case_name}_comparison.png", dpi=200)
    plt.close(fig)

    plot_data = _to_serializable(plot_data)
    plot_data = {"case": case_name, "dim": case.dim} | plot_data
    file_path = data_root / f"{case_name}_comparison_data.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w") as file:
        json.dump(plot_data, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Kolm2d/HIT3d runs across resolutions")
    parser.add_argument("--path", type=Path, default=Path("res"), help="Root with <case>_<nx>/")
    parser.add_argument("--case", choices=tuple(CASE_CONFIGS.keys()), required=True)
    parser.add_argument("--ref-path", type=Path, required=True, help="Path to ref DNS dataset")
    parser.add_argument("--burnin-steps-dns", type=int, default=0)
    args = parser.parse_args()

    plot_ekin_and_spectra(args.path, args.case, args.ref_path, args.burnin_steps_dns)
