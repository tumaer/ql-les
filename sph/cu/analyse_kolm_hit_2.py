from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import _step_from_name

from sph.cu.analyse_kolm_hit import (
    CASE_CONFIGS,
    REF_TRAJ_IDS,
    _ref_spectrum,
    _step_from_filename,
    _state_spectrum,
    _stats,
    get_ref_ekin,
    get_spectra,
    list_cached_spectra,
    load_spectra_stats,
)


def _available_run_dirs(data_root: Path, case_name: str) -> list[tuple[int, Path]]:
    run_dirs = []
    for save_dir in data_root.glob(f"{case_name}_*"):
        if (not save_dir.is_dir()) or save_dir.name.endswith("_init"):
            continue
        print(f"{save_dir=}")
        nx = int(save_dir.name.split("_")[-1])
        run_dirs.append((nx, save_dir))
    return sorted(run_dirs, key=lambda pair: pair[0])


def _load_ekin_stats(trajs: list[Path]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load Ekin stats for a list of trajectories."""
    trajs_ekin = []
    time = None
    for traj in trajs:
        df = pd.read_csv(traj / "diagnostics.csv")
        trajs_ekin.append(df["ekin"].to_numpy())
        print(f"{trajs_ekin[-1].shape=}")
        if time is None:
            time = df["time"].to_numpy()

    return _stats(np.array(trajs_ekin)), time


def compute_spectra_for_all(
    data_root: Path,
    case_name: str,
    ref_path: Path,
    ref_traj_ids=REF_TRAJ_IDS,
) -> None:
    case = CASE_CONFIGS[case_name]
    run_dirs = _available_run_dirs(data_root, case_name)

    ref_trajs = [ref_path / f"traj_{idx}" for idx in ref_traj_ids]
    get_spectra(
        ref_trajs,
        case.ref_pattern,
        _step_from_filename,
        lambda file: _ref_spectrum(file, dim=case.dim),
        recompute=True,
    )

    for _, save_dir in run_dirs:
        trajs = sorted(save_dir.glob("traj_*"))
        get_spectra(
            trajs,
            case.run_pattern,
            _step_from_name,
            lambda file: _state_spectrum(file, dim=case.dim),
            recompute=True,
        )


def plot_ekin_and_spectra(
    data_root: Path,
    case_name: str,
    ref_path: Path,
    burnin_steps_dns: int,
    ref_traj_ids=REF_TRAJ_IDS,
) -> None:
    """Plot Ekin and spectra across resolutions, comparing to DNS reference."""
    case = CASE_CONFIGS[case_name]
    run_dirs = _available_run_dirs(data_root, case_name)

    plot_data = {"ekin": {"ref": {}, "pred": {}}, "spectrum": {"ref": {}, "pred": {}}}

    plt.rcParams.update({"font.size": 13})
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    ax_ekin, ax_spectrum = axs

    # Reference Ekin
    ekin_ref, t_ref = get_ref_ekin(
        ref_path=ref_path,
        burnin_steps=burnin_steps_dns,
        dim=case.dim,
        ref_traj_ids=ref_traj_ids,
    )
    plot_data["ekin"]["ref"] = {
        "x": t_ref,
        "min": ekin_ref["min"],
        "med": ekin_ref["med"],
        "max": ekin_ref["max"],
    }
    ax_ekin.plot(t_ref, ekin_ref["med"], "k", label="DNS")
    ax_ekin.fill_between(t_ref, ekin_ref["min"], ekin_ref["max"], color="k", alpha=0.2)

    # Reference spectrum
    ref_trajs = [ref_path / f"traj_{idx}" for idx in ref_traj_ids]
    k_ref, ref_stats = load_spectra_stats(list_cached_spectra(ref_trajs))
    plot_data["spectrum"]["ref"] = {
        "x": k_ref,
        "min": ref_stats["min"],
        "med": ref_stats["med"],
        "max": ref_stats["max"],
    }
    ax_spectrum.plot(k_ref, ref_stats["med"], "k", label="DNS")
    ax_spectrum.fill_between(k_ref, ref_stats["min"], ref_stats["max"], color="k", alpha=0.2)

    for nx, save_dir in run_dirs:
        trajs = sorted(save_dir.glob("traj_*"))

        # Plot Ekin
        ekin_stats, time = _load_ekin_stats(trajs)
        plot_data["ekin"]["pred"][nx] = {
            "x": time,
            "min": ekin_stats["min"],
            "med": ekin_stats["med"],
            "max": ekin_stats["max"],
        }
        ax_ekin.plot(time, ekin_stats["med"], label=rf"SPH, {nx}$^{case.dim}$")
        ax_ekin.fill_between(time, ekin_stats["min"], ekin_stats["max"], alpha=0.2)

        # Plot spectrum
        spectrum_files = list_cached_spectra(trajs)
        k, spectra_stats = load_spectra_stats(spectrum_files)
        plot_data["spectrum"]["pred"][nx] = {
            "x": k_ref,
            "min": spectra_stats["min"],
            "med": spectra_stats["med"],
            "max": spectra_stats["max"],
        }
        ax_spectrum.plot(
            k_ref, spectra_stats["med"][: len(k_ref)], label=rf"SPH, {nx}$^{case.dim}$"
        )
        ax_spectrum.fill_between(
            k_ref,
            spectra_stats["min"][: len(k_ref)],
            spectra_stats["max"][: len(k_ref)],
            alpha=0.2,
        )

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

    plot_data = jax.tree.map(lambda arr: arr.tolist(), plot_data)
    plot_data = {"case": case_name, "dim": case.dim} | plot_data
    file_path = data_root / f"{case_name}_comparison_data.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w") as file:
        json.dump(plot_data, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Kolm2d/HIT3d runs across resolutions")
    parser.add_argument("--path", type=Path, default=Path("res"), help="Root with <case>_<nx>/")
    parser.add_argument("--case", choices=tuple(CASE_CONFIGS.keys()), required=True)
    parser.add_argument("--ref-path", type=Path, required=True, help="Path to ref DNS dataset")
    parser.add_argument("--burnin-steps-dns", type=int, default=0)
    parser.add_argument(
        "--recompute-spectra",
        action="store_true",
        help="Only recompute spectrum_*.csv for all runs/reference; skip plotting.",
    )
    args = parser.parse_args()

    if args.recompute_spectra:
        compute_spectra_for_all(args.path, args.case, args.ref_path)
    plot_ekin_and_spectra(args.path, args.case, args.ref_path, args.burnin_steps_dns)
