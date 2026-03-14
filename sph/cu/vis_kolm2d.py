from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import _step_from_name

from analyse_kolm2d import (
    REF_PATH,
    REF_TRAJ_IDS,
    _ref_spectrum,
    _state_spectrum,
    _stats,
    _step_from_h5_name,
    get_ref_ekin,
    get_spectra,
    load_spectra_stats,
)

RESOLUTIONS = [64, 128, 256, 512]


def _load_ekin_stats(trajs: list[Path]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load Ekin stats for a list of trajectories."""
    trajs_ekin = []
    time = None
    for traj in trajs:
        df = pd.read_csv(traj / "diagnostics.csv")
        trajs_ekin.append(df["ekin"].to_numpy())
        if time is None:
            time = df["time"].to_numpy()

    if not trajs_ekin or time is None:
        raise FileNotFoundError("No diagnostics found for selected trajectories")

    return _stats(np.array(trajs_ekin)), time


def plot_ekin_and_spectra(data_root: Path, recompute: bool = False) -> None:
    """Plot Ekin and spectra for Kolm2d runs across resolutions, comparing to reference DNS."""
    plt.rcParams.update({"font.size": 13})
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    ax_ekin, ax_spectrum = axs

    # Reference Ekin
    ekin_ref, t_ref = get_ref_ekin()
    ax_ekin.plot(t_ref, ekin_ref["med"], "k", label="DNS")
    ax_ekin.fill_between(t_ref, ekin_ref["min"], ekin_ref["max"], color="k", alpha=0.2)

    # Reference spectrum
    ref_trajs = [REF_PATH / f"traj_{idx}" for idx in REF_TRAJ_IDS]
    ref_spectra_files = get_spectra(
        ref_trajs,
        "com/step_*.h5",
        _step_from_h5_name,
        _ref_spectrum,
        recompute=recompute,
    )
    k_ref, ref_stats = load_spectra_stats(ref_spectra_files)
    ax_spectrum.plot(k_ref, ref_stats["med"], "k", label="DNS")
    ax_spectrum.fill_between(k_ref, ref_stats["min"], ref_stats["max"], color="k", alpha=0.2)

    for nx in RESOLUTIONS:
        save_dir = data_root / f"kolm2d_{nx}"
        trajs = sorted(save_dir.glob("traj_*"))
        if not trajs:
            continue

        # Plot Ekin
        ekin_stats, time = _load_ekin_stats(trajs)
        ax_ekin.plot(time, ekin_stats["med"], label=rf"SPH, {nx}$^2$")
        ax_ekin.fill_between(time, ekin_stats["min"], ekin_stats["max"], alpha=0.2)

        # Plot spectrum
        spectrum_files = get_spectra(
            trajs,
            "state_step_*.bin",
            _step_from_name,
            _state_spectrum,
            recompute=recompute,
        )
        k, spectra_stats = load_spectra_stats(spectrum_files)
        ax_spectrum.plot(k_ref, spectra_stats["med"][: len(k_ref)], label=rf"SPH, {nx}$^2$")
        ax_spectrum.fill_between(
            k_ref,
            spectra_stats["min"][: len(k_ref)],
            spectra_stats["max"][: len(k_ref)],
            alpha=0.2,
        )

    ax_ekin.set_xlabel("Time")
    ax_ekin.set_ylabel("Kinetic energy")
    ax_ekin.set_xlim(time[0], time[-1])

    ax_spectrum.set_xlabel("Wavenumber k")
    ax_spectrum.set_ylabel("Energy spectrum")
    ax_spectrum.set_xscale("log")
    ax_spectrum.set_yscale("log")
    ax_spectrum.set_xlim(0.8, k_ref[-1] * 1.1)
    ax_spectrum.legend()
    for ax in axs:
        # ax.grid(True, which="both", ls="--", lw=0.5)
        ax.grid()

    fig.savefig(data_root / "kolm2d_comparison.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Kolm2d runs across resolutions")
    parser.add_argument("--path", type=Path, default=Path("res"), help="Root with kolm2d_<nx>/")
    parser.add_argument("--recompute-spectra", action="store_true", help="Compute spectrum_*.csv")
    args = parser.parse_args()

    plot_ekin_and_spectra(args.path, recompute=args.recompute_spectra)
