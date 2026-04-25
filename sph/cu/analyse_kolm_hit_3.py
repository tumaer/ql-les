"""Compare Ekin and spectra across resolutions for both base and withA datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analyse_kolm_hit_2 import (
    _available_run_dirs,
    _load_ekin_values,
    _load_raw_metric,
    _load_ref_ekin_values,
    _to_serializable,
)
from sph.cu.analyse_kolm_hit import (
    CASE_CONFIGS,
    REF_TRAJ_IDS,
    SPECTRUM_CSV_NAME,
    _list_cached_metric,
    stats,
)


def _run_dirs_by_resolution(data_root: Path, case_name: str) -> dict[int, Path]:
    """Map each available resolution to its run directory."""
    return {nx: run_dir for nx, run_dir in _available_run_dirs(data_root, case_name)}


def plot_ekin_and_spectra_merged(
    data_root: Path,
    case_name: str,
    ref_path: Path,
    burnin_steps_dns: int,
    ref_traj_ids=REF_TRAJ_IDS,
) -> None:
    """Plot Ekin and spectra for two datasets, sharing colors across matching resolutions."""
    case = CASE_CONFIGS[case_name]
    data_root_with_a = data_root / "withA"
    label_base = "SPH"
    label_with_a = r"SPH$_A$"

    run_dirs_base = _run_dirs_by_resolution(data_root, case_name)
    run_dirs_with_a = _run_dirs_by_resolution(data_root_with_a, case_name)
    resolutions = sorted(set(run_dirs_base.keys()) | set(run_dirs_with_a.keys()))

    if len(resolutions) == 0:
        raise FileNotFoundError(
            f"No run directories found for case '{case_name}' in {data_root} or {data_root_with_a}"
        )

    plot_data = {
        "ekin": {"ref": {}, "pred": {"base": {}, "with_a": {}}},
        "spectrum": {"ref": {}, "pred": {"base": {}, "with_a": {}}},
    }

    plt.rcParams.update({"font.size": 17})
    fig, axs = plt.subplots(1, 2, figsize=(8, 4.8), layout="constrained")
    ax_ekin, ax_spectrum = axs

    # DNS reference for Ekin
    ekin_ref_values, t_ref = _load_ref_ekin_values(
        ref_path=ref_path,
        burnin_steps=burnin_steps_dns,
        dim=case.dim,
        ref_traj_ids=ref_traj_ids,
    )
    ekin_ref = stats(ekin_ref_values)
    plot_data["ekin"]["ref"] = {"x": t_ref, "values": ekin_ref_values}
    ax_ekin.plot(t_ref, ekin_ref["med"], color="k", linestyle="-", linewidth=3.0, label="DNS")

    # DNS reference for spectrum
    ref_trajs = [ref_path / f"traj_{idx}" for idx in ref_traj_ids]
    ref_spectrum_files = _list_cached_metric(ref_trajs, SPECTRUM_CSV_NAME)
    k_ref, ref_values = _load_raw_metric(ref_spectrum_files, x_col="k", y_col="energy")
    ref_stats = stats(ref_values)
    plot_data["spectrum"]["ref"] = {"x": k_ref, "values": ref_values}
    ax_spectrum.plot(k_ref, ref_stats["med"], color="k", linestyle="-", linewidth=3.0, label="DNS")

    cmap = plt.get_cmap("tab10")
    resolution_colors = {nx: cmap(i % 10) for i, nx in enumerate(resolutions)}

    variant_specs = [
        ("base", run_dirs_base, "-", 1.0),
        ("with_a", run_dirs_with_a, "--", 0.75),
    ]

    for variant_key, run_dirs, line_style, line_alpha in variant_specs:
        for nx in resolutions:
            save_dir = run_dirs.get(nx)
            if save_dir is None:
                continue

            trajs = sorted(save_dir.glob("traj_*"))
            if len(trajs) == 0:
                continue

            color = resolution_colors[nx]

            # Ekin curves for this dataset and resolution
            ekin_values, time = _load_ekin_values(trajs)
            ekin_stats = stats(ekin_values)
            plot_data["ekin"]["pred"][variant_key][nx] = {"x": time, "values": ekin_values}
            ax_ekin.plot(
                time,
                ekin_stats["med"],
                color=color,
                linestyle=line_style,
                alpha=line_alpha,
                linewidth=2.0,
            )

            # Spectrum curves for this dataset and resolution
            spectrum_files = _list_cached_metric(trajs, SPECTRUM_CSV_NAME)
            k, spectra_values = _load_raw_metric(spectrum_files, x_col="k", y_col="energy")
            spectra_stats = stats(spectra_values)
            plot_data["spectrum"]["pred"][variant_key][nx] = {
                "x": k,
                "values": spectra_values,
            }

            n_common = min(len(k_ref), len(spectra_stats["med"]))
            ax_spectrum.plot(
                k_ref[:n_common],
                spectra_stats["med"][:n_common],
                color=color,
                linestyle=line_style,
                alpha=line_alpha,
                linewidth=2.0,
            )

    ax_ekin.set_xlabel("Time")
    ax_ekin.set_ylabel("Kinetic energy")
    if case.dim == 3:
        ax_ekin.set_yscale("log")
    ax_ekin.set_xlim(t_ref[0], t_ref[-1])

    ax_spectrum.set_xlabel("Wavenumber k")
    ax_spectrum.set_ylabel("Energy spectrum")
    ax_spectrum.set_xscale("log")
    ax_spectrum.set_yscale("log")
    ax_spectrum.set_xlim(0.8, k_ref[-1] * 1.1)

    if case_name == "kolm2d":
        ax_ekin.set_ylim(17, 57)

    resolution_handles = [
        Line2D(
            [0],
            [0],
            color=resolution_colors[nx],
            linestyle="-",
            linewidth=2.0,
            label=rf"SPH, {nx}$^{{{case.dim}}}$",
        )
        for nx in resolutions
    ]
    style_handles = [
        Line2D([0], [0], color="k", linestyle="-", linewidth=2.0, label=label_base),
        Line2D([0], [0], color="k", linestyle="--", linewidth=2.0, alpha=0.75, label=label_with_a),
        Line2D([0], [0], color="k", linestyle="-", linewidth=3.0, label="DNS"),
    ]

    kwargs = dict(frameon=False, loc="lower left", fontsize=15, title_fontsize=15)
    ax_ekin.legend(handles=resolution_handles, title="Resolution", **kwargs)
    ax_spectrum.legend(handles=style_handles, title="Line style", **kwargs)

    fig.savefig(data_root / f"{case_name}_comparison_merged.png", dpi=300)
    fig.savefig(data_root / f"{case_name}_comparison_merged.pdf", dpi=300)
    plt.close(fig)

    serializable = _to_serializable(plot_data)
    serializable = {
        "case": case_name,
        "dim": case.dim,
        "paths": {"base": str(data_root), "with_a": str(data_root_with_a)},
    } | serializable

    out_json = data_root / f"{case_name}_comparison_merged_data.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as file:
        json.dump(serializable, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Kolm2d/HIT3d runs across resolutions for base and withA datasets"
    )
    parser.add_argument("--path", type=Path, default=Path("res"), help="Root with <case>_<nx>/")
    parser.add_argument(
        "--case",
        choices=tuple(CASE_CONFIGS.keys()),
        required=True,
    )
    parser.add_argument("--ref-path", type=Path, required=True, help="Path to ref DNS dataset")
    parser.add_argument("--burnin-steps-dns", type=int, default=0)

    args = parser.parse_args()

    plot_ekin_and_spectra_merged(
        data_root=args.path,
        case_name=args.case,
        ref_path=args.ref_path,
        burnin_steps_dns=args.burnin_steps_dns,
    )
