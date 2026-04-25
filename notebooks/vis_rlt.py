import argparse
import json
import struct

import io
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from collections import defaultdict
from pathlib import Path

from src.utils.data_utils import load_metadata
from src.utils.visualize import (
    infer_n_per_dim,
    interpolate_velocity_to_grid_mls2,
    spectrum_from_grid,
    stats,
    ZeroNeighborsInterpolationError,
)
from matplotlib.colors import Normalize

from matplotlib.colors import ListedColormap
import matplotlib as mpl


def turbo_darkmid(strength=0.35, width=0.18, n=256):
    base = mpl.colormaps["turbo"]
    x = np.linspace(0, 1, n)
    colors = base(x)

    # Gaussian dip centered at 0.5
    dip = np.exp(-0.5 * ((x - 0.5) / width) ** 2)

    # Darken RGB channels near the center
    colors[:, :3] *= 1 - strength * dip[:, None]

    return ListedColormap(colors, name="turbo_darkmid")


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["cmr10"],  # Use Matplotlib's internal CM font
        "mathtext.fontset": "cm",  # Keep the math the same
        "axes.formatter.use_mathtext": True,  # Prevents empty boxes for minus signs
    }
)


def ekin_fn(u, dx, axis=-1):
    """Compute kinetic energy."""
    dim = u.shape[-1]
    return 0.5 * (u**2).sum(axis) * dx**dim


def mse_fn(a, b, axis=-1):
    """Compute MSE error along given dimensions."""
    return ((a - b) ** 2).sum(axis)


def _sample_indices(n_total, n_frames):
    """Sample equidistant frame indices."""
    n_frames = max(1, min(n_frames, n_total))
    return np.linspace(0, n_total - 1, num=n_frames, dtype=int)


def _mae_per_trajectory(pred_vals, ref_vals):
    """Compute per-trajectory MAE, pairing trajectories when counts match."""
    pred = np.asarray(pred_vals, dtype=np.float64)
    ref = np.asarray(ref_vals, dtype=np.float64)

    if pred.ndim == 1:
        pred = pred[None, :]
    if ref.ndim == 1:
        ref = ref[None, :]

    t_ref = ref.shape[1]
    if pred.shape[1] != t_ref:
        x_pred = np.linspace(0.0, 1.0, pred.shape[1], dtype=np.float64)
        x_ref = np.linspace(0.0, 1.0, t_ref, dtype=np.float64)
        pred = np.vstack([np.interp(x_ref, x_pred, row) for row in pred])

    if ref.shape[0] == pred.shape[0]:
        return np.abs(pred - ref).mean(axis=1)

    ref_mean = ref.mean(axis=0, keepdims=True)
    return np.abs(pred - ref_mean).mean(axis=1)


def _baseline_error_stats(baseline, nx_key="256"):
    """Extract min/mean/max error stats for a given SPH resolution from baseline JSON."""
    nx_key = str(nx_key)

    ekin_pred = baseline.get("ekin", {}).get("pred", {}).get(nx_key, {}).get("values")
    ekin_ref = baseline.get("ekin", {}).get("ref", {}).get("values")
    spec_pred = baseline.get("spectrum", {}).get("pred", {}).get(nx_key, {}).get("values")
    spec_ref = baseline.get("spectrum", {}).get("ref", {}).get("values")

    ekin_pred_x = baseline.get("ekin", {}).get("pred", {}).get(nx_key, {}).get("x")
    ekin_ref_x = baseline.get("ekin", {}).get("ref", {}).get("x")
    spec_pred_x = baseline.get("spectrum", {}).get("pred", {}).get(nx_key, {}).get("x")
    spec_ref_x = baseline.get("spectrum", {}).get("ref", {}).get("x")

    if ekin_pred is None:
        return
    assert np.isclose(ekin_pred_x[0], ekin_ref_x[0])
    assert np.isclose(ekin_pred_x[-1], ekin_ref_x[-1])
    assert all(np.isclose(spec_pred_x, spec_ref_x))
    ekin_err = _mae_per_trajectory(ekin_pred, ekin_ref)
    spec_err = _mae_per_trajectory(spec_pred, spec_ref)

    return {
        "nx": nx_key,
        "ekin": {
            "min": float(np.min(ekin_err)),
            "med": float(np.mean(ekin_err)),
            "max": float(np.max(ekin_err)),
        },
        "spec": {
            "min": float(np.min(spec_err)),
            "med": float(np.mean(spec_err)),
            "max": float(np.max(spec_err)),
        },
    }


def plt_stats_lastframe_2panel(
    paths,
    names,
    fig_suffix="",
    data_dir="./data/2D_KOLM_4096_140kevery1",
    fig_dir="figs_kolm1",
    baseline_json="./sph/cu/res/kolm2d_comparison_data.json",
    is_fill=False,
):
    """Plot 2 panels: shifted kinetic energy over time and last-frame spectrum."""
    assert len(paths) == len(names), "Number of paths and names must match"
    metadata = load_metadata(data_dir)
    dx = metadata["dx"]
    box_size = metadata.get("L", 2 * np.pi)

    if "kolm1" in fig_dir:
        dt_run = 0.01 if "kolm10" in fig_dir else 0.001
        dim = 2
    else:  # HIT every 10
        dt_run = 0.02
        dim = 3

    data = {name: defaultdict(list) for name in names}
    error_stats = {name: {} for name in names}
    for path, name in zip(paths, names):
        rlts = [f for f in os.listdir(path) if f.startswith("rollout_") and f.endswith(".pkl")]
        rlts.sort()

        ekin_err = []
        spec_err = []
        for rlt_i in rlts:
            rollout = pickle.load(open(os.path.join(path, rlt_i), "rb"))
            pred_r = rollout["predicted_rollout"][0:]
            gt_r = rollout["ground_truth_rollout"][0:]
            pred_u = rollout["predicted_u_vel"][0:]
            gt_u = rollout["ground_truth_u_vel"][0:]

            ek_pred = ekin_fn(pred_u, dx, (1, 2))
            ek_gt = ekin_fn(gt_u, dx, (1, 2))

            data[name]["ekin"].append(ek_pred)
            data[name]["ekin_gt"].append(ek_gt)
            ekin_err.append(np.abs(ek_pred - ek_gt).mean())
            try:
                pred_grid = interpolate_velocity_to_grid_mls2(
                    r=pred_r[-1],
                    u=pred_u[-1],
                    nx=infer_n_per_dim(pred_r.shape[1], dim=pred_r.shape[-1]),
                    dim=pred_r.shape[-1],
                    box_size=box_size,
                )
                gt_grid = interpolate_velocity_to_grid_mls2(
                    r=gt_r[-1],
                    u=gt_u[-1],
                    nx=infer_n_per_dim(gt_r.shape[1], dim=gt_r.shape[-1]),
                    dim=gt_r.shape[-1],
                    box_size=box_size,
                )
            except ZeroNeighborsInterpolationError:
                print(f"Skipping spectrum for {name}::{rlt_i}: zero-neighbor MLS2 at last frame")
                continue

            k, spec_pred = spectrum_from_grid(pred_grid)
            _, spec_gt = spectrum_from_grid(gt_grid)
            data[name]["k"].append(k)
            data[name]["spec"].append(spec_pred)
            data[name]["spec_gt"].append(spec_gt)
            spec_err.append(np.abs(spec_pred - spec_gt).mean())

        if len(ekin_err) == 0:
            raise RuntimeError(f"No valid trajectories for {name} in lastframe 2-panel plot")

        error_stats[name]["ekin"] = {
            "min": float(np.min(ekin_err)),
            "med": float(np.mean(ekin_err)),
            "max": float(np.max(ekin_err)),
        }
        if len(spec_err) > 0:
            error_stats[name]["spec"] = {
                "min": float(np.min(spec_err)),
                "med": float(np.mean(spec_err)),
                "max": float(np.max(spec_err)),
            }
        print(
            f"{name}: Ekin MAE = {np.mean(ekin_err):.2e}[{np.min(ekin_err):.2e},{np.max(ekin_err):.2e}] over {len(ekin_err)} trajs"
        )

    data_plt = {name: {} for name in names}
    for name in names:
        if len(data[name]["ekin"]) == 0:
            raise RuntimeError(f"No valid trajectories for {name} in lastframe 2-panel plot")

        ek_array = np.array(data[name]["ekin"])
        ek_gt_array = np.array(data[name]["ekin_gt"])
        ek_med = np.mean(ek_array, axis=0)
        ek_gt_med = np.mean(ek_gt_array, axis=0)

        # Shift each trajectory so the first time step aligns with the mean at t=0.
        ek_centered = ek_array - ek_array[:, [0]] + ek_med[0]
        ek_gt_centered = ek_gt_array - ek_gt_array[:, [0]] + ek_gt_med[0]

        data_plt[name]["ekin"] = {
            "med": np.mean(ek_centered, axis=0),
            "min": np.min(ek_centered, axis=0),
            "max": np.max(ek_centered, axis=0),
        }
        data_plt[name]["ekin_gt"] = {
            "med": np.mean(ek_gt_centered, axis=0),
            "min": np.min(ek_gt_centered, axis=0),
            "max": np.max(ek_gt_centered, axis=0),
        }
        data_plt[name]["ekin_axis"] = np.arange(ek_array.shape[1]) * dt_run

        if len(data[name]["spec"]) == 0:
            raise RuntimeError(f"No valid spectra for {name} in lastframe 2-panel plot")

        data_plt[name]["k"] = np.array(data[name]["k"])[0]
        data_plt[name]["spec"] = stats(np.array(data[name]["spec"]))
        data_plt[name]["spec_gt"] = stats(np.array(data[name]["spec_gt"]))

    plt.rcParams.update({"font.size": 16})
    fig, axs = plt.subplots(1, 2, figsize=(6.5, 4), layout="constrained")
    ax_ekin, ax_spec = axs

    baseline_items_ekin = []
    baseline_items_spec = []
    baseline_64_errors = None
    baseline_128_errors = None
    baseline_256_errors = None
    baseline_path = Path(baseline_json)
    if baseline_path.exists():  # SPH baselines
        with baseline_path.open("r") as file:
            baseline = json.load(file)
        baseline_64_errors = _baseline_error_stats(baseline, nx_key="64")
        baseline_128_errors = _baseline_error_stats(baseline, nx_key="128")
        baseline_256_errors = _baseline_error_stats(baseline, nx_key="256")

        pred_ekin = baseline["ekin"].get("pred", {})
        ref_ekin_x = np.asarray(baseline["ekin"]["ref"]["x"], dtype=np.float64)
        for nx in sorted(pred_ekin.keys(), key=int):
            ek_vals = np.asarray(pred_ekin[nx]["values"], dtype=np.float64)
            t = np.asarray(pred_ekin[nx]["x"], dtype=np.float64)
            assert np.isclose(t[0], ref_ekin_x[0]) and np.isclose(t[-1], ref_ekin_x[-1])
            ek_med = ek_vals.mean(axis=0)
            ek_centered = ek_vals - ek_vals[:, [0]] + ek_med[0]
            ek_stats = stats(ek_centered)
            baseline_items_ekin.append((nx, t, ek_stats))

        pred_spec = baseline["spectrum"].get("pred", {})
        for nx in sorted(pred_spec.keys(), key=int):
            k = np.asarray(pred_spec[nx]["x"], dtype=np.float64)
            spec_vals = np.asarray(pred_spec[nx]["values"], dtype=np.float64)
            spec_stats = stats(spec_vals)
            baseline_items_spec.append((nx, k, spec_stats))
    else:
        print(f"Baseline comparison data not found at {baseline_path}")

    n_baseline = len(baseline_items_ekin)
    n_ml = len(names)
    baseline_colors = plt.cm.Grays(np.linspace(0.4, 0.7, n_baseline)) if n_baseline > 0 else []
    ml_colors = [f"C{i}" for i in range(n_ml)]

    # Plot GT first so legend/order starts with GT.
    ref_data = data_plt[names[0]]
    ek_axis = ref_data["ekin_axis"]
    ek_gt = ref_data["ekin_gt"]
    ax_ekin.plot(ek_axis, ek_gt["med"], "k", lw=3)
    if is_fill:
        ax_ekin.fill_between(ek_axis, ek_gt["min"], ek_gt["max"], color="k", alpha=0.1)

    k_ref = ref_data["k"]
    spec_gt = ref_data["spec_gt"]
    ax_spec.plot(k_ref, spec_gt["med"], "k", label="GT", lw=3)
    if is_fill:
        ax_spec.fill_between(k_ref, spec_gt["min"], spec_gt["max"], color="k", alpha=0.1)

    # Plot baselines second with a consistent baseline palette.
    for color, (nx, t, ek_stats) in zip(baseline_colors, baseline_items_ekin):
        ax_ekin.plot(t, ek_stats["med"], "--", color=color, lw=2)
        if is_fill:
            ax_ekin.fill_between(t, ek_stats["min"], ek_stats["max"], color=color, alpha=0.15)

    for color, (nx, k, spec_stats) in zip(baseline_colors, baseline_items_spec):
        ax_spec.plot(k, spec_stats["med"], "--", color=color, label=rf"SPH, {nx}$^{dim}$", lw=2)
        if is_fill:
            ax_spec.fill_between(k, spec_stats["min"], spec_stats["max"], color=color, alpha=0.15)

    # Plot ML runs last with a separate ML palette.
    for i, name in enumerate(names):
        data_i = data_plt[name]
        color = ml_colors[i]

        ek = data_i["ekin"]
        ek_axis = data_i["ekin_axis"]
        ax_ekin.plot(ek_axis, ek["med"], color=color, label=name, lw=2)
        if is_fill:
            ax_ekin.fill_between(ek_axis, ek["min"], ek["max"], color=color, alpha=0.12)

        k = data_i["k"]
        spec = data_i["spec"]
        ax_spec.plot(k, spec["med"], color=color, lw=2)
        if is_fill:
            ax_spec.fill_between(k, spec["min"], spec["max"], color=color, alpha=0.12)

    ax_ekin.set_xlabel("Time")
    ax_ekin.set_ylabel("Kinetic energy")
    if "kolm1" in fig_dir:
        ax_ekin.set_xticks([0, 3.5, 7, 10.5, 14])
        ax_ekin.set_xlim(-0.2, 14.2)
        ax_ekin.set_ylim(5)
    ax_ekin.legend(fontsize=11, frameon=False)

    ax_spec.set_xlabel("Wavenumber")
    ax_spec.set_ylabel("Energy spectrum at time=" + ("14" if "kolm1" in fig_dir else "5"))
    ax_spec.set_xscale("log")
    ax_spec.set_yscale("log")
    ax_spec.legend(fontsize=11, frameon=False)
    if "kolm1" in fig_dir:
        ax_spec.set_xlim(0.95, 34)
    # ax_spec.set_ylim(2e-5)

    # for ax in axs:
    #     ax.grid()

    fig.savefig(f"./logs/{fig_dir}/{fig_suffix}2panel_lastframe.pdf", dpi=300)
    fig.savefig(f"./logs/{fig_dir}/{fig_suffix}2panel_lastframe.png", dpi=300)
    plt.close(fig)

    # Marginalized interval plot across models for scalar error summaries.
    x = np.arange(len(names))
    ek_mean = np.array([error_stats[name]["ekin"]["med"] for name in names], dtype=np.float64)
    ek_min = np.array([error_stats[name]["ekin"]["min"] for name in names], dtype=np.float64)
    ek_max = np.array([error_stats[name]["ekin"]["max"] for name in names], dtype=np.float64)

    spec_mean = np.array([error_stats[name]["spec"]["med"] for name in names], dtype=np.float64)
    spec_min = np.array([error_stats[name]["spec"]["min"] for name in names], dtype=np.float64)
    spec_max = np.array([error_stats[name]["spec"]["max"] for name in names], dtype=np.float64)

    plt.rcParams.update({"font.size": 16})
    fig, axs = plt.subplots(
        1, 2, figsize=(6.5, 4), layout="constrained", gridspec_kw={"wspace": 0.04}
    )
    ax_ekin_err, ax_spec_err = axs

    ax_ekin_err.errorbar(
        x,
        ek_mean,
        yerr=[ek_mean - ek_min, ek_max - ek_mean],
        fmt="o",
        color="C0",
        ecolor="C0",
        capsize=5,
        lw=2,
        label=r"$[\mathbf{v} \rightarrow \mathbf{u}]$",
    )
    ax_spec_err.errorbar(
        x,
        spec_mean,
        yerr=[spec_mean - spec_min, spec_max - spec_mean],
        fmt="o",
        color="C0",
        ecolor="C0",
        capsize=5,
        lw=2,
        label=r"$[\mathbf{v} \rightarrow \mathbf{u}]$",
    )

    xmin, xmax = -0.5, len(names) - 0.5

    if baseline_64_errors is not None and "hit1" in fig_dir:
        b_ek = baseline_64_errors["ekin"]
        b_sp = baseline_64_errors["spec"]
        ax_ekin_err.axhline(b_ek["med"], color="0.35", ls=":", lw=1.5, label=rf"SPH, $64^{dim}$")
        ax_ekin_err.fill_between(
            [xmin, xmax],
            [b_ek["min"], b_ek["min"]],
            [b_ek["max"], b_ek["max"]],
            color="0.35",
            alpha=0.06,
        )

        ax_spec_err.axhline(b_sp["med"], color="0.35", ls=":", lw=1.5, label=rf"SPH, $64^{dim}$")
        ax_spec_err.fill_between(
            [xmin, xmax],
            [b_sp["min"], b_sp["min"]],
            [b_sp["max"], b_sp["max"]],
            color="0.35",
            alpha=0.06,
        )

    if baseline_128_errors is not None:
        ls = "--" if "hit1" in fig_dir else ":"
        b_ek = baseline_128_errors["ekin"]
        b_sp = baseline_128_errors["spec"]
        ax_ekin_err.axhline(b_ek["med"], color="0.35", ls=ls, lw=1.5, label=rf"SPH, $128^{dim}$")
        ax_ekin_err.fill_between(
            [xmin, xmax],
            [b_ek["min"], b_ek["min"]],
            [b_ek["max"], b_ek["max"]],
            color="0.35",
            alpha=0.06,
        )

        ax_spec_err.axhline(b_sp["med"], color="0.35", ls=ls, lw=1.5, label=rf"SPH, $128^{dim}$")
        ax_spec_err.fill_between(
            [xmin, xmax],
            [b_sp["min"], b_sp["min"]],
            [b_sp["max"], b_sp["max"]],
            color="0.35",
            alpha=0.06,
        )

    if baseline_256_errors is not None and "kolm1" in fig_dir:
        b_ek = baseline_256_errors["ekin"]
        b_sp = baseline_256_errors["spec"]
        ax_ekin_err.axhline(b_ek["med"], color="k", ls="--", lw=1.7, label=rf"SPH, $256^{dim}$")
        ax_ekin_err.fill_between(
            [xmin, xmax],
            [b_ek["min"], b_ek["min"]],
            [b_ek["max"], b_ek["max"]],
            color="k",
            alpha=0.08,
        )

        ax_spec_err.axhline(b_sp["med"], color="k", ls="--", lw=1.7, label=rf"SPH, $256^{dim}$")
        ax_spec_err.fill_between(
            [xmin, xmax],
            [b_sp["min"], b_sp["min"]],
            [b_sp["max"], b_sp["max"]],
            color="k",
            alpha=0.08,
        )

    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels([n.split("std=")[-1] for n in names])
        ax.set_xlabel("Noise Std")
        ax.margins(x=0.0)

    ax_ekin_err.set_ylabel("Ekin MAE over trajectories")

    if "hit1" in fig_dir:
        ax_ekin_err.set_ylim(0, 16)
        ax_ekin_err.legend(fontsize=11, frameon=False, loc="upper left")
    else:
        ax_ekin_err.set_ylim(0)
        ax_ekin_err.legend(fontsize=11, frameon=False)
    ax_spec_err.set_ylim(0)

    ax_spec_err.set_ylabel("Spectrum MAE at time=" + ("14" if "kolm1" in fig_dir else "5"))

    fig.savefig(f"./logs/{fig_dir}/{fig_suffix}2panel_lastframe_marginalized.pdf", dpi=300)
    fig.savefig(f"./logs/{fig_dir}/{fig_suffix}2panel_lastframe_marginalized.png", dpi=300)
    plt.close(fig)


def plt_runtime_vs_ekin_mae(
    paths,
    names,
    fig_suffix="",
    data_dir="./data/2D_KOLM_4096_140kevery1",
    fig_dir="figs_kolm1",
    baseline_json="./sph/cu/res/kolm2d_comparison_data.json",
    baseline_perf_json="./sph/cu/kolm2d_perf.json",
):
    """Plot runtime vs kinetic energy MAE, structured like plt_runtime_vs_corr_time."""
    assert len(paths) == len(names), "Number of paths and names must match"
    metadata = load_metadata(data_dir)
    dx = metadata["dx"]

    if "hit1" in fig_dir:
        baseline_perf_json = "./sph/cu/hit3d_perf.json"
        ml_runtime = 20.5
        dim = 3
    else:
        ml_runtime = 22.0
        dim = 2

    # Compute ekin MAE for each ML run (same logic as plt_stats_lastframe_2panel)
    ml_error_stats = {name: {} for name in names}
    for path, name in zip(paths, names):
        rlts = [f for f in os.listdir(path) if f.startswith("rollout_") and f.endswith(".pkl")]
        rlts.sort()

        ekin_err = []
        for rlt_i in rlts:
            rollout = pickle.load(open(os.path.join(path, rlt_i), "rb"))
            pred_u = rollout["predicted_u_vel"][2:]
            gt_u = rollout["ground_truth_u_vel"][2:]

            ek_pred = ekin_fn(pred_u, dx, (1, 2))
            ek_gt = ekin_fn(gt_u, dx, (1, 2))
            ekin_err.append(np.abs(ek_pred - ek_gt).mean())

        if len(ekin_err) == 0:
            raise RuntimeError(f"No valid trajectories for {name}")

        ml_error_stats[name]["ekin"] = {
            "min": float(np.min(ekin_err)),
            "med": float(np.mean(ekin_err)),
            "max": float(np.max(ekin_err)),
        }

    # Load baseline data: extract ekin MAE and runtime for each resolution
    baseline_x_vals, baseline_x_min, baseline_x_max = [], [], []
    baseline_y_vals, baseline_labels = [], []
    baseline_path = Path(baseline_json)
    perf_path = Path(baseline_perf_json)

    if baseline_path.exists() and perf_path.exists():
        with perf_path.open("r") as file:
            perf_data = json.load(file)
        with baseline_path.open("r") as file:
            baseline = json.load(file)

        runtimes = perf_data.get("runtime", [])
        resolutions = perf_data.get("resolution", [])

        for nx, runtime in zip(resolutions, runtimes):
            baseline_errors = _baseline_error_stats(baseline, nx_key=str(nx))
            if baseline_errors:  # Only include if error stats exist
                ekin_med = baseline_errors["ekin"]["med"]
                ekin_min = baseline_errors["ekin"]["min"]
                ekin_max = baseline_errors["ekin"]["max"]
                baseline_x_vals.append(float(ekin_med))
                baseline_x_min.append(float(ekin_min))
                baseline_x_max.append(float(ekin_max))
                baseline_y_vals.append(float(runtime))
                baseline_labels.append(str(nx))
    else:
        if not baseline_path.exists():
            print(f"Baseline comparison data not found at {baseline_path}")
        if not perf_path.exists():
            print(f"SPH performance data not found at {perf_path}")

    # Match the visual style of plt_stats_lastframe_2panel: larger text and compact width.
    plt.rcParams.update({"font.size": 16})
    fig, ax = plt.subplots(figsize=(5, 3.7), layout="constrained")

    # Plot baseline SPH line
    if baseline_x_vals:
        ax.plot(baseline_x_vals, baseline_y_vals, "o-", lw=2.5, ms=8)
        bx = np.asarray(baseline_x_vals, dtype=np.float64)
        bx_min = np.asarray(baseline_x_min, dtype=np.float64)
        bx_max = np.asarray(baseline_x_max, dtype=np.float64)
        ax.errorbar(
            bx,
            baseline_y_vals,
            xerr=[bx - bx_min, bx_max - bx],
            fmt="none",
            ecolor="0.35",
            elinewidth=1.5,
            capsize=4,
            zorder=2,
        )
        for x, y, label in zip(baseline_x_vals, baseline_y_vals, baseline_labels):
            ax.annotate(
                rf"{label}$^{dim}$", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=12
            )

    kwargs = dict(xycoords="axes fraction", ha="left", fontsize=13)
    if "hit1" in fig_dir:
        ax.annotate("SPH baselines", (0.4, 0.8), **kwargs)
    else:
        ax.annotate("SPH baselines", (0.15, 0.8), **kwargs)
    # Plot only the best ML run (lowest ekin MAE) at fixed runtime.
    ml_x_vals = [ml_error_stats[name]["ekin"]["med"] for name in names]
    best_i = int(np.argmin(ml_x_vals))
    best_name = names[best_i]
    best_x = float(ml_x_vals[best_i])
    best_x_min = float(ml_error_stats[best_name]["ekin"]["min"])
    best_x_max = float(ml_error_stats[best_name]["ekin"]["max"])
    best_y = float(ml_runtime)
    ax.scatter([best_x], [best_y], color="C1", s=120, zorder=5)
    ax.errorbar(
        [best_x],
        [best_y],
        xerr=[[best_x - best_x_min], [best_x_max - best_x]],
        fmt="none",
        ecolor="C1",
        elinewidth=2,
        capsize=4,
        zorder=4,
    )
    ax.annotate(
        best_name, (best_x, best_y), textcoords="offset points", xytext=(6, 6), fontsize=12
    )

    ax.set_xlabel("Kinetic Energy MAE")
    ax.set_ylabel("Runtime per simulation [s]")
    ax.set_yscale("log")
    ax.set_xlim(0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(f"./logs/{fig_dir}/{fig_suffix}runtime_vs_ekin_mae.pdf", dpi=300)
    fig.savefig(f"./logs/{fig_dir}/{fig_suffix}runtime_vs_ekin_mae.png", dpi=300)
    plt.close(fig)


def plt_scatter_final_frames_2d(
    paths,
    names,
    fig_suffix="",
    rlt_idx=0,
    vmin=0.0,
    vmax=6.0,
    font_size=28,
    fig_dir="figs_kolm1",
):
    """Scatter panel: GT target and final predictions of all runs."""
    assert len(paths) == len(names), "Number of paths and names must match"

    rollouts = []
    for path in paths:
        rlts = [f for f in os.listdir(path) if f.startswith("rollout_") and f.endswith(".pkl")]
        rlts.sort()
        rlt_i = rlts[rlt_idx]
        rollouts.append(pickle.load(open(os.path.join(path, rlt_i), "rb")))

    ref = rollouts[0]
    gt_tar_pos = ref["ground_truth_rollout"][-1]
    gt_tar_u = ref["ground_truth_u_vel"][-1]

    pred_states = []
    for rollout in rollouts:
        pred_states.append((rollout["predicted_rollout"][-1], rollout["predicted_u_vel"][-1]))

    ncols = 1 + len(paths)
    # Add extra width for left text space
    fig = plt.figure(figsize=(0.8 + 3.2 * ncols, 3.7), layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.08, wspace=0.02, hspace=0.02)
    # Create GridSpec with extra column on left for text
    gs = fig.add_gridspec(1, ncols + 1, width_ratios=[0.25] + [1] * ncols)

    # Create axes for plots (skip first gridspec column)
    axs = [fig.add_subplot(gs[0, i + 1]) for i in range(ncols)]

    # Create invisible text axis on the left
    ax_text = fig.add_subplot(gs[0, 0])
    ax_text.axis("off")

    norm = Normalize(vmin=vmin, vmax=vmax)

    all_pos = [gt_tar_pos] + [state[0] for state in pred_states]
    x_min = min(pos[:, 0].min() for pos in all_pos)
    x_max = max(pos[:, 0].max() for pos in all_pos)
    y_min = min(pos[:, 1].min() for pos in all_pos)
    y_max = max(pos[:, 1].max() for pos in all_pos)

    x_mid = 0.5 * (x_min + x_max)
    y_mid = 0.5 * (y_min + y_max)
    half_span = 0.5 * max(x_max - x_min, y_max - y_min)
    x_lim = (x_mid - half_span, x_mid + half_span)
    y_lim = (y_mid - half_span, y_mid + half_span)

    # import seaborn as sns
    def draw(ax, pos, vel, title):
        sc = ax.scatter(
            pos[:, 0],
            pos[:, 1],
            c=np.linalg.norm(vel, axis=-1),
            cmap="viridis",
            norm=norm,
            s=8,
        )
        ax.set_title(title, fontsize=font_size, pad=15)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        return sc

    last_sc = draw(axs[0], gt_tar_pos, gt_tar_u, r"Ground truth")
    for i, (name, (pos, vel)) in enumerate(zip(names, pred_states), start=1):
        draw(axs[i], pos, vel, f"{name}")

    cbar = fig.colorbar(last_sc, ax=axs, fraction=0.02, pad=0.01, shrink=0.8)
    cbar.set_label(r"$|\mathbf{u}|$", fontsize=font_size)
    cbar.ax.tick_params(labelsize=font_size)

    # Add vertical text on the left showing step and NSPH status
    step = len(rollouts[0]["predicted_rollout"]) - 3
    nsph_status = "with NSPH" if "n" in fig_suffix else "without NSPH"
    vertical_text = f"step={step}\n{nsph_status}"
    ax_text.text(
        0.5,
        0.5,
        vertical_text,
        rotation=90,
        va="center",
        ha="center",
        fontsize=font_size,
        weight="bold",
        transform=ax_text.transAxes,
    )

    kwargs = dict(dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"./logs/{fig_dir}/{fig_suffix}scatter_final_{rlt_idx}.pdf", **kwargs)
    fig.savefig(f"./logs/{fig_dir}/{fig_suffix}scatter_final_{rlt_idx}.png", **kwargs)
    plt.close(fig)


def plt_scatter_gt_and_runs_timeline_2d(
    paths,
    names,
    fig_suffix="",
    rlt_idx=0,
    vmin=0.0,
    vmax=6.0,
    font_size=22,
    fig_dir="figs_kolm1",
    baseline_root="./sph/cu/res",
    baseline_case="kolm2d",
    is_baseline=False,
):
    """Timeline scatter grid with GT row, model rows, and lowest-resolution baseline row."""
    assert len(paths) == len(names), "Number of paths and names must match"

    rollouts = []
    for path in paths:
        rlts = [f for f in os.listdir(path) if f.startswith("rollout_") and f.endswith(".pkl")]
        rlts.sort()
        rlt_i = rlts[rlt_idx]
        rollouts.append(pickle.load(open(os.path.join(path, rlt_i), "rb")))

    ref = rollouts[0]
    gt_pos = ref["ground_truth_rollout"]
    gt_vel = ref["ground_truth_u_vel"]
    t0 = 0
    dt_run = 0.01 if "kolm10" in fig_dir else 0.001
    t_end = len(gt_pos) - 1
    t_quarter = t_end // 4
    t_mid = t_end // 2

    baseline_row = None
    baseline_root_path = Path(baseline_root)
    baseline_case_dirs = []
    for cand in baseline_root_path.glob(f"{baseline_case}_*"):
        if not cand.is_dir():
            continue
        suffix = cand.name.split("_")[-1]
        if suffix.isdigit():
            baseline_case_dirs.append(cand)
    baseline_case_dirs = sorted(baseline_case_dirs, key=lambda p: int(p.name.split("_")[-1]))
    if baseline_case_dirs and is_baseline:
        lowest_dir = baseline_case_dirs[0]
        traj_id = 15 + rlt_idx
        baseline_traj = lowest_dir / f"traj_{traj_id}"
        if not baseline_traj.exists():
            traj_dirs = sorted([d for d in lowest_dir.glob("traj_*") if d.is_dir()])
            baseline_traj = traj_dirs[0] if traj_dirs else None
        if baseline_traj is not None and baseline_traj.exists():
            state_files = sorted(
                baseline_traj.glob("state_step_*.bin"),
                key=lambda p: int(p.stem.split("_")[-1]),
            )
            if state_files:
                step_ids = [0, len(state_files) // 4, len(state_files) // 2, len(state_files) - 1]

                def _load_state_bin(path):
                    raw = Path(path).read_bytes()
                    nx, n = struct.unpack_from("<ii", raw, 0)
                    _ = struct.unpack_from("<d", raw, 8)[0]
                    vec_bytes = n * 3 * 8
                    x = (
                        np.frombuffer(raw, dtype="<f8", count=n * 3, offset=16)
                        .reshape(n, 3)
                        .copy()
                    )
                    u = (
                        np.frombuffer(raw, dtype="<f8", count=n * 3, offset=16 + vec_bytes)
                        .reshape(n, 3)
                        .copy()
                    )
                    return nx, x[:, :2], u[:, :2]

                baseline_states = [_load_state_bin(state_files[i]) for i in step_ids]
                baseline_row = {
                    "nx": baseline_states[0][0],
                    "quarter": (baseline_states[1][1], baseline_states[1][2]),
                    "mid": (baseline_states[2][1], baseline_states[2][2]),
                    "end": (baseline_states[3][1], baseline_states[3][2]),
                }

    n_rows = 1 + len(paths) + (1 if baseline_row is not None else 0)
    n_cols = 4
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.38 * n_cols, 2.2 * n_rows),
        layout="constrained",
    )
    if n_rows == 1:
        axs = np.expand_dims(axs, axis=0)

    norm = Normalize(vmin=vmin, vmax=vmax)

    all_pos = [gt_pos[t0], gt_pos[t_quarter], gt_pos[t_mid], gt_pos[t_end]]
    for rollout in rollouts:
        pred_pos = rollout["predicted_rollout"]
        all_pos.extend([pred_pos[t_quarter], pred_pos[t_mid], pred_pos[t_end]])
    if baseline_row is not None:
        all_pos.extend(
            [baseline_row["quarter"][0], baseline_row["mid"][0], baseline_row["end"][0]]
        )

    x_lim, y_lim = (0, 2 * np.pi), (0, 2 * np.pi)

    def draw(ax, pos, vel, title=None):
        sc = ax.scatter(
            pos[:, 0],
            pos[:, 1],
            c=np.linalg.norm(vel, axis=-1),
            cmap="viridis",
            norm=norm,
            s=3,
        )
        if title is not None:
            ax.set_title(title, fontsize=font_size, pad=8)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        return sc

    def put_row_label(row_idx, label):
        axs[row_idx, 0].text(
            0.99,
            0.5,
            label,
            transform=axs[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=font_size,
            rotation="vertical",
        )

    last_sc = draw(axs[0, 0], gt_pos[t0], gt_vel[t0], title=f"Time={t0 * dt_run:.1f}")
    draw(axs[0, 1], gt_pos[t_quarter], gt_vel[t_quarter], title=f"Time={t_quarter * dt_run:.1f}")
    draw(axs[0, 2], gt_pos[t_mid], gt_vel[t_mid], title=f"Time={t_mid * dt_run:.1f}")
    draw(axs[0, 3], gt_pos[t_end], gt_vel[t_end], title=f"Time={t_end * dt_run:.1f}")
    axs[0, 0].set_ylabel("Ground truth", fontsize=font_size)

    for row_i, (name, rollout) in enumerate(zip(names, rollouts), start=1):
        pred_pos = rollout["predicted_rollout"]
        pred_vel = rollout["predicted_u_vel"]
        axs[row_i, 0].axis("off")
        draw(axs[row_i, 1], pred_pos[t_quarter], pred_vel[t_quarter], title=None)
        draw(axs[row_i, 2], pred_pos[t_mid], pred_vel[t_mid], title=None)
        draw(axs[row_i, 3], pred_pos[t_end], pred_vel[t_end], title=None)
        put_row_label(row_i, name)

    if baseline_row is not None:
        row_i = len(paths) + 1
        axs[row_i, 0].axis("off")
        draw(axs[row_i, 1], baseline_row["quarter"][0], baseline_row["quarter"][1], title=None)
        draw(axs[row_i, 2], baseline_row["mid"][0], baseline_row["mid"][1], title=None)
        draw(axs[row_i, 3], baseline_row["end"][0], baseline_row["end"][1], title=None)
        put_row_label(row_i, rf"SPH {baseline_row['nx']}$^2$")

    cbar = fig.colorbar(last_sc, ax=axs, fraction=0.02, pad=0.01, shrink=0.85)
    cbar.set_label(r"$|\mathbf{u}|$", fontsize=font_size)
    cbar.ax.tick_params(labelsize=max(10, font_size - 2))

    kwargs = dict(dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"./logs/{fig_dir}/{fig_suffix}scatter_timeline_{rlt_idx}.pdf", **kwargs)
    fig.savefig(f"./logs/{fig_dir}/{fig_suffix}scatter_timeline_{rlt_idx}.png", **kwargs)
    plt.close(fig)


def plt_scatter_gt_and_runs_timeline_3d(
    paths,
    names,
    fig_suffix="",
    rlt_idx=0,
    vmin=0.0,
    vmax=3.0,
    font_size=20,
    fig_dir="figs_hit10",
    baseline_root="./sph/cu/res",
    baseline_case="hit3d",
    is_baseline=False,
    elev=25.0,
    azim=-35.0,
    roll=0.0,
):
    """Timeline scatter grid for 3D rollouts in corner view.

    Uses three frames only: t=0, t=2.5, and t=5.0 (final frame).
    Layout follows the 2D timeline: GT fills the top row, model rows use
    an empty first column and show only t=2.5 and t=5.0.
    """
    assert len(paths) == len(names), "Number of paths and names must match"

    rollouts = []
    for path in paths:
        rlts = [f for f in os.listdir(path) if f.startswith("rollout_") and f.endswith(".pkl")]
        rlts.sort()
        rlt_i = rlts[rlt_idx]
        rollouts.append(pickle.load(open(os.path.join(path, rlt_i), "rb")))

    ref = rollouts[0]
    gt_pos = ref["ground_truth_rollout"]
    gt_vel = ref["ground_truth_u_vel"]

    t0 = 0
    t_end = len(gt_pos) - 1
    t_mid = int(round(0.5 * t_end))
    frame_ids = [t0, t_mid, t_end]
    time_labels = [0.0, 2.5, 5.0]

    baseline_row = None
    baseline_root_path = Path(baseline_root)
    baseline_case_dirs = []
    for cand in baseline_root_path.glob(f"{baseline_case}_*"):
        if not cand.is_dir():
            continue
        suffix = cand.name.split("_")[-1]
        if suffix.isdigit():
            baseline_case_dirs.append(cand)
    baseline_case_dirs = sorted(baseline_case_dirs, key=lambda p: int(p.name.split("_")[-1]))
    if baseline_case_dirs and is_baseline:
        lowest_dir = baseline_case_dirs[0]
        traj_id = 15 + rlt_idx
        baseline_traj = lowest_dir / f"traj_{traj_id}"
        if not baseline_traj.exists():
            traj_dirs = sorted([d for d in lowest_dir.glob("traj_*") if d.is_dir()])
            baseline_traj = traj_dirs[0] if traj_dirs else None
        if baseline_traj is not None and baseline_traj.exists():
            state_files = sorted(
                baseline_traj.glob("state_step_*.bin"),
                key=lambda p: int(p.stem.split("_")[-1]),
            )
            if state_files:
                step_ids = [0, len(state_files) // 2, len(state_files) - 1]

                def _load_state_bin(path):
                    raw = Path(path).read_bytes()
                    nx, n = struct.unpack_from("<ii", raw, 0)
                    _ = struct.unpack_from("<d", raw, 8)[0]
                    vec_bytes = n * 3 * 8
                    x = (
                        np.frombuffer(raw, dtype="<f8", count=n * 3, offset=16)
                        .reshape(n, 3)
                        .copy()
                    )
                    u = (
                        np.frombuffer(raw, dtype="<f8", count=n * 3, offset=16 + vec_bytes)
                        .reshape(n, 3)
                        .copy()
                    )
                    return nx, x, u

                baseline_states = [_load_state_bin(state_files[i]) for i in step_ids]
                baseline_row = {
                    "nx": baseline_states[0][0],
                    "mid": (baseline_states[1][1], baseline_states[1][2]),
                    "end": (baseline_states[2][1], baseline_states[2][2]),
                }

    n_rows = 1 + len(paths) + (1 if baseline_row is not None else 0)
    n_cols = 3
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.38 * n_cols, 2.2 * n_rows),
        subplot_kw={"projection": "3d"},
        layout="constrained",
    )
    if n_rows == 1:
        axs = np.expand_dims(axs, axis=0)

    # Prefer explicit artist z-order so box edges remain visible over dense point clouds.
    for ax_i in axs.flat:
        if hasattr(ax_i, "computed_zorder"):
            ax_i.computed_zorder = False

    norm = Normalize(vmin=vmin, vmax=vmax)
    lim = (0, 2 * np.pi)

    def draw_box_edges(ax):
        lo, hi = lim
        corners = np.array(
            [
                [lo, lo, lo],
                [lo, lo, hi],
                [lo, hi, lo],
                [lo, hi, hi],
                [hi, lo, lo],
                [hi, lo, hi],
                [hi, hi, lo],
                [hi, hi, hi],
            ],
            dtype=np.float64,
        )
        # Front-only cube cue: one face + 4 depth connectors (8 edges total).
        edges = [
            (0, 1),
            (0, 4),
            (1, 5),
            (4, 5),
            (3, 7),
            (1, 3),
            (4, 6),
            (5, 7),
        ]
        for i, j in edges:
            p0, p1 = corners[i], corners[j]
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color="0.1",
                lw=1.1,
                alpha=0.95,
                zorder=30,
            )

    def draw(ax, pos, vel, title=None):
        sc = ax.scatter(
            pos[:, 0],
            pos[:, 1],
            pos[:, 2],
            c=np.linalg.norm(vel, axis=-1),
            cmap="viridis",
            norm=norm,
            s=3,
            depthshade=False,
            zorder=5,
        )
        if title is not None:
            ax.set_title(title, fontsize=font_size, pad=8)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_zlim(*lim)
        ax.view_init(elev=elev, azim=azim, roll=roll)
        draw_box_edges(ax)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        return sc

    def put_row_label(row_idx, label):
        axs[row_idx, 0].text2D(
            0.99,
            0.5,
            label,
            transform=axs[row_idx, 0].transAxes,
            ha="right",
            va="center",
            fontsize=font_size,
            rotation="vertical",
        )

    # Ground-truth row uses all timeline columns.
    last_sc = None
    for col_i, (frame_id, time_label) in enumerate(zip(frame_ids, time_labels)):
        last_sc = draw(
            axs[0, col_i], gt_pos[frame_id], gt_vel[frame_id], title=f"Time={time_label:.1f}"
        )
    fig.canvas.draw()
    gt_pos_y = 0.5 * (axs[0, 0].get_position().y0 + axs[0, 0].get_position().y1)
    fig.text(
        0.012,
        gt_pos_y,
        "Ground truth",
        ha="right",
        va="center",
        fontsize=font_size,
        rotation="vertical",
    )

    for row_i, (name, rollout) in enumerate(zip(names, rollouts), start=1):
        pred_pos = rollout["predicted_rollout"]
        pred_vel = rollout["predicted_u_vel"]
        axs[row_i, 0].set_axis_off()
        draw(axs[row_i, 1], pred_pos[t_mid], pred_vel[t_mid], title=None)
        draw(axs[row_i, 2], pred_pos[t_end], pred_vel[t_end], title=None)
        put_row_label(row_i, name)

    if baseline_row is not None:
        row_i = len(paths) + 1
        axs[row_i, 0].set_axis_off()
        draw(axs[row_i, 1], baseline_row["mid"][0], baseline_row["mid"][1], title=None)
        draw(axs[row_i, 2], baseline_row["end"][0], baseline_row["end"][1], title=None)
        put_row_label(row_i, rf"SPH {baseline_row['nx']}$^3$")

    cbar = fig.colorbar(last_sc, ax=axs, fraction=0.02, pad=0.01, shrink=0.85)
    cbar.set_label(r"$|\mathbf{u}|$", fontsize=font_size)
    cbar.ax.tick_params(labelsize=max(10, font_size - 2))

    kwargs = dict(dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"./logs/{fig_dir}/{fig_suffix}scatter_timeline_3d_corner_{rlt_idx}.pdf", **kwargs)
    fig.savefig(f"./logs/{fig_dir}/{fig_suffix}scatter_timeline_3d_corner_{rlt_idx}.png", **kwargs)
    plt.close(fig)


def get_grid_counts(points, L=2 * np.pi, grid_size=5):
    """How many particles are in each of grid_size**2 cells
    points: Tensor of shape (B, N, 2)
    """
    B, N, _ = points.shape
    num_bins = grid_size**2

    # Scale to [0, 1] then to [0, grid_size]
    scaled = (points / L) * grid_size
    # Clamp to avoid out-of-bounds errors for points exactly on the maximum boundary
    # We want indices from 0 to grid_size - 1
    scaled = np.clip(scaled, 0, grid_size - 1e-5)
    # Cast to integer indices
    grid_indices = scaled.astype(np.int64)  # Shape: (B, N, 2)

    # Flatten 2D indices to 1D (y_index * width + x_index). Shape (B, N)
    flat_indices = grid_indices[..., 1] * grid_size + grid_indices[..., 0]
    # Shape (B, 1) so it broadcasts correctly against (B, N)
    batch_offsets = np.arange(B)[:, None] * num_bins
    # Add offsets to create globally unique bins
    global_indices = flat_indices + batch_offsets  # Shape: (B, N)

    # Run the single 1D bincount on the flattened array
    # minlength ensures we get exactly B * 25 bins, even if the last batch is empty
    h_flat = np.bincount(global_indices.ravel(), minlength=B * num_bins)
    # Reshape back to (B, 25)
    h = h_flat.reshape(B, num_bins)

    return h


def plt_stats(
    paths, names, fig_suffix="", data_dir="./data/2D_KOLM_4096_140kevery1", fig_dir="figs_kolm1"
):
    """Plot kinetic energy error evolution over trajectory length."""
    assert len(paths) == len(names), "Number of paths and names must match"
    metadata = load_metadata(data_dir)
    dx = metadata["dx"]

    # for each path/name, and for each trajectory therein, we compute the quantities
    data = {name: defaultdict(list) for name in names}
    for i, (path, name) in enumerate(zip(paths, names)):
        # list all rollout files of format "rollout_00000.pkl"
        rlts = os.listdir(path)
        rlts = [f for f in rlts if f.startswith("rollout_") and f.endswith(".pkl")]
        rlts.sort()  # sort by name to ensure order

        # compute stats for each rollout
        for rlt_i in rlts:
            rollout = pickle.load(open(os.path.join(path, rlt_i), "rb"))
            ek_gt = ekin_fn(rollout["ground_truth_u_vel"][2:], dx, (1, 2))
            data[name]["ekin_gt"].append(ek_gt)  # (T,)
            ek_pred = ekin_fn(rollout["predicted_u_vel"][2:], dx, (1, 2))
            data[name]["ekin"].append(ek_pred)  # (T,)
            data[name]["mean_v"].append(rollout["predicted_u_vel"][2:].mean(1))  # (T, 2)
            grid_counts = get_grid_counts(rollout["predicted_rollout"][2:])
            data[name]["grid_cnt_min"].append(grid_counts.min(-1))  # (T,)

    # compute means and bounds
    data_plt = {name: defaultdict(dict) for name in names}
    for name in names:
        ek_array = np.array(data[name]["ekin"])
        print(f"{name} ek_array.shape: {ek_array.shape}")
        ek_med = np.mean(ek_array, axis=0)
        data_plt[name]["ekin"]["med"] = ek_med
        # Shift each trajectory so that its first element matches ek_med[0]
        ek_array_mean_centered = ek_array - ek_array[:, [0]] + ek_med[0]
        data_plt[name]["ekin"]["min"] = np.min(ek_array_mean_centered, axis=0)
        data_plt[name]["ekin"]["max"] = np.max(ek_array_mean_centered, axis=0)

        ek_gt = np.array(data[name]["ekin_gt"])
        data_plt[name]["ekin_gt"]["med"] = np.mean(ek_gt, axis=0)

        mean_v = np.array(data[name]["mean_v"])
        data_plt[name]["mean_v"]["med"] = np.mean(mean_v, axis=0)
        data_plt[name]["mean_v"]["min"] = np.min(mean_v, axis=0)
        data_plt[name]["mean_v"]["max"] = np.max(mean_v, axis=0)

        grid_cnt_ref = len(rollout["predicted_rollout"][0]) / 25  # 25 is number of grid cells
        grid_cnt_min = np.array(data[name]["grid_cnt_min"]) / grid_cnt_ref
        data_plt[name]["grid_cnt_min"]["med"] = np.mean(grid_cnt_min, axis=0)
        data_plt[name]["grid_cnt_min"]["min"] = np.min(grid_cnt_min, axis=0)
        data_plt[name]["grid_cnt_min"]["max"] = np.max(grid_cnt_min, axis=0)

    # plot results
    font_size = 14
    plt.rcParams.update({"font.size": font_size})
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    for i, (key, data_i) in enumerate(data_plt.items()):
        ek = data_i["ekin"]
        axis = np.arange(len(ek["med"]))  # * int(key.split("_")[0])
        if i == 0:
            ek_gt = data_i["ekin_gt"]
            axs[0, 0].plot(axis, ek_gt["med"], "k", label="GT")
        axs[0, 0].plot(axis, ek["med"], label=names[i])
        # Use the same color as the line just plotted above
        color = axs[0, 0].lines[-1].get_color()
        kwargs = {"alpha": 0.2, "color": color}

        v = data_i["mean_v"]
        axs[0, 1].plot(axis, v["med"][:, 0], label=f"{names[i]}")
        axs[1, 1].plot(axis, v["med"][:, 1], label=f"{names[i]}")

        cnt = data_i["grid_cnt_min"]
        axs[1, 0].plot(axis, cnt["med"], label=f"{names[i]}")

        if True:  # plot band gap
            axs[0, 0].fill_between(axis, ek["min"], ek["max"], **kwargs)
            axs[0, 1].fill_between(axis, v["min"][:, 0], v["max"][:, 0], **kwargs)
            axs[1, 1].fill_between(axis, v["min"][:, 1], v["max"][:, 1], **kwargs)
            axs[1, 0].fill_between(axis, cnt["min"], cnt["max"], **kwargs)

    axs[0, 0].set_ylabel("Kinetic energy")
    axs[0, 0].set_yscale("log")
    axs[0, 1].set_ylabel("Mean velocity x")
    axs[1, 0].set_ylabel("Min grid density")
    axs[1, 1].set_ylabel("Mean velocity y")
    # axs[0, 0].set_yscale("log")
    axs[1, 0].set_ylim(top=1.01)
    axs[1, 0].legend()
    for ax in axs.flat:
        ax.grid()
        ax.set_xlabel("Time step")

    plt.tight_layout()
    fig.savefig(f"./logs/{fig_dir}/{fig_suffix}.pdf", dpi=300)
    fig.savefig(f"./logs/{fig_dir}/{fig_suffix}.png", dpi=300)
    plt.close()


def animate_rlt(path, slice_n=10, suffix="", rlt_idx=0, fig_dir="figs_kolm1"):
    """Animate side-by-side ground-truth and predicted particle trajectories."""

    rlts = os.listdir(path)
    rlts = [f for f in rlts if f.startswith("rollout_") and f.endswith(".pkl")]
    rlts.sort()
    rlt = rlts[rlt_idx]

    rollout = pickle.load(open(os.path.join(path, rlt), "rb"))
    pred_positions = rollout["predicted_rollout"][::slice_n]
    gt_positions = rollout["ground_truth_rollout"][::slice_n]
    pred_colors = np.linalg.norm(rollout["predicted_u_vel"][::slice_n], axis=-1)
    gt_colors = np.linalg.norm(rollout["ground_truth_u_vel"][::slice_n], axis=-1)

    norm = Normalize(vmin=0, vmax=6)

    fig, axs = plt.subplots(1, 2, figsize=(5, 2.45), layout="constrained")
    step_width = len(str((len(pred_positions) - 1) * slice_n))
    step_text = fig.text(0.35, 0.975, "", ha="left", va="top", fontsize=10, family="monospace")

    scatter_gt = axs[0].scatter(
        gt_positions[0, :, 0],
        gt_positions[0, :, 1],
        c=gt_colors[0],
        cmap="viridis",
        norm=norm,
        s=3,
    )
    scatter_pred = axs[1].scatter(
        pred_positions[0, :, 0],
        pred_positions[0, :, 1],
        c=pred_colors[0],
        cmap="viridis",
        norm=norm,
        s=3,
    )

    axs[0].set_title("Ground truth")
    axs[1].set_title("Prediction")
    for ax in axs:
        ax.set_xlim(0, 2 * np.pi)
        ax.set_ylim(0, 2 * np.pi)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    sm = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axs, fraction=0.03, pad=0.02)
    cbar.set_label(r"$|\mathbf{u}|$")

    # Warm-up render: let constrained_layout settle before collecting
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200)  # discarded

    # Render all frames to memory
    frames = []
    for frame in range(len(pred_positions)):
        scatter_gt.set_offsets(gt_positions[frame])
        scatter_gt.set_array(gt_colors[frame])
        scatter_pred.set_offsets(pred_positions[frame])
        scatter_pred.set_array(pred_colors[frame])
        step_text.set_text(f"(Step: {frame * slice_n:{step_width}d})")

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    plt.close()

    # Save with a single global palette across all frames
    imageio.mimsave(
        f"./logs/{fig_dir}/anim_{suffix[-15:]}_{rlt_idx}.gif",
        frames,
        fps=10,
        quantizer="nq",
        palettesize=256,
    )


def rlt_path(ckpt_date, rlt_type):
    """Construct rollout path."""
    return os.path.join("./logs/train/runs/", ckpt_date, "rlt", rlt_type)


def get_paths(name):
    """Select a comparison case."""
    names = [
        r"$[\mathbf{v} \rightarrow \mathbf{u}]$",  # LAG
        r"$[\mathbf{u} \leftrightarrow \mathbf{v}]_\mathbf{v}$",  # GNS simple
        r"$[\mathbf{u} \leftrightarrow \mathbf{v}]_\mathbf{u_t}$",  # GNS tvf
        r"$[\mathbf{u} \leftrightarrow \mathbf{v}]_\mathbf{u_{t+1}}$",  # GNS simple_u
        r"$[\mathbf{u} \rightarrow \mathbf{v}]$",  # GINO lr=0.001
    ]

    ### Kolm-1 runs
    if name == "kolm1_500":
        paths = [
            rlt_path("2026-04-02_20-40-23", "501"),  # LAG
            rlt_path("2026-04-03_01-42-23", "501"),  # GNS simple
            rlt_path("2026-04-03_12-32-13", "501"),  # GNS tvf
            rlt_path("2026-04-03_17-49-09", "501"),  # GNS simple_u
            rlt_path("2026-04-02_20-40-40", "501"),  # GINO
        ]
    elif name == "kolm1_500_1n3":
        paths = [
            rlt_path("2026-04-02_20-40-23", "501_1n3"),  # LAG
            rlt_path("2026-04-03_01-42-23", "501_1n3"),  # GNS simple
            rlt_path("2026-04-03_12-32-13", "501_1n3"),  # GNS tvf
            rlt_path("2026-04-03_17-49-09", "501_1n3"),  # GNS simple_u
            rlt_path("2026-04-02_20-40-40", "501_1n3"),  # GINO
        ]
    elif name == "kolm1_2000_1n3":
        paths = [
            rlt_path("2026-04-02_20-40-23", "2001_1n3"),  # LAG
            rlt_path("2026-04-03_01-42-23", "2001_1n3"),  # GNS simple
            rlt_path("2026-04-03_12-32-13", "2001_1n3"),  # GNS tvf
            rlt_path("2026-04-03_17-49-09", "2001_1n3"),  # GNS simple_u
            rlt_path("2026-04-02_20-40-40", "2001_1n3"),  # GINO
        ]
    elif name == "kolm1_14000_sub":
        paths = [
            rlt_path("2026-04-02_20-40-23", "13999_1n1"),  # LAG
            rlt_path("2026-04-02_20-40-23", "13999_1n1_nov2u"),  # LAG
            rlt_path("2026-04-02_20-40-23", "13999_1n3"),  # LAG
            rlt_path("2026-04-02_20-40-40", "13999"),  # GINO - GINO doesn't need NSPH!
            rlt_path("2026-04-02_20-40-40", "13999_1n3"),  # GINO - NSPH3
        ]
        names = [
            names[0] + "$_{\\text{NSPH}=1}$",
            names[0] + "$_{\\text{NSPH}=1, \\text{ no }\mathbf{v}2\mathbf{u}}$",
            names[0] + "$_{\\text{NSPH}=3}$",
            names[-1],
            names[-1] + "$_{\\text{NSPH}=3}$",
        ]
    elif name == "kolm1_14000_paper":
        paths = [
            rlt_path("2026-04-02_20-40-23", "13999_1n1"),  # LAG
            rlt_path("2026-04-02_20-40-40", "13999"),  # GINO
        ]
        names = [names[0] + "$_{\\text{NSPH}=1}$", names[-1]]

    ### Kolm-10
    elif name == "kolm10_100_2n3":
        paths = [
            rlt_path("2026-04-02_00-02-07", "101_2n3"),  # LAG
            rlt_path("2026-04-02_00-03-14", "101_2n3"),  # GNS simple
            rlt_path("2026-04-02_00-03-36", "101_2n3"),  # GNS tvf
            rlt_path("2026-04-02_00-03-57", "101_2n3"),  # GNS simple_u
            rlt_path("2026-04-02_00-04-22", "101_2n3"),  # GINO
        ]
    elif name == "kolm10_1400":
        paths = [
            rlt_path("2026-04-02_00-02-07", "1399_2n3"),  # LAG
            rlt_path("2026-04-02_00-04-22", "1399_2n3"),  # GINO
        ]
        names = [names[0], names[-1]]
    elif name == "kolm10_1400_std":
        paths = [
            rlt_path("2026-04-02_00-02-07", "1399_2n3"),  # LAG
            rlt_path("2026-04-02_00-13-43", "1399_2n3"),  # std=1e-5
            rlt_path("2026-04-02_00-13-54", "1399_2n3"),  # std=3e-5
            rlt_path("2026-04-02_16-34-44", "1399_2n3"),  # std=1e-4
            rlt_path("2026-04-02_04-59-47", "1399_2n3"),  # std=3e-4
        ]
        x = names[0]
        names = [
            x + ", std=0",
            x + ", std=1e-5",
            x + ", std=3e-5",
            x + ", std=1e-4",
            x + ", std=3e-4",
        ]
    elif name == "kolm10_1400_wd":
        paths = [
            rlt_path("2026-04-02_00-02-07", "1399_2n3"),  # LAG
            rlt_path("2026-04-02_21-44-51", "1399_2n3"),  # wd=1e-5
            rlt_path("2026-04-02_16-35-08", "1399_2n3"),  # wd=1e-4
            rlt_path("2026-04-02_22-06-01", "1399_2n3"),  # wd=1e-3
        ]
        x = names[0]
        names = [x + ", wd=0", x + ", wd=1e-5", x + ", wd=1e-4", x + ", wd=1e-3"]
    elif name == "kolm10_1400_std_paper":
        paths = [
            rlt_path("2026-04-02_00-13-54", "1399_2n3"),  # std000003
        ]
        x = names[0]
        names = [x]

    # HIT-10
    elif name == "hit10_250":
        paths = [
            rlt_path("2026-04-03_12-50-53", "249_2n1"),  # std=0
            rlt_path("2026-04-05_05-02-10", "249_2n1"),  # std=1e-5
            rlt_path("2026-04-03_18-37-18", "249_2n1"),  # std=3e-5
            rlt_path("2026-04-03_22-04-27", "249_2n1"),  # std=1e-4
            rlt_path("2026-04-04_03-52-00", "249_2n1"),  # std=3e-4
        ]
        x = names[0]
        names = [x + f", std={std}" for std in ["0", "1e-5", "3e-5", "1e-4", "3e-4"]]
    elif name == "hit10_250_view":
        paths = [
            rlt_path("2026-04-03_18-37-18", "249_2n1"),  # std=3e-4
        ]
        x = names[0]
        names = [x + ", std=3e-5"]
    elif name == "hit10_250_seeds":
        paths = [
            rlt_path("2026-04-03_18-37-18", "249_2n1"),  # std=3e-5
            rlt_path("2026-04-19_13-35-06", "249_2n1"),  # std=3e-5 seed=124
            rlt_path("2026-04-19_13-38-16", "249_2n1"),  # std=3e-5 seed=125
        ]
        x = names[0]
        names = [x + f", seed={s}" for s in ["123", "124", "125"]]
    else:
        raise ValueError(f"name={name} not supported")

    slices = [10 if "kolm10" in name else 100 for _ in range(len(paths))]
    fig_dir = "figs_kolm10" if "kolm10" in name else "figs_kolm1"
    fig_dir = "figs_hit10" if "hit10" in name else fig_dir
    baseline_json = (
        "./sph/cu/res/kolm2d_comparison_data.json"
        if "kolm1" in name
        else ("./sph/cu/res/hit3d_comparison_data.json")
    )
    fig_suffix = name + "-"
    os.makedirs(f"./logs/{fig_dir}", exist_ok=True)
    return paths, names, fig_suffix, slices, fig_dir, baseline_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot stats and animations for rlt comparison")
    parser.add_argument("--name", type=str, required=True, default="kolm1_500")
    parser.add_argument("--animate", action="store_true", help="GIG of 1st traj per rlt.")
    args = parser.parse_args()

    if "kolm" in args.name:
        data_dir = "./data/2D_KOLM_4096_140kevery1"
    elif "hit" in args.name:
        data_dir = "./data/3D_HIT_32768_25kevery1"

    paths, names, fig_suffix, slices, fig_dir, baseline_json = get_paths(args.name)
    if args.animate:
        for i, (slice_n, name) in enumerate(zip(slices, names)):
            animate_rlt(paths[i], slice_n, name + "_" + fig_suffix, 4, fig_dir=fig_dir)
        exit()

    plt_stats(paths, names, fig_suffix, data_dir, fig_dir)
    if "kolm1_14000" in args.name or "kolm10_1400" in args.name or "hit10_250" in args.name:
        plt_stats_lastframe_2panel(paths, names, fig_suffix, data_dir, fig_dir, baseline_json)
        plt_runtime_vs_ekin_mae(paths, names, fig_suffix, data_dir, fig_dir, baseline_json)

    if "kolm1" in args.name:
        plt_scatter_gt_and_runs_timeline_2d(
            paths, names, fig_suffix, rlt_idx=4, fig_dir=fig_dir, is_baseline=False
        )
        for i in range(4, 5):
            plt_scatter_final_frames_2d(paths, names, fig_suffix, rlt_idx=i, fig_dir=fig_dir)
        pass
    else:
        plt_scatter_gt_and_runs_timeline_3d(paths, names, fig_suffix, rlt_idx=4, fig_dir=fig_dir)
        pass
