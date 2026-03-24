"""Runs after 'analyse_kolm_hit_2.py' to get the time until correlation drops below a threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _first_crossing_time(x: np.ndarray, y: np.ndarray, threshold: float) -> float | None:
    """Linearly interpolate to determine the first time x at which y crosses a threshold."""
    below = np.where(y < threshold)[0]
    if len(below) == 0:
        return None
    idx = int(below[0])
    if idx == 0 or y[idx] == y[idx - 1]:
        return float(x[idx])
    alpha = (threshold - y[idx - 1]) / (y[idx] - y[idx - 1])
    return float(x[idx - 1] + alpha * (x[idx] - x[idx - 1]))


def _median_with_nones_as_large(times: list[float | None]) -> float | None:
    """Compute the median of a list of times, treating None as larger than any number."""
    n_none = sum(t is None for t in times)
    if n_none > len(times) / 2:
        return None
    valid = [t if t is not None else float("inf") for t in times]
    return None if len(valid) == 0 else float(np.median(valid))


def get_time_until_corr(args) -> None:
    """Get the time until correlation drops below a threshold for each nx, and save to dst_file."""
    with args.src_file.open("r") as f:
        data = json.load(f)

    pred_blocks = data.get("corr", {}).get("pred", {})
    if not pred_blocks:
        raise ValueError("Missing corr.pred in input JSON (run analyse_kolm_hit_2.py with corr)")

    print(f"case={data.get('case')} threshold={args.threshold}")

    def _sort_key(nx_key: str) -> tuple[int, str]:
        try:
            return (0, int(nx_key))
        except ValueError:
            return (1, nx_key)

    times_by_nx = {}
    medians_by_nx = {}
    for nx in sorted(pred_blocks.keys(), key=_sort_key):
        x = np.asarray(pred_blocks[nx]["x"], dtype=float)
        vals = np.asarray(pred_blocks[nx]["values"], dtype=float)
        times = [_first_crossing_time(x, vals[i], args.threshold) for i in range(vals.shape[0])]
        times_by_nx[int(nx)] = times
        medians_by_nx[int(nx)] = _median_with_nones_as_large(times)

    for nx in sorted(times_by_nx):
        times = times_by_nx[nx]
        n_crossed = sum(t is not None for t in times)
        print(f"nx={nx}: crossed={n_crossed}/{len(times)} median={medians_by_nx[nx]}")

    dst = {}
    if args.dst_file.exists():
        with args.dst_file.open("r") as f:
            dst = json.load(f)

    threshold_tag = str(args.threshold).replace(".", "p")
    key_median = f"corr_time_until_lt_{threshold_tag}_median"
    key = f"corr_time_until_lt_{threshold_tag}"
    dst[key_median] = {str(nx): medians_by_nx[nx] for nx in sorted(medians_by_nx)}
    dst[key] = {str(nx): times_by_nx[nx] for nx in sorted(times_by_nx)}

    with args.dst_file.open("w") as f:
        json.dump(dst, f, indent=4)
    print(f"Merged into: {args.dst_file}")


def plt_runtime_vs_corr_time(dst_file: Path, threshold_tag: str) -> None:
    """Plot runtime vs median time until correlation drops below threshold. Save to dst_file."""
    with dst_file.open("r") as f:
        data = json.load(f)

    runtimes = data.get("runtime", [])
    resolutions = data.get("resolution", [])
    key = f"corr_time_until_lt_{threshold_tag}_median"
    medians_by_nx = data.get(key)
    dim = data["dim"]

    x_vals, y_vals, labels = [], [], []
    for nx, runtime in zip(resolutions, runtimes):
        median = medians_by_nx.get(str(nx))
        if median is None:
            continue
        x_vals.append(float(median))
        y_vals.append(float(runtime))
        labels.append(str(nx))

    fig, ax = plt.subplots(figsize=(4.5, 3), layout="constrained")
    ax.plot(x_vals, y_vals, "o-")
    for x, y, label in zip(x_vals, y_vals, labels):
        ax.annotate(
            rf"{label}$^{dim}$", (x, y), textcoords="offset points", xytext=(-len(label) * 10, 0)
        )
    ax.set_xlabel(f"Median time until correlation < {args.threshold}")
    ax.set_ylabel("Runtime per simulation [s]")
    ax.set_yscale("log")
    ax.set_xlim(0, 14 if "kolm2d" in dst_file.stem else 2.5)
    ax.grid()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    out_path = dst_file.with_name(f"{dst_file.stem}_{key}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get t until correlation drops below threshold")
    parser.add_argument("--src_file", type=Path, required=True, help="<case>_comparison_data.json")
    parser.add_argument("--dst_file", type=Path, required=True, help="path/to/<case>_perf.json")
    parser.add_argument("--threshold", type=float, required=True, help="Correlation threshold")
    args = parser.parse_args()

    get_time_until_corr(args)

    threshold_tag = str(args.threshold).replace(".", "p")
    plt_runtime_vs_corr_time(args.dst_file, threshold_tag)
