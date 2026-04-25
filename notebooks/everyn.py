import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from src.utils.data_utils import load_metadata
from matplotlib.colors import Normalize
import matplotlib.animation as animation


def ekin_fn(u, dx, axis=-1):
    """Compute kinetic energy."""
    return 0.5 * (u**2).sum(axis) * dx**2


def mse_fn(a, b, axis=-1):
    """Compute MSE error along given dimensions."""
    return ((a - b) ** 2).sum(axis)


def plt_ekin(paths, names, fig_suffix="", data_dir="./data/2D_KOLM_4096_20kevery10"):
    """Plot kinetic energy error evolution over trajectory length."""
    assert len(paths) == len(names), "Number of paths and names must match"
    metadata = load_metadata(data_dir)
    dx = metadata["dx"]

    # for each path/name, and for each trajectory therein, we compute the quantities
    ekin = {name: [] for name in names}
    for i, (path, name) in enumerate(zip(paths, names)):
        # list all rollout files of format "rollout_00000.pkl"
        rlts = os.listdir(path)
        rlts = [f for f in rlts if f.startswith("rollout_") and f.endswith(".pkl")]
        rlts.sort()  # sort by name to ensure order

        # compute stats for each rollout
        for rlt_i in rlts:
            rollout = pickle.load(open(os.path.join(path, rlt_i), "rb"))
            ek_gt = ekin_fn(rollout["ground_truth_u_vel"], dx, (1, 2))
            ek_pred = ekin_fn(rollout["predicted_u_vel"], dx, (1, 2))
            ekin[name].append(mse_fn(ek_gt, ek_pred, ()))  # shape has to be (T,)

    # compute means and bounds
    ekin_plt = {name: {} for name in names}
    for name in names:
        ek_array = np.array(ekin[name])[:, 2:]
        print(f"{name} ek_array.shape: {ek_array.shape}")

        ek_mean = np.mean(ek_array, axis=0)
        ekin_plt[name]["mean"] = ek_mean
        # Shift each trajectory so that its first element matches ek_mean[0]
        ek_array_mean_centered = ek_array - ek_array[:, [0]] + ek_mean[0]
        ekin_plt[name]["min"] = np.min(ek_array_mean_centered, axis=0)
        ekin_plt[name]["max"] = np.max(ek_array_mean_centered, axis=0)

    # plot results
    font_size = 12
    plt.rcParams.update({"font.size": font_size})
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    for i, (key, ek) in enumerate(ekin_plt.items()):
        ekin_axis = np.arange(len(ek["mean"])) * int(key.split("_")[0])
        ax.plot(ekin_axis, ek["mean"], label=names[i])
        # Use the same color as the line just plotted above
        color = ax.lines[-1].get_color()
        ax.fill_between(ekin_axis, ek["min"], ek["max"], alpha=0.2, color=color)

    ax.set_xlabel("Time step")
    ax.set_ylabel("Kinetic energy error")
    ax.set_yscale("log")
    ax.legend()
    ax.grid()

    plt.tight_layout()
    fig.savefig(f"./logs/figs/everyn_{fig_suffix}.pdf")
    fig.savefig(f"./logs/figs/everyn_{fig_suffix}.png")
    plt.close()


def animate_rlt(path, slice_n=1, suffix="", rlt_idx=0):
    """Animate particle evolution trajectory."""
    # list all rollout files of format "rollout_00000.pkl"
    rlts = os.listdir(path)
    rlts = [f for f in rlts if f.startswith("rollout_") and f.endswith(".pkl")]
    rlts.sort()  # sort by name to ensure order
    rlt = rlts[rlt_idx]

    rollout = pickle.load(open(os.path.join(path, rlt), "rb"))
    positions = rollout["predicted_rollout"][::slice_n]  # shape: (T, P, D)
    colors = (rollout["predicted_u_vel"] ** 2).mean(-1)[::slice_n]  # shape: (T, P)

    fig, ax = plt.subplots(figsize=(8, 8))
    norm = Normalize(vmin=0, vmax=10)

    def update(frame):
        ax.clear()
        scatter = ax.scatter(
            positions[frame, :, 0],
            positions[frame, :, 1],
            c=colors[frame],
            cmap="viridis",
            norm=norm,
            s=10,
        )
        ax.set_xlim(positions[:, :, 0].min(), positions[:, :, 0].max())
        ax.set_ylim(positions[:, :, 1].min(), positions[:, :, 1].max())
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Frame {frame}")
        return (scatter,)

    anim = animation.FuncAnimation(fig, update, frames=len(positions), interval=100, blit=False)
    anim.save(f"./logs/figs/everyn_anim_{suffix}_{rlt_idx}.gif", writer="pillow", fps=10)
    plt.close()


def rlt_path(ckpt_date, rlt_type):
    """Construct rollout path."""
    return os.path.join("./logs/train/runs/", ckpt_date, "rlt", rlt_type)


def get_paths(name):
    """Select a comparison case."""
    if name == "everyn-2k":
        paths = [
            rlt_path("2025-07-07_11-43-25", "21"),
            rlt_path("2026-03-01_01-43-27", "41"),
            rlt_path("2026-03-01_01-43-06", "101"),
            rlt_path("2026-03-01_02-03-36", "201"),
            rlt_path("2026-03-01_02-02-03", "2001"),
        ]
        names = ["100", "50", "20", "10", "1"]
        fig_suffix = ""
        slices = [1, 2, 5, 10, 100]
        # names==1 blows up hardest, names=20/50 look most promising
    elif name == "everyn_nsph3-2k":
        paths = [
            rlt_path("2025-07-07_11-43-25", "21_nsph3"),
            rlt_path("2026-03-01_01-43-27", "41_nsph3"),
            rlt_path("2026-03-01_01-43-06", "101_nsph3"),
            rlt_path("2026-03-01_02-03-36", "201_nsph3"),
            rlt_path("2026-03-01_02-02-03", "2001_nsph3"),
        ]
        names = ["100_nsph3-2k", "50_nsph3-2k", "20_nsph3-2k", "10_nsph3-2k", "1_nsph3-2k"]
        fig_suffix = "nsph3-2k"
        slices = [1, 2, 5, 10, 100]
        # names=1/100 blow up worst, names=20 does best
    elif name == "everyn_nsph3-10k":
        paths = [
            rlt_path("2026-03-01_01-43-27", "201_nsph3"),
            rlt_path("2026-03-01_01-43-06", "501_nsph3"),
            rlt_path("2026-03-01_02-03-36", "1001_nsph3"),
        ]
        names = ["50_nsph3-10k", "20_nsph3-10k", "10_nsph3-10k"]
        fig_suffix = "nsph3-10k"
        slices = [2, 5, 10]
        # from best to worst: names=10->20->50
    elif name == "everyn_noise-2k":
        paths = [
            rlt_path("2026-03-01_02-03-36", "201"),
            rlt_path("2026-03-02_01-42-18", "201_00001"),
            rlt_path("2026-03-02_01-43-03", "201_00003"),
            rlt_path("2026-03-02_01-43-30", "201_0001"),
        ]
        names = ["10_0-2k", "10_00001-2k", "10_00003-2k", "10_0001-2k"]
        fig_suffix = "noise-2k"
        slices = [10, 10, 10, 10]
        # std=0.0001 does best, and no noise worst
    elif name == "everyn_n3_noise-10k":
        paths = [
            rlt_path("2026-03-01_02-03-36", "1001_nsph3"),
            rlt_path("2026-03-02_01-42-18", "1001_n3_00001"),
            rlt_path("2026-03-02_01-43-03", "1001_n3_00003"),
            rlt_path("2026-03-02_01-43-30", "1001_n3_0001"),
        ]
        names = ["10_n3-10k", "10_n3_00001-10k", "10_n3_00003-10k", "10_n3_0001-10k"]
        fig_suffix = "n3_noise-10k"
        slices = [10, 10, 10, 10]
        # no noise stays most stable; very small noise helps mid-term, but then explodes
    elif name == "everyn_h5-2k":
        paths = [
            rlt_path("2026-03-01_02-03-36", "201"),
            rlt_path("2026-03-02_09-32-50", "201_h5"),
            rlt_path("2026-03-02_14-52-38", "201_00001_h5"),
        ]
        names = ["10_0-2k", "10_0_h5-2k", "10_00001_h5-2k"]
        fig_suffix = "noise_h5-2k"
        slices = [10, 10, 10]
        # until step=600, 00001_h5 is best. 5 steps w/o noise is worst.
    elif name == "everyn_n3_h5-10k":
        paths = [
            rlt_path("2026-03-01_02-03-36", "1001_nsph3"),
            rlt_path("2026-03-02_09-32-50", "1001_n3_h5"),
            rlt_path("2026-03-02_14-52-38", "1001_n3_00001_h5"),
        ]
        names = ["10_n3-10k", "10_n3_0_h5-10k", "10_n3_00001_h5-10k"]
        fig_suffix = "noise_n3_h5-10k"
        slices = [10, 10, 10]
        # 1001_nsph3 is best.
    elif name == "everyn_h5_e50-2k":
        paths = [
            rlt_path("2026-03-01_02-03-36", "201"),
            rlt_path("2026-03-01_01-43-27", "41"),
            rlt_path("2026-03-02_23-14-55", "41_h5"),
        ]
        names = ["10_0", "50_0", "50_00001_h5"]
        fig_suffix = "h5_e50-2k"
        slices = [10, 2, 2]
        # last one does best until middle, them blows up
    elif name == "everyn_n3_h5_e50-2k":
        paths = [
            rlt_path("2026-03-01_02-03-36", "1001_nsph3"),
            rlt_path("2026-03-01_01-43-27", "201_nsph3"),
            rlt_path("2026-03-02_23-14-55", "201_n1_h5"),
            rlt_path("2026-03-02_23-14-55", "201_n3_h5"),
            rlt_path("2026-03-02_23-14-55", "201_n5_h5"),
        ]
        names = ["10_n3", "50_n3_0", "50_n1_00001_h5", "50_n3_00001_h5", "50_n5_00001_h5"]
        fig_suffix = "h5_n3-10k"
        slices = [10, 2, 2, 2, 2]
        # 1001_nsph3 is best.
    else:
        raise ValueError(f"name={name} not supported")

    return paths, names, fig_suffix, slices


paths, names, fig_suffix, slices = get_paths("everyn_n3_h5_e50-2k")
plt_ekin(paths, names, fig_suffix)
for i, (slice_n, name) in enumerate(zip(slices, names)):
    animate_rlt(paths[i], slice_n, name, 0)
