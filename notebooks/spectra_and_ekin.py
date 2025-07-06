import os
import pickle
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from jax import config

from src.utils.jax_utils.vis_utils import (
    energy_spectrum,
    mls_2nd_order,
    pos_init_cartesian_2d,
    read_h5,
)
from src.utils.data_utils import load_metadata

config.update("jax_platforms", "cpu")


def get_metadata(path):
    """Load dataset metadata from the given path."""
    metadata = load_metadata(path)
    dim, dx = metadata["dim"], metadata["dx"]
    N_total = metadata["num_particles_max"]
    Nx = round(N_total ** (1 / dim))
    bounds = metadata["bounds"]
    box_size = np.array(bounds)[:, 1] - np.array(bounds)[:, 0]
    return dim, dx, N_total, Nx, box_size


class EkinSpectrumComputer:
    """Class for computing kinetic energy and energy spectrum."""

    def __init__(self, metadata_root):
        dim, dx, N_total, Nx, box_size = get_metadata(metadata_root)
        self.dim = dim
        self.dx = dx
        self.N_total = N_total
        self.Nx = Nx

        self.k_axis = np.arange(1, Nx // 2 + 1)
        if dim == 2:
            self.r_target = pos_init_cartesian_2d(box_size, dx)
        elif dim == 3:
            raise NotImplementedError("3D case is not implemented yet")

        self.mls_fn = partial(
            mls_2nd_order,
            r_target=self.r_target,
            box_size=box_size,
            dx=dx,
            dim=dim,
            kernel_name="Quintic",
            h_factor=0.8,
        )

    def comp_spectrum(self, r, u):
        """Compute the energy spectrum from the particle positions and velocities."""
        dim, Nx, N_total = self.dim, self.Nx, self.N_total
        assert r.shape == (N_total, dim)
        assert u.shape == (N_total, dim)

        u_mls = np.array([self.mls_fn(r, f=u[:, i]) for i in range(dim)])
        # # visualize MLS
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        # axs[0].scatter(r[:, 0], r[:, 1], c=u[:, 0], s=5)
        # axs[1].scatter(self.r_target[:, 0], self.r_target[:, 1], c=u_mls[0, :], s=5)
        # fig.savefig("mls.png")
        # plt.close(fig)
        u_mls = u_mls.reshape(dim, Nx, Nx) if dim == 2 else u_mls.reshape(dim, Nx, Nx, Nx)
        spectrum = energy_spectrum(u_mls)
        spectrum *= 1 / (4 * np.pi)  # undo normalization to match magnitude from PDF above
        return spectrum

    def get_ekin_and_spectrum(self, rollout, ekin_axis, is_spectrum, is_gt=False):
        """Compute kinetic energy and energy spectrum given rollout and time steps."""
        if is_gt:
            keys = ["ground_truth_rollout", "ground_truth_u_vel"]
        else:
            keys = ["predicted_rollout", "predicted_u_vel"]

        ekin = []
        for t in ekin_axis:
            u = rollout[keys[1]][t]
            ekin.append((u**2).sum())
        ekin = 0.5 * np.array(ekin) * self.dx**2

        if is_spectrum:
            r = rollout[keys[0]][t]
            spectrum = self.comp_spectrum(r, u)
            spectrum = spectrum[1 : len(self.k_axis) + 1]
        else:
            spectrum = None

        return ekin, spectrum


def plt_ekin_and_spectra(
    experiment, every_n, step_last, step_stride=50, spectrum_num_plts=99999999, save_suffix=""
):
    """Plot kinetic energy and energy spectra for given paths and names.

    Args:
        experiment (str): Name of the experiment. Should be defined in `get_paths_names`.
        every_n (int): Dataset coarsening level. One of: {1, 10}.
        step_last (int): Rollout length and step at which to compute the spectra.
        step_stride (int): Step stride for kinetic energy evaluations. Based on "_every1" dataset.
            Has to be multiple of 10 as this is the writing frequency of the SPH reference.
        spectrum_num_plts (int): Only this many first paths will have energy spectrum computed.
        save_suffix (str): Suffix for saving the plots.
    """

    assert every_n in [1, 10], "Only every_n=1 or every_n=10 is supported"
    paths, names = get_paths_names(experiment, every_n == 1)
    assert len(paths) == len(names), "Number of paths and names must match"
    ekin_axis = np.arange(0, step_last + 1, step_stride)

    computer = EkinSpectrumComputer(paths[0])
    dx, k_axis = computer.dx, computer.k_axis

    # for each path/name, and for each trajectory therein, we compute the quantities
    ekin = {name: [] for name in names}
    spectra = {name: [] for name in names}
    for i, (path, name) in enumerate(zip(paths, names)):
        # print("Start path", path)

        if name == "SPH":
            for path_i in path:
                # check if preprocessed stats file exists
                stats_file = os.path.join(path_i, "stats.pkl")
                if os.path.exists(stats_file):
                    stats = pickle.load(open(stats_file, "rb"))
                else:
                    stats = {"ekin": {}, "spectra": {}}

                ekin_sub = []
                for t in ekin_axis:
                    if t * every_n not in stats["ekin"]:
                        # read the trajectory file
                        frame = read_h5(f"{path_i}/traj_{str(t * every_n).zfill(5)}.h5")
                        stats["ekin"][t * every_n] = 0.5 * (frame["u"] ** 2).sum() * dx**2
                    ekin_sub.append(stats["ekin"][t * every_n])
                ekin[name].append(np.array(ekin_sub))

                if t * every_n not in stats["spectra"]:
                    frame = read_h5(f"{path_i}/traj_{str(t * every_n).zfill(5)}.h5")
                    stats["spectra"][t * every_n] = computer.comp_spectrum(frame["r"], frame["u"])
                # print(len(spectrum), len(ekin_sub))
                spectra[name].append(stats["spectra"][t * every_n][1 : len(k_axis) + 1])

                # save the stats for this path
                with open(stats_file, "wb") as f:
                    pickle.dump(stats, f)

        elif "rlt" in path:
            # load precomputed stats if available
            if ("stats.pkl" in os.listdir(path)) and ("stats_gt.pkl" in os.listdir(path)):
                # check whether stats.pkl matches what we want
                stats = pickle.load(open(os.path.join(path, "stats.pkl"), "rb"))
                if (stats["ekin_axis"] == ekin_axis).all() and (stats["k_axis"] == k_axis).all():
                    ekin[name] = stats["ekin"]
                    spectra[name] = stats["spectra"]

                if len(ekin["Dataset"]) == 0:
                    # check for ground truth stats
                    stats = pickle.load(open(os.path.join(path, "stats_gt.pkl"), "rb"))
                    if (stats["ekin_axis"] == ekin_axis).all() and (
                        stats["k_axis"] == k_axis
                    ).all():
                        ekin["Dataset"] = stats["ekin"]
                        spectra["Dataset"] = stats["spectra"]

                continue

            # list all rollout files of format "rollout_00000.pkl"
            rlts = os.listdir(path)
            rlts = [f for f in rlts if f.startswith("rollout_") and f.endswith(".pkl")]
            rlts.sort()  # sort by name to ensure order

            # compute stats for each rollout
            do_dataset = "Dataset" not in ekin or len(ekin["Dataset"]) == 0
            for rlt_i in rlts:
                rollout = pickle.load(open(os.path.join(path, rlt_i), "rb"))
                for is_gt, name_ in zip([False, True], [name, "Dataset"]):
                    if is_gt and not do_dataset:
                        continue
                    is_sp = i < spectrum_num_plts
                    ek, sp = computer.get_ekin_and_spectrum(rollout, ekin_axis, is_sp, is_gt=is_gt)
                    ekin[name_].append(ek)
                    spectra[name_].append(sp)

            # write stats.pkl for this path
            for file_name, name_ in zip(["stats.pkl", "stats_gt.pkl"], [name, "Dataset"]):
                with open(os.path.join(path, file_name), "wb") as f:
                    stats = {
                        "ekin_axis": ekin_axis,
                        "k_axis": k_axis,
                        "ekin": ekin[name_],
                        "spectra": spectra[name_],
                    }
                    pickle.dump(stats, f)

            # font_size = 12
            # plt.rcParams.update({"font.size": font_size})
            # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            # axs[0].plot(ekin_axis, ekin["SPH"][0], f"k-", linewidth=2.0, label=f"SPH")
            # axs[1].plot(k_axis, spectra["SPH"][0], f"k-", linewidth=2.0, label=f"SPH")
            # for j in range(len(ekin[name])):
            #     axs[0].plot(ekin_axis, ekin["Dataset"][j], f"C{j}-", label=f"Dataset {j}")
            #     axs[0].plot(ekin_axis, ekin[name][j], f"C{j}--", label=f"{name} {j}")
            #     axs[1].plot(k_axis, spectra["Dataset"][j], f"C{j}-", label=f"Dataset {j}")
            #     axs[1].plot(k_axis, spectra[name][j], f"C{j}--", label=f"{name} {j}")
            # # axs[0].legend()
            # # axs[1].legend()

            # axs[0].set_xlabel("Time step")
            # axs[0].set_ylabel(r"Kinetic energy  $E_{kin}$")
            # # axs[0].set_ylabel(r"Dissipation rate  $- \partial E_{kin} / \partial t$")
            # axs[0].set_xlim(0, ekin_axis[-1])

            # axs[1].set_xlabel("Wavenumber k")
            # axs[1].set_ylabel(f"Energy spectrum at step {step_last}")
            # axs[1].set_xscale("log")
            # axs[1].set_xlim(1, len(k_axis))
            # axs[1].set_xticks([1, 10, 32])
            # axs[1].set_xticklabels([f"{int(k)}" for k in axs[1].get_xticks()])

            # for ax in axs:
            #     ax.grid()
            # # axs[0].set_yscale('log')
            # axs[0].set_ylim(80, 160)
            # axs[1].set_yscale("log")
            # axs[1].legend(loc="lower left")
            # plt.tight_layout()

            # """
            # if "SPH2" in name:
            #     fig_name = "2"
            # elif "SPH05" in name:
            #     fig_name = "05"
            # else:
            #     fig_name = "1"
            # plt.savefig(f"kolm_every1_5000_nsph{fig_name}.png")
            # if "NSPHtvf1" in name:
            #     fig_name = "tvf1"
            # elif "NSPHtvf01" in name:
            #     fig_name = "tvf01"
            # else:
            #     fig_name = ""
            # plt.savefig(f"kolm_every1_100_nsph1{fig_name}.png")
            # """
            # if "NSPHtvf01" in name:
            #     fig_name = "tvf01"
            # elif "NSPHtvf002" in name:
            #     fig_name = "tvf002"
            # elif "NSPHtvf001" in name:
            #     fig_name = "tvf001"
            # elif "NSPHtvf0005" in name:
            #     fig_name = "tvf0005"
            # elif "NSPHtvf0002" in name:
            #     fig_name = "tvf0002"
            # elif "NSPHtvf0001" in name:
            #     fig_name = "tvf0001"
            # else:
            #     fig_name = "tvf"
            # plt.savefig(f"kolm_every1_5000_nsph1{fig_name}.png")
            # plt.close()

    # print(jax.tree.map(lambda x: x.shape if hasattr(x, 'shape') else x, ekin))
    # print(jax.tree.map(lambda x: x.shape if hasattr(x, 'shape') else x, spectra))

    # average out the spectra and kinetic energy over all trajectories
    ekin_plt, spectra_plt = {name: {} for name in names}, {name: {} for name in names}
    for name in names:
        ek_array = np.array(ekin[name])
        sp_array = np.array(spectra[name])
        print(f"{name} ek_array.shape: {ek_array.shape}, sp_array.shape: {sp_array.shape}")

        ek_mean = np.mean(ek_array, axis=0)
        ekin_plt[name]["mean"] = ek_mean
        # Shift each trajectory so that its first element matches ek_mean[0]
        ek_array_mean_centered = ek_array - ek_array[:, [0]] + ek_mean[0]
        ekin_plt[name]["min"] = np.min(ek_array_mean_centered, axis=0)
        ekin_plt[name]["max"] = np.max(ek_array_mean_centered, axis=0)

        spectra_plt[name]["mean"] = np.mean(sp_array, axis=0)
        spectra_plt[name]["min"] = np.min(np.array(spectra[name]), axis=0)
        spectra_plt[name]["max"] = np.max(np.array(spectra[name]), axis=0)

    font_size = 12
    plt.rcParams.update({"font.size": font_size})
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for i, key in enumerate(ekin_plt.keys()):
        ek, sp = ekin_plt[key], spectra_plt[key]
        style = {"Dataset": ["k-", 2.5], "SPH": ["-", 1.5]}.get(names[i], ["--", 1.5])
        axs[0].plot(ekin_axis, ek["mean"], style[0], linewidth=style[1], label=names[i])
        # Use the same color as the line just plotted above
        color = axs[0].lines[-1].get_color()
        axs[0].fill_between(ekin_axis, ek["min"], ek["max"], alpha=0.2, color=color)
        if sp is not None:
            axs[1].plot(k_axis, sp["mean"], style[0], linewidth=style[1], label=names[i])
            axs[1].fill_between(k_axis, sp["min"], sp["max"], alpha=0.2, color=color)

    axs[0].set_xlabel("Time step")
    axs[0].set_ylabel(r"Kinetic energy  $E_{kin}$")
    # axs[0].set_ylabel(r"Dissipation rate  $- \partial E_{kin} / \partial t$")
    axs[0].set_xlim(0, ekin_axis[-1])
    # axs[0].set_yscale('log')
    axs[0].set_ylim(70, 140)
    if experiment == "lag_101_v2u_nsph":
        axs[0].set_yticks(range(128, 137, 2))
    elif experiment == "lag_1001_v2u_nsph":
        axs[0].set_ylim(110, 150)
        axs[0].set_yticks(range(110, 151, 10))
    elif experiment == "lag_20000_v2u_nsph":
        axs[0].set_ylim(40, 150)
        axs[0].set_xlim(0, 20000)

    axs[1].set_xlabel("Wavenumber k")
    axs[1].set_ylabel(f"Energy spectrum at step {step_last}")
    axs[1].set_xscale("log")
    axs[1].set_xlim(1, len(k_axis))
    axs[1].set_xticks([1, 10, 32])
    axs[1].set_xticklabels([f"{int(k)}" for k in axs[1].get_xticks()])
    axs[1].set_yscale("log")
    axs[1].set_ylim(1e-4, 2e0)
    axs[0].legend(loc="lower left")

    for ax in axs:
        ax.grid()
    plt.tight_layout()
    plt.savefig(f"./logs/figs/spectra_{every_n}_{experiment}{save_suffix}.pdf")
    plt.savefig(f"./logs/figs/spectra_{every_n}_{experiment}{save_suffix}.png")
    plt.show()
    plt.close()


def get_paths_names(experiment, is_every1=True, root_logs="./logs/train/runs", data_root="."):
    """Get paths and names for the given experiment name. Used by `plt_ekin_and_spectra`."""
    if is_every1:
        ckpts = {  # on every 1 step
            "lag": "2025-06-02_03-21-52",
            # "1000_nsph1": "2025-02-08_02-47-01",
        }
    else:
        ckpts = {  # on every 10 step
            "lag": "2025-06-15_23-15-04",
            "lag_noisy": "2025-06-15_23-18-37",
        }

    def rlt_path(ckpt_date, rlt_type):
        return os.path.join(root_logs, ckpt_date, "rlt", rlt_type)

    sph_paths = [
        f"{data_root}/data/2D_KOLM_SPH_0_20250210-232340_TVF_15",
        f"{data_root}/data/2D_KOLM_SPH_0_20250616-002306_TVF_16",
        f"{data_root}/data/2D_KOLM_SPH_0_20250616-002425_TVF_17",
        f"{data_root}/data/2D_KOLM_SPH_0_20250616-002543_TVF_18",
        f"{data_root}/data/2D_KOLM_SPH_0_20250616-002701_TVF_19",
    ]

    if experiment == "lag_1001_nsph":
        paths = [
            f"{data_root}/data/2D_KOLM_4096_200kevery1",
            sph_paths,
            rlt_path(ckpts["lag"], "1001_nsph02"),
            rlt_path(ckpts["lag"], "1001_nsph05"),
            rlt_path(ckpts["lag"], "1001_nsph1"),
            rlt_path(ckpts["lag"], "1001_nsph2"),
        ]
        names = [
            "Dataset",  # used only to get the metadata
            "SPH",
            "Simple-Base-02",
            "Simple-Base-05",
            "Simple-Base-1",
            "Simple-Base-2",
        ]
    elif experiment == "lag_101_v2u_nsph":
        paths = [
            f"{data_root}/data/2D_KOLM_4096_200kevery1",
            sph_paths,
            rlt_path(ckpts["lag"], "101"),
            rlt_path(ckpts["lag"], "101_nsph1"),
            rlt_path(ckpts["lag"], "101_gnn"),
            rlt_path(ckpts["lag"], "101_gnn_nsph1"),
        ]
        names = [
            "Dataset",  # used only to get the metadata
            "SPH",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$)",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$) + NSPH",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = f_{\theta}^{\mathbf{v}\to \mathbf{u}}(\mathbf{v})$)",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = f_{\theta}^{\mathbf{v}\to \mathbf{u}}(\mathbf{v})$) + NSPH",
        ]
    elif experiment == "lag_1001_v2u_nsph":
        paths = [
            f"{data_root}/data/2D_KOLM_4096_200kevery1",
            sph_paths,
            rlt_path(ckpts["lag"], "1001_nsph1"),
            rlt_path(ckpts["lag"], "1001_gnn_nsph1"),
            rlt_path(ckpts["lag"], "1001_gnn"),
            # rlt_path(ckpts["lag"], "1001_nsph1tvf100"),  with TVF every 100th step
        ]
        names = [
            "Dataset",  # used only to get the metadata
            "SPH",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$ ) + NSPH",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = f_{\theta}^{\mathbf{v}\to \mathbf{u}}(\mathbf{v})$) + NSPH",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = f_{\theta}^{\mathbf{v}\to \mathbf{u}}(\mathbf{v})$)",
            # r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$ ) + NSPHtvf100",
        ]
    elif experiment == "lag_5001_v2u_nsph":
        paths = [
            f"{data_root}/data/2D_KOLM_4096_200kevery1",
            sph_paths,
            # rlt_path(ckpts["lag"], "5001_nsph1tvf"),
            # rlt_path(ckpts["lag"], "5001_nsph1tvf01"),
            # rlt_path(ckpts["lag"], "5001_nsph1tvf002"),
            rlt_path(ckpts["lag"], "5001_nsph1tvf001"),
            rlt_path(ckpts["lag"], "5001_nsph1tvf0005"),  # best so far
            # rlt_path(ckpts["lag"], "5001_nsph1tvf0002"),
            # rlt_path(ckpts["lag"], "5001_nsph1tvf0001"),
            rlt_path(ckpts["lag"], "5001_nsph1tvf100"),  # second best
            # rlt_path(ckpts["lag"], "5001_nsph1tvf500"),
            # rlt_path(ckpts["lag"], "5001_nsph05"),
            rlt_path(ckpts["lag"], "5001_nsph1"),  # blows up as all others without TVF
            # rlt_path(ckpts["lag"], "5001_nsph1f2"),  # this and the one below both suck equally
            # rlt_path(ckpts["lag"], "5001_nsph2"),
            # rlt_path(ckpts["lag"], "5001_gnn_nsph1"),
            rlt_path(ckpts["lag"], "5001_nsph1tvf10f01"),  # same as 5001_nsph1tvf001
        ]
        names = [
            "Dataset",  # used only to get the metadata
            "SPH",
            # r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$ ) + NSPHtvf",
            # r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$ ) + NSPHtvf01",
            # r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$ ) + NSPHtvf002",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$ ) + NSPHtvf001",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$ ) + NSPHtvf0005",
            # r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$ ) + NSPHtvf0002",
            # r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$ ) + NSPHtvf0001",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$ ) + NSPHtvf100",
            # r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$ ) + NSPHtvf500",
            # r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$ ) + NSPH05",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$ ) + NSPH",
            # r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$ ) + NSPHf2",
            # r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$ ) + NSPH2",
            # r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = f_{\theta}^{\mathbf{v}\to \mathbf{u}}(\mathbf{v})$) + NSPH",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$ ) + NSPHtvf10f01",
        ]
    elif experiment == "lag_101_v2u_nsph1tvf":
        paths = [
            f"{data_root}/data/2D_KOLM_4096_200kevery1",
            sph_paths,
            rlt_path(ckpts["lag"], "101_nsph1"),
            rlt_path(ckpts["lag"], "101_nsph1tvf01"),
            rlt_path(ckpts["lag"], "101_nsph1tvf1"),
        ]
        names = [
            "Dataset",  # used only to get the metadata
            "SPH",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$) + NSPH",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$) + NSPHtvf01",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$) + NSPHtvf1",
        ]
    elif experiment == "lag_20000_v2u_nsph":
        paths = [
            f"{data_root}/data/2D_KOLM_4096_200kevery1",
            sph_paths,
            rlt_path(ckpts["lag"], "20000_nsph1tvf001"),
            rlt_path(ckpts["lag"], "20000_nsph1tvf0005"),
            rlt_path(ckpts["lag"], "20000_nsph1tvf100"),
        ]
        names = [
            "Dataset",  # used only to get the metadata
            "SPH",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$ ) + NSPHtvf001",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$ ) + NSPHtvf0005",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$ ) + NSPHtvf100",
        ]
    elif experiment == "lag10_101":
        paths = [
            f"{data_root}/data/2D_KOLM_4096_20kevery10",
            sph_paths,
            rlt_path(ckpts["lag"], "101"),
            rlt_path(ckpts["lag"], "101_nsph1"),
            rlt_path(ckpts["lag"], "101_nsph1tvf1"),
            rlt_path(ckpts["lag"], "101_gnn"),
            rlt_path(ckpts["lag"], "101_gnn_nsph1"),
        ]
        names = [
            "Dataset",  # used only to get the metadata
            "SPH",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$)",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$) + NSPH",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = \mathbf{v}$) + NSPHtvf1",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = f_{\theta}^{\mathbf{v}\to \mathbf{u}}(\mathbf{v})$)",
            r"$\mathbf{v}\to \mathbf{u}$ ($\mathbf{u} = f_{\theta}^{\mathbf{v}\to \mathbf{u}}(\mathbf{v})$) + NSPH",
        ]
    elif experiment == "lag10_500":
        paths = [
            f"{data_root}/data/2D_KOLM_4096_20kevery10",
            sph_paths,
            rlt_path(ckpts["lag"], "500_rho"),
            rlt_path(ckpts["lag"], "500_rho101"),
            rlt_path(ckpts["lag"], "500_rho10"),
            # rlt_path(ckpts["lag"], "500_rhotvf1"),
            # rlt_path(ckpts["lag"], "500_rhotvf2"),
            # rlt_path(ckpts["lag"], "500_rho_nu"),
            # rlt_path(ckpts["lag"], "500_rho_artif1"),
            # rlt_path(ckpts["lag"], "500_rho_sph2"),
            # rlt_path(ckpts["lag_noisy"], "1999_nsph1"),
            # rlt_path(ckpts["lag_noisy"], "1999_nsph1tvf1"),
        ]
        names = [
            "Dataset",  # used only to get the metadata
            "SPH",
            r"$\mathbf{v}\to \mathbf{u}$ + rlx",
            r"$\mathbf{v}\to \mathbf{u}$ + rlx101",
            r"$\mathbf{v}\to \mathbf{u}$ + rlx10",
            # r"$\mathbf{v}\to \mathbf{u}$ + rlx_tvf1",
            # r"$\mathbf{v}\to \mathbf{u}$ + rlx_tvf2",
            # r"$\mathbf{v}\to \mathbf{u}$ + rlx_nu",
            # r"$\mathbf{v}\to \mathbf{u}$ + rlx_artif1",
            # r"$\mathbf{v}\to \mathbf{u}$ + rlx_sph",
            # r"$\mathbf{v}\to \mathbf{u}$ + noise + NSPH",
            # r"$\mathbf{v}\to \mathbf{u}$ + noise + NSPHtvf1",
        ]
    else:
        paths = [
            f"{data_root}/data/2D_KOLM_4096_200kevery1",
            # "/home/atoshev/code/sph-turbulence/gen_dataset/data/2D_KOLM_SPH_0_20250210-232517_SPH_15",
            sph_paths,
            # rlt_path(ckpts["simple_rlx"], rlt_type),
        ]
        names = [
            "Dataset",  # used only to get the metadata
            # "SPH",
            "SPH",
            "Simple-Base",
            # "TVF-Base",
            # r"Simple-$\mathbf{u}$",
            # "TVF-Rlx" if is_every1 else "TVF-Rlx-Closure",
        ]

    return paths, names


# plt_ekin_and_spectra("lag_1001_nsph", every_n=1, step_last=1000, step_stride=50)  # TODO: still needed?
# plt_ekin_and_spectra("lag_101_v2u_nsph", every_n=1, step_last=100, step_stride=10)
# plt_ekin_and_spectra("lag_1001_v2u_nsph", every_n=1, step_last=1000, step_stride=50)
# plt_ekin_and_spectra("lag_5001_v2u_nsph", every_n=1, step_last=5000, step_stride=100)
# plt_ekin_and_spectra("lag_101_v2u_nsph1tvf", every_n=1, step_last=100, step_stride=10)
# plt_ekin_and_spectra("lag_20000_v2u_nsph", every_n=1, step_last=19999, step_stride=100)

# plt_ekin_and_spectra("lag10_101", every_n=10, step_last=100, step_stride=10)
# plt_ekin_and_spectra("lag10_500", every_n=10, step_last=500, step_stride=10)
