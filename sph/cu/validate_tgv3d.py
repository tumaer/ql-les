"""Compare umax traces for the three 2-D TGV production runs."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(run_dir: Path) -> tuple[pd.Series, pd.Series, pd.Series | None]:
    """Load time, umax, and optional uref from a run diagnostics file."""
    diag_path = run_dir / "diagnostics.csv"
    df = pd.read_csv(diag_path)

    stride = max(len(df) // 50, 1)  # use 50 sample points for plotting
    df_sample = df.iloc[::stride]
    t = df_sample["time"].to_numpy()
    dt = np.diff(t)
    # normalize by domain volume
    epsilon = -np.diff(df_sample["ekin"].to_numpy()) / dt / (2 * np.pi) ** 3
    epsilon_v = -np.diff(df_sample["ekinv"].to_numpy()) / dt / (2 * np.pi) ** 3
    t_mid = t[:-1] + dt / 2
    return t_mid, epsilon, epsilon_v


if __name__ == "__main__":
    runs = [
        ("tgv3d_64_tvf", "SPH"),
        ("withA/tgv3d_64_tvf", r"SPH$_A$"),
    ]
    df_ref = pd.read_csv("ref_tgv3d_Re100.csv")

    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(4.5, 3.5), layout="constrained")

    ax.plot(df_ref["time"].values, df_ref["epsilon"].values, "k", label="DNS")

    res_dir = Path("res")
    for i, (run_name, label) in enumerate(runs):
        run_dir = res_dir / run_name
        t, epsilon, epsilon_v = load_data(run_dir)
        ax.plot(t, epsilon, f"C{i}--", linewidth=2.5, label=r"$u$, " + label)
        ax.plot(t, epsilon_v, f"C{i}:", linewidth=2.5, label=r"$v$, " + label)

    # ax.plot(t_mid, epsilon, "ok", fillstyle="none", label="SPH", markersize=8, lw=0.1)

    ax.set_xlabel("Time")
    ax.set_ylabel(r"Dissipation rate $\epsilon$")
    ax.legend(frameon=False)
    ax.set_ylim(0.004, 0.015)
    ax.set_xlim(-0.2, 10.2)
    ax.set_yticks([0.005, 0.010, 0.015])
    fig.savefig("res/tgv3d_epsilon.png", dpi=300)
    fig.savefig("res/tgv3d_epsilon.pdf", dpi=300)
    plt.close(fig)

    print("Finished plotting.")
