"""Compare umax traces for the three 2-D TGV production runs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_umax(run_dir: Path) -> tuple[pd.Series, pd.Series, pd.Series | None]:
    """Load time, umax, and optional uref from a run diagnostics file."""
    diag_path = run_dir / "diagnostics.csv"
    df = pd.read_csv(diag_path)
    uref = df["uref"]
    return df["time"].values, df["umax"].values, uref


if __name__ == "__main__":
    runs = [
        ("tgv2d_50_tvf", "dx=0.020, SPH", "--"),
        ("tgv2d_200_tvf", "dx=0.005, SPH", ":"),
        ("tgv2d_50_notvf", "dx=0.020, standard SPH", "-."),
    ]

    fig, ax = plt.subplots(figsize=(4.5, 3.5), layout="constrained")

    res_dir = Path("res")
    for run_name, label, linestyle in runs:
        run_dir = res_dir / run_name
        t, umax, uref = load_umax(run_dir)
        if run_name == "tgv2d_50_tvf":
            ax.plot(t, uref, "k", linewidth=1, label="Theory")
        ax.plot(t, umax, linestyle, label=label)

    ax.set_xlabel("Time")
    ax.set_ylabel(r"$U_{\max}$")
    ax.set_yscale("log")
    ax.set_ylim(0.001, 1.2)
    ax.set_xlim(0, 6)
    ax.legend()

    fig.savefig(res_dir / "tgv2d_umax_production.png", dpi=300)
    fig.savefig(res_dir / "tgv2d_umax_production.pdf", dpi=300)
    plt.close(fig)

    print("Finished plotting.")
