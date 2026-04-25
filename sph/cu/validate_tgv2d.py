"""Compare umax traces for the three 2-D TGV production runs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_data(run_dir: Path) -> tuple[pd.Series, pd.Series, pd.Series | None]:
    """Load time, umax, and optional uref from a run diagnostics file."""
    diag_path = run_dir / "diagnostics.csv"
    df = pd.read_csv(diag_path)
    return [df[col].values for col in ["time", "umax", "vmax", "uref"]]


if __name__ == "__main__":
    runs = [
        ("tgv2d_100_tvf", "SPH"),
        ("withA/tgv2d_100_tvf", r"SPH$_A$"),
        ("tgv2d_100_notvf", "standard SPH"),
    ]
    plt.rcParams.update({"font.size": 12})
    fig, ax = plt.subplots(figsize=(4.5, 3.5), layout="constrained")

    res_dir = Path("res")
    for i, (run_name, label) in enumerate(runs):
        run_dir = res_dir / run_name
        t, umax, vmax, uref = load_data(run_dir)
        if run_name == "tgv2d_100_tvf":
            ax.plot(t, uref, "k", label="Theory")
        ax.plot(t, umax, f"C{i}--", linewidth=2.5, label=r"$u$, " + label)
        if run_name != "tgv2d_100_notvf":
            ax.plot(t, vmax, f"C{i}:", linewidth=2.5, label=r"$v$, " + label)

    ax.set_xlabel("Time")
    ax.set_ylabel("Largest velocity magnitude")
    ax.set_yscale("log")
    ax.set_ylim(0.001, 1.2)
    ax.set_xlim(0, 6)
    ax.legend(frameon=False)

    fig.savefig(res_dir / "tgv2d_umax.png", dpi=300)
    fig.savefig(res_dir / "tgv2d_umax.pdf", dpi=300)
    plt.close(fig)

    print("Finished plotting.")
