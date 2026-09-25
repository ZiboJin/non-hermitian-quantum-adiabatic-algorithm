"""Plot mixed-scaling data from an existing CSV only.

This module intentionally does not import any solver or instance-preparation
code.  It is safe to use for plot-style iteration without recomputation.
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd


def plot_mixed_scaling_from_csv(
    csv: str,
    plot: str,
    pdf: str | None = None,
    dpi: int = 300,
    figsize: tuple[float, float] = (7.0, 4.8),
    legend_loc: str = "best",
    left_ylabel: str = "NHMIS success probability",
    right_ylabel: str = "HMIS-HDQAA success probability",
):
    """Read a cached mixed-scaling CSV and save PNG/PDF plots."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    df = pd.read_csv(csv).sort_values("n")
    x = df["n"].to_numpy(dtype=int)

    def positive(column: str) -> np.ndarray:
        if column not in df:
            return np.full(len(df), np.nan)
        values = df[column].to_numpy(dtype=float)
        return np.where(values > 0.0, values, np.nan)

    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()

    line_fk = ax_left.plot(
        x,
        positive("p_FK"),
        marker="o",
        linewidth=1.8,
        label="NHMIS-FKQAA",
    )[0]
    line_nhhd = ax_left.plot(
        x,
        positive("p_NHHD"),
        marker="s",
        linewidth=1.8,
        label="NHMIS-HDQAA",
    )[0]
    line_hhd = ax_right.plot(
        x,
        positive("p_HHD"),
        marker="^",
        linestyle="--",
        linewidth=1.8,
        color="tab:green",
        label="HMIS-HDQAA",
    )[0]

    ax_left.set_xlabel(r"CK graph size $n$")
    ax_left.set_ylabel(left_ylabel)
    ax_right.set_ylabel(right_ylabel)
    ax_left.set_yscale("log")
    ax_right.set_yscale("log")
    ax_left.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_left.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax_left.legend(
        [line_fk, line_nhhd, line_hhd],
        ["NHMIS-FKQAA", "NHMIS-HDQAA", "HMIS-HDQAA"],
        loc=legend_loc,
    )
    fig.tight_layout()

    out_dir = os.path.dirname(plot)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(plot, dpi=dpi)
    pdf_path = pdf or f"{os.path.splitext(plot)[0]}.pdf"
    fig.savefig(pdf_path)
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot mixed scaling from CSV only.")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--plot", required=True)
    parser.add_argument("--pdf")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figsize", nargs=2, type=float, default=(7.0, 4.8))
    parser.add_argument("--legend-loc", default="best")
    parser.add_argument("--left-ylabel", default="NHMIS success probability")
    parser.add_argument("--right-ylabel", default="HMIS-HDQAA success probability")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fig = plot_mixed_scaling_from_csv(
        args.csv,
        args.plot,
        pdf=args.pdf,
        dpi=args.dpi,
        figsize=(args.figsize[0], args.figsize[1]),
        legend_loc=args.legend_loc,
        left_ylabel=args.left_ylabel,
        right_ylabel=args.right_ylabel,
    )
    fig.clf()
    print(f"saved {args.plot}")
    print(f"saved {args.pdf or os.path.splitext(args.plot)[0] + '.pdf'}")


if __name__ == "__main__":
    main()
