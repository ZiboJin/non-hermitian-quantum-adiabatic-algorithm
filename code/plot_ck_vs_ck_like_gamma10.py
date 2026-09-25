"""Plot CK and CK-like edge-deletion scaling curves from cached CSV files."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd


CK_LIKE_LINESTYLE = (0, (3.0, 1.6))
MAIN_AXES_RECT = (0.12, 0.20, 0.84, 0.92)


ALGORITHM_STYLE = {
    "NHMIS-FKQAA": {
        "label": "FK QAA",
        "color": "#3C5488",
        "marker": "o",
        "axis": "left",
        "ck_column": "p_FK",
    },
    "NHMIS-HDQAA": {
        "label": "HD QAA",
        "color": "#E64B35",
        "marker": "o",
        "axis": "left",
        "ck_column": "p_NHHD",
    },
    "HMIS-HDQAA": {
        "label": "H QAA",
        "color": "#00A087",
        "marker": "o",
        "axis": "right",
        "ck_column": "p_HHD",
    },
}


PALETTES = {
    "npg_classic": {
        "NHMIS-FKQAA": "#3C5488",
        "NHMIS-HDQAA": "#E64B35",
        "HMIS-HDQAA": "#00A087",
    },
    "npg_bright": {
        "NHMIS-FKQAA": "#4DBBD5",
        "NHMIS-HDQAA": "#E64B35",
        "HMIS-HDQAA": "#00A087",
    },
    "npg_muted": {
        "NHMIS-FKQAA": "#8491B4",
        "NHMIS-HDQAA": "#F39B7F",
        "HMIS-HDQAA": "#91D1C2",
    },
    "npg_deep": {
        "NHMIS-FKQAA": "#3C5488",
        "NHMIS-HDQAA": "#DC0000",
        "HMIS-HDQAA": "#7E6148",
    },
    "npg_balanced": {
        "NHMIS-FKQAA": "#4DBBD5",
        "NHMIS-HDQAA": "#DC0000",
        "HMIS-HDQAA": "#00A087",
    },
    "nature_colorblind": {
        "NHMIS-FKQAA": "#0072B2",
        "NHMIS-HDQAA": "#D55E00",
        "HMIS-HDQAA": "#009E73",
    },
    "prx_balanced": {
        "NHMIS-FKQAA": "#3366AA",
        "NHMIS-HDQAA": "#CC3311",
        "HMIS-HDQAA": "#009988",
    },
    "prx_neon": {
        "NHMIS-FKQAA": "#00A6D6",
        "NHMIS-HDQAA": "#E84A5F",
        "HMIS-HDQAA": "#D4A017",
    },
    "prl_cover": {
        "NHMIS-FKQAA": "#3B5BA9",
        "NHMIS-HDQAA": "#E83F2E",
        "HMIS-HDQAA": "#D79B00",
    },
    "prl_muted": {
        "NHMIS-FKQAA": "#4C72B0",
        "NHMIS-HDQAA": "#DD8452",
        "HMIS-HDQAA": "#55A868",
    },
    "prxq_dark": {
        "NHMIS-FKQAA": "#4CC9F0",
        "NHMIS-HDQAA": "#F72585",
        "HMIS-HDQAA": "#F9C80E",
    },
    "prxq_cool": {
        "NHMIS-FKQAA": "#4361EE",
        "NHMIS-HDQAA": "#B5179E",
        "HMIS-HDQAA": "#06D6A0",
    },
}


def _enable_miktex_text_rendering() -> None:
    """Use the manuscript's Matplotlib text settings without machine-specific caches."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "text.usetex": False,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "axes.unicode_minus": False,
    })



def set_palette(name: str) -> None:
    try:
        colors = PALETTES[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PALETTES))
        raise ValueError(f"unknown palette {name!r}; choose one of: {choices}") from exc
    for algorithm, color in colors.items():
        ALGORITHM_STYLE[algorithm]["color"] = color


def _positive(values: pd.Series | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.where(array > 0.0, array, np.nan)


def _plot_ck_lines(ck_csv: str, ax_left, ax_right, n_max: int) -> list:
    ck = pd.read_csv(ck_csv).sort_values("n")
    ck = ck[ck["n"] <= n_max].copy()
    handles = []

    for algorithm, style in ALGORITHM_STYLE.items():
        column = style["ck_column"]
        if column not in ck:
            continue
        y = _positive(ck[column])
        valid = np.isfinite(y)
        if not np.any(valid):
            continue
        axis = ax_left if style["axis"] == "left" else ax_right
        line = axis.plot(
            ck.loc[valid, "n"].to_numpy(dtype=int),
            y[valid],
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=4.0,
            markersize=9.6,
            label=f"CK {style['label']}",
        )[0]
        handles.append(line)

    return handles


def _plot_ck_like_lines(ck_like_summary_csv: str, ax_left, ax_right, n_max: int) -> list:
    summary = pd.read_csv(ck_like_summary_csv).sort_values(["algorithm", "n"])
    summary = summary[summary["n"] <= n_max].copy()
    handles = []

    for algorithm, style in ALGORITHM_STYLE.items():
        sub = summary[summary["algorithm"] == algorithm].sort_values("n")
        if sub.empty:
            continue
        x = sub["n"].to_numpy(dtype=int)
        median = _positive(sub["median_success"])
        ymin = _positive(sub["min_success"])
        ymax = _positive(sub["max_success"])
        valid = np.isfinite(median)
        if not np.any(valid):
            continue

        axis = ax_left if style["axis"] == "left" else ax_right
        line = axis.plot(
            x[valid],
            median[valid],
            color=style["color"],
            marker=style["marker"],
            linestyle=CK_LIKE_LINESTYLE,
            linewidth=4.0,
            markersize=9.6,
            label=f"CK-like {style['label']}",
        )[0]
        envelope = np.isfinite(ymin) & np.isfinite(ymax)
        if np.any(envelope):
            axis.fill_between(
                x[envelope],
                ymin[envelope],
                ymax[envelope],
                color=style["color"],
                alpha=0.16,
                linewidth=0,
            )
        handles.append(line)

    return handles


def _set_hmis_log_ticks(ax_right, ck_csv: str, ck_like_summary_csv: str, n_max: int) -> None:
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterMathtext

    positive_values: list[float] = []
    ck = pd.read_csv(ck_csv)
    ck = ck[ck["n"] <= n_max]
    if "p_HHD" in ck:
        positive_values.extend(_positive(ck["p_HHD"])[np.isfinite(_positive(ck["p_HHD"]))])

    summary = pd.read_csv(ck_like_summary_csv)
    summary = summary[(summary["n"] <= n_max) & (summary["algorithm"] == "HMIS-HDQAA")]
    for column in ["median_success", "min_success", "max_success"]:
        if column in summary:
            values = _positive(summary[column])
            positive_values.extend(values[np.isfinite(values)])

    lowest_power = -6
    if positive_values:
        lowest_power = min(lowest_power, int(np.floor(np.log10(min(positive_values)))))
    ticks = [10.0**k for k in range(0, lowest_power - 1, -1)]
    ticks = ticks[:7]
    ax_right.set_ylim(ticks[-1], 1.0)
    labeled_ticks = set(ticks[::2])

    ax_right.yaxis.set_major_locator(FixedLocator(ticks))
    ax_right.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _pos: LogFormatterMathtext(base=10)(value)
            if value in labeled_ticks
            else ""
        )
    )


def _set_graph_size_ticks(ax_left, ck_csv: str, ck_like_summary_csv: str, n_max: int) -> None:
    from matplotlib.ticker import FixedLocator

    values: set[int] = set()
    ck = pd.read_csv(ck_csv)
    if "n" in ck:
        values.update(int(n) for n in ck.loc[ck["n"] <= n_max, "n"])
    summary = pd.read_csv(ck_like_summary_csv)
    if "n" in summary:
        values.update(int(n) for n in summary.loc[summary["n"] <= n_max, "n"])
    ticks = [9, 21, 33, 45]
    if not ticks:
        return

    ax_left.xaxis.set_major_locator(FixedLocator(ticks))
    minor_ticks = sorted(values)
    ax_left.xaxis.set_minor_locator(FixedLocator(minor_ticks))


def _legend_handles():
    from matplotlib.lines import Line2D

    handles = []
    for _, style in ALGORITHM_STYLE.items():
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle="-",
                linewidth=2.4,
                label=f"CK {style['label']}",
            )
        )
    for _, style in ALGORITHM_STYLE.items():
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=CK_LIKE_LINESTYLE,
                linewidth=2.4,
                label=f"CK-like {style['label']}",
            )
        )
    return handles


def plot_ck_vs_ck_like(
    ck_csv: str,
    ck_like_summary_csv: str,
    png: str,
    pdf: str | None = None,
    n_max: int = 45,
    dpi: int = 300,
    figsize: tuple[float, float] = (10.4, 7.4),
):
    """Save a combined CK/CK-like plot with NHMIS on the left axis and HMIS on the right."""
    import matplotlib.pyplot as plt

    _enable_miktex_text_rendering()
    fig, ax_left = plt.subplots(figsize=figsize)
    ax_right = ax_left.twinx()

    _plot_ck_lines(ck_csv, ax_left, ax_right, n_max)
    _plot_ck_like_lines(ck_like_summary_csv, ax_left, ax_right, n_max)

    tick_size = 28
    ax_left.set_ylim(0.0, 1.05)
    from matplotlib.ticker import FixedLocator, FuncFormatter

    left_ticks = np.linspace(0.0, 1.0, 7)
    left_labeled_ticks = set(left_ticks[::2])
    ax_left.yaxis.set_major_locator(FixedLocator(left_ticks))
    ax_left.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _pos: f"{value:.1f}" if value in left_labeled_ticks else "")
    )
    _set_graph_size_ticks(ax_left, ck_csv, ck_like_summary_csv, n_max)
    ax_left.tick_params(axis="both", which="major", labelsize=tick_size, length=6)
    ax_left.tick_params(axis="both", which="minor", length=3)
    ax_right.tick_params(axis="both", which="major", labelsize=tick_size, length=6)
    ax_right.tick_params(axis="both", which="minor", length=3)
    ax_left.grid(False)
    ax_right.grid(False)
    ax_right.set_yscale("log")
    _set_hmis_log_ticks(ax_right, ck_csv, ck_like_summary_csv, n_max)

    fig.subplots_adjust(
        left=MAIN_AXES_RECT[0],
        bottom=MAIN_AXES_RECT[1],
        right=MAIN_AXES_RECT[2],
        top=MAIN_AXES_RECT[3],
    )
    out_dir = os.path.dirname(png)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.12)
    pdf_path = pdf or f"{os.path.splitext(png)[0]}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.12)
    return fig, png, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CK and CK-like scaling together.")
    parser.add_argument("--ck-csv", default="outputs/extended_scaling_gamma10/mixed_scaling_gamma10.csv")
    parser.add_argument(
        "--ck-like-summary-csv",
        default="outputs/ck_like_random_deletion_exp1_gamma10/ck_like_summary.csv",
    )
    parser.add_argument("--png", default="outputs/ck_vs_ck_like_gamma10/ck_vs_ck_like_gamma10.png")
    parser.add_argument("--pdf")
    parser.add_argument("--n-max", type=int, default=45)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--figsize", nargs=2, type=float, default=(10.4, 7.4))
    parser.add_argument("--palette", choices=sorted(PALETTES), default="prl_muted")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_palette(args.palette)
    fig, png, pdf = plot_ck_vs_ck_like(
        args.ck_csv,
        args.ck_like_summary_csv,
        args.png,
        pdf=args.pdf,
        n_max=args.n_max,
        dpi=args.dpi,
        figsize=(args.figsize[0], args.figsize[1]),
    )
    fig.clf()
    print(f"saved {png}")
    print(f"saved {pdf}")


if __name__ == "__main__":
    main()
