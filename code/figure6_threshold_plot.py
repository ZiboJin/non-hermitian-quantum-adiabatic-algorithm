from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from experiment_two_resolvent_pseudospectrum import (
    bitstring_label,
    build_ck_graph_from_n,
    eigenvalue_condition_estimate_for_sector,
    history_sector_data_for_r,
)
from figure6_plot_utils import (
    BLUE,
    DATA_DIR,
    FIG_DIR,
    TERRACOTTA,
    draw_left_panel,
    read_metadata,
)


def endpoint_yticks(values: pd.Series, step: float) -> list[float]:
    ymin = float(values.min())
    ymax = float(values.max())
    inner_min = np.ceil(ymin / step) * step
    inner_max = np.floor(ymax / step) * step
    inner = list(np.arange(inner_min, inner_max + 0.5 * step, step))
    ticks = [ymin, *inner, ymax]
    out: list[float] = []
    for tick in sorted(ticks):
        if not out or abs(tick - out[-1]) > 1e-9:
            out.append(float(tick))
    return out


def endpoint_tick_labels(ticks: list[float], endpoint_values: list[float]) -> list[str]:
    labels: list[str] = []
    for tick in ticks:
        if any(abs(tick - endpoint) < 1e-9 for endpoint in endpoint_values):
            labels.append(f"{tick:.1f}")
        else:
            labels.append(f"{tick:.0f}")
    return labels


def mis_estimates_by_r(r_values: list[int]) -> pd.DataFrame:
    mis_bitstring = bitstring_label(build_ck_graph_from_n(5)["x_mis"])
    rows: list[dict] = []
    for r in r_values:
        sectors = history_sector_data_for_r(r, n=5, p=2.0, q=4.0)
        sector = next(item for item in sectors if item["bitstring"] == mis_bitstring)
        est = eigenvalue_condition_estimate_for_sector(sector["logW"], chunk_size=128)
        rows.append(
            {
                "r": int(r),
                "n": 5,
                "m": 2,
                "L": len(sector["v"]),
                "sector_used": mis_bitstring,
                "log10_epsilon_est_mis": float(est["log10_epsilon_est"]),
                "k_star_est_mis": int(est["k_star_est"]),
            }
        )
    return pd.DataFrame(rows)


def annotate_point(
    ax: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    color: str,
    xytext: tuple[float, float],
    ha: str,
    va: str,
    fontsize: float,
    arrowprops: dict | None = None,
) -> None:
    ax.annotate(
        text,
        xy=(x, y),
        xytext=xytext,
        textcoords="offset points",
        ha=ha,
        va=va,
        fontsize=fontsize,
        color=color,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.74, "pad": 0.35},
        arrowprops=arrowprops,
        zorder=8,
    )


def draw_threshold_panel(ax: plt.Axes) -> None:
    df = pd.read_csv(DATA_DIR / "thresholds_numerical.csv").sort_values("r")
    mis = mis_estimates_by_r([int(r) for r in df["r"]])
    merged = df.merge(mis[["r", "log10_epsilon_est_mis", "k_star_est_mis", "sector_used"]], on="r", how="left")
    # Published input data remain read-only; derived estimates are returned in memory.

    ax.plot(
        merged["r"],
        merged["log10_epsilon_c_grid_hp"],
        color="black",
        marker="o",
        markerfacecolor="none",
        markeredgecolor="black",
        markeredgewidth=1.8,
        lw=2.35,
        ms=7.4,
        linestyle="-",
        zorder=7,
    )
    ax.plot(
        merged["r"],
        merged["log10_epsilon_est_mis"],
        color="black",
        marker="s",
        markerfacecolor="black",
        markeredgecolor="black",
        markeredgewidth=1.15,
        linestyle="--",
        lw=2.25,
        ms=4.8,
        zorder=5,
    )
    ax.set_title(
        r"FK QAA gap-closing noise strength $\varepsilon$ threshold",
        loc="left",
        fontweight="semibold",
        fontsize=13.5,
    )
    ax.set_xlabel(r"$r$", fontsize=14.0, labelpad=2.0)
    ax.set_ylabel(r"$\log_{10}\varepsilon$", fontsize=14.0, labelpad=2.0)
    ax.set_xticks(merged["r"])
    ax.set_xlim(0.78, 5.22)
    y_values = pd.concat([merged["log10_epsilon_c_grid_hp"], merged["log10_epsilon_est_mis"]], ignore_index=True)
    y_min = float(y_values.min())
    y_max = float(y_values.max())
    y_lower = int(np.floor(y_min))
    y_upper = int(np.ceil(y_max))
    y_ticks = np.linspace(float(y_lower), float(y_upper), 7)
    ax.set_ylim(y_lower, y_upper + 0.8)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{int(round(tick))}" if idx % 2 == 0 else "" for idx, tick in enumerate(y_ticks)])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.95)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=7.2,
                label="Numerical grid search",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="0.45",
                linestyle="--",
                markerfacecolor="white",
                markeredgecolor="0.45",
                markeredgewidth=1.35,
                markersize=7.0,
                label="Theoretical prediction",
            ),
        ],
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        fontsize=11.2,
        borderpad=0.45,
        handletextpad=0.6,
    )
    ax.tick_params(axis="both", labelsize=12.0, length=3.8, width=0.90, pad=2.0, direction="out")
    ax.set_box_aspect(1.0)
    ax.set_anchor("C")


def main() -> None:
    rows = read_metadata()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(17.2, 7.8), constrained_layout=False)
    fig.suptitle(r"CK graph ($m=2$, $n=5$)", y=0.985, fontsize=12.2, fontweight="normal")
    outer = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1.65, 1.0],
        left=0.045,
        right=0.985,
        bottom=0.075,
        top=0.915,
        wspace=0.10,
    )
    draw_left_panel(fig, outer[0, 0], rows)
    ax_right = fig.add_subplot(outer[0, 1])
    draw_threshold_panel(ax_right)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / "fig2_two_panel_pseudospectrum_threshold_mis_right_preview.png"
    pdf = FIG_DIR / "fig2_two_panel_pseudospectrum_threshold_mis_right_preview.pdf"
    fig.savefig(png, dpi=280)
    fig.savefig(pdf)
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
