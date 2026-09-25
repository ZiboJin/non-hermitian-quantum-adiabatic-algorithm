from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reproduced"
DATA_DIR = ROOT / "data" / "figure6"
EPSILON_VIS = 1e-32
LOG_EPSILON_VIS = math.log10(EPSILON_VIS)

BLUE = "#4C72B0"
TERRACOTTA = "#DD8452"


def read_metadata() -> list[dict[str, str]]:
    path = DATA_DIR / "contours_metadata.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def row_for(
    rows: list[dict[str, str]],
    algorithm: str,
    center_type: str,
    r: int | str | None = None,
) -> dict[str, str]:
    for row in rows:
        if row["algorithm"] != algorithm or row["center_type"] != center_type:
            continue
        if r is not None and row["r"] != str(r):
            continue
        return row
    raise KeyError((algorithm, center_type, r))


def contour_arrays() -> np.lib.npyio.NpzFile:
    return np.load(DATA_DIR / "contours.npz", allow_pickle=True)


def resolve_cache_path(row: dict[str, str]) -> Path | None:
    value = row.get("cache_path", "").strip()
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def close_polygon(coords: np.ndarray) -> np.ndarray:
    if len(coords) < 2:
        return coords
    if np.linalg.norm(coords[0] - coords[-1]) <= 1e-14 * max(1.0, float(np.max(np.abs(coords)))):
        return coords
    return np.vstack([coords, coords[0]])


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


def scaled_tick_label(value: float, scale: float) -> str:
    scaled = value / scale
    if abs(scaled) < 5e-15:
        scaled = 0.0
    return f"{scaled:.3g}"


def endpoint_axis_ticks(limits: tuple[float, float], extra_ticks: list[float] | None = None) -> list[float]:
    lo, hi = limits
    mid = 0.0 if lo < 0.0 < hi else 0.5 * (lo + hi)
    ticks = [lo, mid, hi]
    if extra_ticks:
        ticks.extend(tick for tick in extra_ticks if lo <= tick <= hi)
    out: list[float] = []
    min_sep = 1e-8 * max(np.finfo(float).tiny, abs(hi - lo))
    for tick in sorted(ticks):
        if not out or abs(tick - out[-1]) > min_sep:
            out.append(float(tick))
    return out


def set_scaled_endpoint_ticks(
    ax: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    scale: float,
    labelsize: float = 6.0,
    x_extra_ticks: list[float] | None = None,
) -> None:
    xticks = endpoint_axis_ticks(xlim, x_extra_ticks)
    yticks = endpoint_axis_ticks(ylim)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    for idx, tick in enumerate(xticks):
        x_span = max(np.finfo(float).tiny, xlim[1] - xlim[0])
        is_zero = abs(tick) < 1e-12 * x_span
        ha = "left" if idx == 0 else "right" if idx == len(xticks) - 1 else "center"
        y_pos = 0.026 if is_zero else -0.040
        va = "bottom" if is_zero else "top"
        if idx > 0 and not is_zero:
            prev_tick = xticks[idx - 1]
            close_to_prev = abs(tick - prev_tick) < 0.18 * x_span
            if close_to_prev:
                y_pos = -0.125
        ax.text(
            tick,
            y_pos,
            scaled_tick_label(tick, scale),
            transform=ax.get_xaxis_transform(),
            ha=ha,
            va=va,
            fontsize=labelsize,
            clip_on=False,
        )

    for idx, tick in enumerate(yticks):
        y_span = max(np.finfo(float).tiny, ylim[1] - ylim[0])
        is_zero = abs(tick) < 1e-12 * y_span
        va = "center" if is_zero else "bottom" if idx == 0 else "top" if idx == len(yticks) - 1 else "center"
        x_pos = 0.026 if is_zero else -0.035
        ha = "left" if is_zero else "right"
        ax.text(
            x_pos,
            tick,
            scaled_tick_label(tick, scale),
            transform=ax.get_yaxis_transform(),
            ha=ha,
            va=va,
            fontsize=labelsize,
            clip_on=False,
        )


def scale_for_limits(xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple[float, int]:
    magnitude = max(abs(xlim[0]), abs(xlim[1]), abs(ylim[0]), abs(ylim[1]), 1e-300)
    exponent = int(math.floor(math.log10(magnitude)))
    return 10.0**exponent, exponent


def style_small_axis(
    ax: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str = "Re",
    ylabel: str = "Im",
    x_extra_ticks: list[float] | None = None,
) -> None:
    span = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    xmid = 0.5 * (xlim[0] + xlim[1])
    ymid = 0.5 * (ylim[0] + ylim[1])
    half = 0.52 * span
    xlim_equal = (xmid - half, xmid + half)
    ylim_equal = (ymid - half, ymid + half)
    ax.set_xlim(*xlim_equal)
    ax.set_ylim(*ylim_equal)
    ax.set_aspect("equal", adjustable="box")

    scale, exponent = scale_for_limits(xlim_equal, ylim_equal)
    ax.set_xlabel(xlabel, fontsize=6.7, labelpad=10.0)
    ax.set_ylabel(ylabel, fontsize=6.7, labelpad=8.0)
    ax.text(
        0.97,
        0.95,
        rf"$\times10^{{{exponent}}}$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.2,
        color="0.30",
    )
    ax.tick_params(axis="both", labelsize=6.0, pad=1.0, length=2.0)
    set_scaled_endpoint_ticks(ax, xlim_equal, ylim_equal, scale, x_extra_ticks=x_extra_ticks)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.85)


def local_axis_labels(center_type: str) -> tuple[str, str]:
    if center_type == "E0":
        return r"$\mathrm{Re}(z-E_0)$", r"$\mathrm{Im}(z-E_0)$"
    if center_type == "E1":
        return r"$\mathrm{Re}(z-E_1)$", r"$\mathrm{Im}(z-E_1)$"
    return r"$\mathrm{Re}\,z$", r"$\mathrm{Im}\,z$"


def draw_center_marker(ax: plt.Axes, center_type: str, x: float = 0.0, y: float = 0.0) -> None:
    marker = "s" if center_type == "E0" else "^"
    ax.scatter(
        [x],
        [y],
        marker=marker,
        s=19,
        facecolor="white",
        edgecolor=BLUE,
        linewidth=0.85,
        zorder=5,
    )


def draw_contour_fill(
    ax: plt.Axes,
    row: dict[str, str],
    arrays: np.lib.npyio.NpzFile,
    title: str,
    center_type: str,
) -> None:
    coords = contour_xy(row, arrays)
    coords = close_polygon(coords)
    draw_polygon_fill(ax, coords, alpha=0.17, linewidth=1.05)
    if center_type == "connected_E0_E1":
        ax.scatter([0.0], [0.0], marker="s", s=18, facecolor="white", edgecolor=BLUE, linewidth=0.85, zorder=5)
        ax.scatter(
            [float(row["E1"])],
            [0.0],
            marker="^",
            s=22,
            facecolor="white",
            edgecolor=BLUE,
            linewidth=0.85,
            zorder=5,
        )
    else:
        draw_center_marker(ax, center_type)

    ax.set_title(title, fontsize=7.1, fontweight="semibold", pad=2.0)
    xlabel, ylabel = local_axis_labels(center_type)
    x_extra_ticks = [0.0, float(row["E1"])] if center_type == "connected_E0_E1" else None
    style_small_axis(
        ax,
        (float(coords[:, 0].min()), float(coords[:, 0].max())),
        (float(coords[:, 1].min()), float(coords[:, 1].max())),
        xlabel=xlabel,
        ylabel=ylabel,
        x_extra_ticks=x_extra_ticks,
    )


def contour_xy(row: dict[str, str], arrays: np.lib.npyio.NpzFile) -> np.ndarray:
    coords = np.asarray(arrays[row["contour_key"]], dtype=float)
    if row.get("contour_coordinate_frame") == "local_scaled_by_epsilon":
        coords = coords * float(row["epsilon_vis"])
    return coords


def draw_polygon_fill(
    ax: plt.Axes,
    coords: np.ndarray,
    alpha: float,
    linewidth: float = 1.05,
    linestyle: str = "-",
    line_alpha: float = 1.0,
    zorder: int = 2,
) -> None:
    coords = close_polygon(coords)
    ax.fill(coords[:, 0], coords[:, 1], color=BLUE, alpha=alpha, linewidth=0, zorder=zorder)
    ax.plot(
        coords[:, 0],
        coords[:, 1],
        color=BLUE,
        lw=linewidth,
        linestyle=linestyle,
        alpha=line_alpha,
        zorder=zorder + 1,
    )


def draw_h_hd_overlay(
    ax: plt.Axes,
    rows: list[dict[str, str]],
    arrays: np.lib.npyio.NpzFile,
    center_type: str,
    title: str,
) -> None:
    hd = row_for(rows, "NHMIS-HDQAA", center_type)
    h = row_for(rows, "HMIS-HDQAA", center_type)
    hd_xy = contour_xy(hd, arrays)
    h_xy = contour_xy(h, arrays)
    draw_polygon_fill(ax, hd_xy, alpha=0.13, linewidth=1.20, linestyle="-", line_alpha=1.0, zorder=2)
    draw_polygon_fill(ax, h_xy, alpha=0.14, linewidth=0.85, linestyle="-", line_alpha=0.72, zorder=4)
    draw_center_marker(ax, center_type)
    all_xy = np.vstack([hd_xy, h_xy])
    ax.set_title(title, fontsize=7.1, fontweight="semibold", pad=2.0)
    xlabel, ylabel = local_axis_labels(center_type)
    style_small_axis(
        ax,
        (float(all_xy[:, 0].min()), float(all_xy[:, 0].max())),
        (float(all_xy[:, 1].min()), float(all_xy[:, 1].max())),
        xlabel=xlabel,
        ylabel=ylabel,
    )


def draw_left_panel(fig: plt.Figure, spec, rows: list[dict[str, str]]) -> None:
    arrays = contour_arrays()
    sub = spec.subgridspec(
        4,
        4,
        height_ratios=[0.14, 1.0, 1.0, 1.0],
        wspace=0.36,
        hspace=0.58,
    )
    title_ax = fig.add_subplot(sub[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.0,
        0.58,
        r"$\varepsilon=10^{-32}$ pseudospectrum",
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="normal",
    )

    slots: list[tuple[int, int, dict[str, str], str, str]] = []
    for idx, r in enumerate([1, 2]):
        slots.append((1, 2 * idx, row_for(rows, "NHMIS-FKQAA-full-direct-sum", "E0", r), f"FK QAA r={r} E0", "E0"))
        slots.append((1, 2 * idx + 1, row_for(rows, "NHMIS-FKQAA-full-direct-sum", "E1", r), f"FK QAA r={r} E1", "E1"))
    for idx, r in enumerate([3, 4]):
        slots.append((2, 2 * idx, row_for(rows, "NHMIS-FKQAA-full-direct-sum", "E0", r), f"FK QAA r={r} E0", "E0"))
        slots.append((2, 2 * idx + 1, row_for(rows, "NHMIS-FKQAA-full-direct-sum", "E1", r), f"FK QAA r={r} E1", "E1"))

    for row_idx, col_idx, row, title, center_type in slots:
        ax = fig.add_subplot(sub[row_idx, col_idx])
        draw_contour_fill(ax, row, arrays, title, center_type)

    ax_r5 = fig.add_subplot(sub[3, 0:2])
    draw_contour_fill(
        ax_r5,
        row_for(rows, "NHMIS-FKQAA-full-direct-sum", "connected_E0_E1", 5),
        arrays,
        "FK QAA r=5 E0/E1",
        "connected_E0_E1",
    )

    ax_hd_h_e0 = fig.add_subplot(sub[3, 2])
    draw_h_hd_overlay(ax_hd_h_e0, rows, arrays, "E0", "H QAA + HD QAA E0\nr=1,2,3,4,5")

    ax_hd_h_e1 = fig.add_subplot(sub[3, 3])
    draw_h_hd_overlay(ax_hd_h_e1, rows, arrays, "E1", "H QAA + HD QAA E1\nr=1,2,3,4,5")
    legend_pos = ax_hd_h_e1.get_position()
    fig.legend(
        handles=[
            Line2D([0], [0], color=BLUE, lw=1.45, linestyle="-", alpha=1.0, label="HD QAA"),
            Line2D([0], [0], color=BLUE, lw=0.95, linestyle="-", alpha=0.72, label="H QAA"),
        ],
        loc="center left",
        bbox_to_anchor=(legend_pos.x1 + 0.012, legend_pos.y0 + 0.57 * legend_pos.height),
        bbox_transform=fig.transFigure,
        frameon=False,
        fontsize=7.4,
        handlelength=2.0,
        borderpad=0.2,
        ncol=1,
    )

    arrays.close()


def draw_threshold_panel(ax: plt.Axes) -> None:
    df = pd.read_csv(DATA_DIR / "thresholds_numerical.csv").sort_values("r")
    ax.plot(
        df["r"],
        df["log10_epsilon_c_grid_hp"],
        color=BLUE,
        marker="o",
        markerfacecolor=BLUE,
        markeredgecolor=BLUE,
        lw=2.2,
        ms=5.0,
        linestyle="-",
    )
    ax.plot(
        df["r"],
        df["log10_epsilon_est_global"],
        color=TERRACOTTA,
        marker="s",
        markerfacecolor=TERRACOTTA,
        markeredgecolor=TERRACOTTA,
        markeredgewidth=1.0,
        linestyle="-",
        lw=2.1,
        ms=4.8,
    )
    ax.set_title(
        r"FK QAA gap-closing noise strength $\varepsilon$ threshold",
        loc="left",
        fontweight="normal",
        fontsize=11.5,
    )
    ax.set_xlabel(r"$r$  $(L=14r)$")
    ax.set_ylabel(r"$\log_{10}\varepsilon$")
    ax.set_xticks(df["r"])
    ax.set_xlim(0.78, 5.22)
    y_values = pd.concat([df["log10_epsilon_c_grid_hp"], df["log10_epsilon_est_global"]], ignore_index=True)
    y_ticks = endpoint_yticks(y_values, step=5.0)
    y_min = float(y_values.min())
    y_max = float(y_values.max())
    ax.set_ylim(y_min - 0.8, y_max + 0.8)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(endpoint_tick_labels(y_ticks, [y_min, y_max]))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.85)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=BLUE,
                markeredgecolor=BLUE,
                markersize=5.5,
                label="Numerical grid search",
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="none",
                markerfacecolor=TERRACOTTA,
                markeredgecolor=TERRACOTTA,
                markersize=5.2,
                label="Theoretical prediction",
            ),
        ],
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        fontsize=8.5,
    )
    ax.tick_params(axis="both", labelsize=8.5)
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
    png = FIG_DIR / "fig2_two_panel_pseudospectrum_threshold_redraw.png"
    pdf = FIG_DIR / "fig2_two_panel_pseudospectrum_threshold_redraw.pdf"
    fig.savefig(png, dpi=280)
    fig.savefig(pdf)
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
