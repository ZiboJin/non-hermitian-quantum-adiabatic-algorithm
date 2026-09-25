from __future__ import annotations

import math
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from figure6_plot_utils import (
    BLUE,
    DATA_DIR,
    EPSILON_VIS,
    FIG_DIR,
    close_polygon,
    contour_arrays,
    contour_xy,
    read_metadata,
    row_for,
)
from experiment_two_resolvent_pseudospectrum import (
    hd_block,
    history_sector_data_for_r,
    local_condition_scales_for_r,
)
from figure6_threshold_plot import annotate_point, draw_threshold_panel


ORANGE = "#DD8452"
CONDITION_COLOR = "#55A868"
HD_COLOR = ORANGE
H_COLOR = CONDITION_COLOR
ASINH_SCALE = 1e-29
H_DASH = (0, (16.0, 9.0))
LEGEND_H_DASH = (0, (1.15, 0.65))
THEORY_DASH_BY_LEVEL = {
    1: (0, (9.0, 5.0)),
    2: (0, (10.5, 5.8)),
    3: (0, (12.2, 6.8)),
    4: (0, (14.0, 8.0)),
}
THRESHOLD_DASH_POINTS = (14.985, 6.48)
AXIS_LABEL_FS = 51.0
TICK_FS = 57.0
PANEL_LABEL_FS = TICK_FS
LEGEND_FS = 8.4
ANNOT_FS = 9.2
THRESHOLD_ANNOT_FS = 10.8
ENERGY_LABEL_FS = 38.0
SCALE_TEXT_FS = 38.0
LINE_SCALE = 0.82
NUMERIC_CONTOUR_LW = 3.15
NUMERIC_DASHED_CONTOUR_LW = 3.35
MAJOR_TICK_LENGTH = 7.4
MAJOR_TICK_WIDTH = 1.35
MINOR_TICK_LENGTH = 4.2
MINOR_TICK_WIDTH = 0.90
ENERGY_POINT_S = 180 * LINE_SCALE


def center_label(value: float) -> str:
    if abs(value) < 5e-15:
        return "0"
    if abs(value - 1.0) < 5e-12:
        return "1"
    return f"{value:.4g}"


def integer_axis_params(half_width: float) -> tuple[float, float, list[float], list[str]]:
    scale = 10.0 ** int(math.floor(math.log10(max(half_width, 1e-300))))
    max_unit = int(math.ceil(half_width / scale))
    if max_unit <= 3:
        step = 1
    elif max_unit <= 6:
        step = 2
    else:
        step = 5
    tick_max = int(math.ceil(max_unit / step) * step)
    units = list(range(-tick_max, tick_max + step, step))
    ticks = [unit * scale for unit in units]
    labels = [str(unit) for unit in units]
    return tick_max * scale, scale, ticks, labels


def integer_ticks_for_range(lo: float, hi: float, scale: float) -> tuple[tuple[float, float], list[float], list[str]]:
    unit_min = int(math.floor(lo / scale))
    unit_max = int(math.ceil(hi / scale))
    ticks = [unit * scale for unit in range(unit_min, unit_max + 1)]
    labels = [str(unit) for unit in range(unit_min, unit_max + 1)]
    return (unit_min * scale, unit_max * scale), ticks, labels


def decimal_ticks_for_range(lo: float, hi: float, step: float, scale: float) -> tuple[list[float], list[str]]:
    first = math.ceil(lo / step) * step
    last = math.floor(hi / step) * step
    ticks = list(np.arange(first, last + 0.5 * step, step))
    labels = [f"{tick / scale:.2f}".rstrip("0").rstrip(".") for tick in ticks]
    return ticks, labels


def r5_x_ticks(lo: float, hi: float, scale: float, e0: float, e1: float) -> tuple[list[float], list[str]]:
    ticks, labels = decimal_ticks_for_range(lo, hi, 0.50 * scale, scale)
    special = [(e0, "0"), (e1, f"{e1 / scale:.0f}")]
    filtered = [
        (tick, label)
        for tick, label in zip(ticks, labels)
        if all(abs(tick - special_tick) > 0.12 * scale for special_tick, _ in special)
    ]
    merged = sorted([*filtered, *special], key=lambda item: item[0])
    return [tick for tick, _ in merged], [label for _, label in merged]


def padded_limits(lo: float, hi: float, pad_fraction: float) -> tuple[float, float]:
    span = hi - lo
    pad = pad_fraction * span
    return lo - pad, hi + pad


def square_limits(xlim: tuple[float, float], ylim: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    x_mid = 0.5 * (xlim[0] + xlim[1])
    y_mid = 0.5 * (ylim[0] + ylim[1])
    half = 0.5 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    return (x_mid - half, x_mid + half), (y_mid - half, y_mid + half)


def contour_half_width(contours: list[np.ndarray], pad: float = 0.05) -> float:
    max_abs = 0.0
    for coords in contours:
        max_abs = max(max_abs, float(np.max(np.abs(coords))))
    return max(max_abs * (1.0 + pad), 1e-300)


def radial_asinh(coords: np.ndarray, scale: float = ASINH_SCALE) -> np.ndarray:
    rho = np.hypot(coords[:, 0], coords[:, 1])
    mapped_rho = np.arcsinh(rho / scale)
    factor = np.divide(mapped_rho, rho, out=np.zeros_like(rho), where=rho > 0.0)
    return np.column_stack([coords[:, 0] * factor, coords[:, 1] * factor])


def true_scale_label(rho: float) -> str:
    exponent = int(round(math.log10(max(rho, 1e-300))))
    return rf"$10^{{{exponent}}}$"


def render_miktex_label_png(r: int, scale_text: str) -> Path:
    """Load the original typeset label; adjacent .tex files record its source.

    The four small label assets remove the runtime LaTeX/Poppler dependency.
    They contain text only and do not replace any numerical curves.
    """
    exponent = scale_text.strip("$").replace("{", "").replace("}", "").replace("^", "p")
    path = ROOT_DIR / "assets" / "figure6_labels" / f"r{r}_{exponent}.png"
    if not path.is_file():
        raise FileNotFoundError(f"Missing archived figure label: {path}")
    return path



def add_miktex_label(ax: plt.Axes, x: float, y: float, r: int, scale_text: str) -> None:
    img = plt.imread(render_miktex_label_png(r, scale_text))
    label = OffsetImage(img, zoom=0.44)
    artist = AnnotationBbox(
        label,
        (x, y),
        xycoords="data",
        frameon=False,
        pad=0.0,
        box_alignment=(0.5, 1.0),
        zorder=38,
    )
    ax.add_artist(artist)


def circle_xy(radius: float, center: complex = 0.0 + 0.0j, n: int = 361) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=True)
    z = complex(center) + float(radius) * (np.cos(angles) + 1j * np.sin(angles))
    return np.column_stack([np.real(z), np.imag(z)])


def eigenvalue_condition_number(H: np.ndarray, target: float) -> float:
    vals, vecs = np.linalg.eig(np.asarray(H, dtype=complex))
    lvals, lvecs = np.linalg.eig(np.asarray(H, dtype=complex).conj().T)
    i = int(np.argmin(np.abs(vals - complex(target))))
    j = int(np.argmin(np.abs(lvals - complex(target).conjugate())))
    v = vecs[:, i]
    w = lvecs[:, j]
    overlap = abs(np.vdot(w, v))
    if overlap == 0.0:
        return math.inf
    return float(np.linalg.norm(v) * np.linalg.norm(w) / overlap)


def hd_condition_radius(sigma: float, center_type: str) -> float:
    target = 0.0 if center_type == "E0" else 1.0
    return float(EPSILON_VIS * eigenvalue_condition_number(hd_block(sigma), target))


def draw_condition_circle(
    ax: plt.Axes,
    coords: np.ndarray,
    *,
    color: str = CONDITION_COLOR,
    linestyle: str | tuple = "-",
    alpha: float = 0.95,
    lw: float = 1.25,
    marker: str | None = None,
    markevery: int | None = None,
    zorder: float = 12,
) -> None:
    ax.plot(
        coords[:, 0],
        coords[:, 1],
        color=color,
        linestyle=linestyle,
        lw=lw * LINE_SCALE,
        alpha=alpha,
        marker=marker,
        markevery=markevery,
        markersize=6.0 * LINE_SCALE,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=0.0,
        zorder=zorder,
    )


def set_visual_dash(line: Line2D, dash_points: tuple[float, float]) -> None:
    """Set rendered dash lengths independent of the line's width."""
    linewidth = line.get_linewidth()
    line.set_linestyle((0, tuple(length / linewidth for length in dash_points)))


def hhd_legend_handles(color: str) -> list:
    return [
        Patch(facecolor=HD_COLOR, edgecolor="none", alpha=0.14, label="HD QAA pseudospectrum"),
        Patch(facecolor=H_COLOR, edgecolor="none", alpha=0.06, label="HM QAA pseudospectrum"),
        Line2D([0], [0], color=HD_COLOR, lw=NUMERIC_CONTOUR_LW * LINE_SCALE, linestyle="-", label="HD QAA pseudospectrum contour"),
        Line2D([0], [0], color=H_COLOR, lw=NUMERIC_DASHED_CONTOUR_LW * LINE_SCALE, linestyle="-", label="HM QAA pseudospectrum contour"),
        Line2D([0], [0], color=HD_COLOR, lw=1.20 * LINE_SCALE, linestyle=H_DASH, label="HD QAA theoretical estimation"),
        Line2D([0], [0], color=H_COLOR, lw=1.20 * LINE_SCALE, linestyle=H_DASH, label="HM QAA theoretical estimation"),
    ]


def draw_contour(
    ax: plt.Axes,
    coords: np.ndarray,
    color: str,
    *,
    alpha: float,
    lw: float,
    linestyle: str = "-",
    line_alpha: float = 1.0,
    marker: str | None = None,
    markevery=None,
    markersize: float = 3.0,
    markeredgewidth: float | None = None,
    zorder: float = 7,
) -> None:
    coords = close_polygon(coords)
    if alpha > 0.0:
        ax.fill(coords[:, 0], coords[:, 1], color=color, alpha=alpha, linewidth=0, zorder=2)
    ax.plot(
        coords[:, 0],
        coords[:, 1],
        color=color,
        lw=lw,
        linestyle=linestyle,
        alpha=line_alpha,
        marker=marker,
        markevery=markevery,
        markersize=markersize * LINE_SCALE,
        markerfacecolor=color,
        markeredgecolor=color,
        markeredgewidth=markeredgewidth,
        zorder=zorder,
    )


def nonduplicate_marker_indices(coords: np.ndarray, count: int, offset: int = 0) -> list[int]:
    """Uniform marker positions, excluding repeated closure points at the tail."""
    if len(coords) == 0:
        return []
    coordinate_scale = max(float(np.max(np.abs(coords))), np.finfo(float).tiny)
    duplicate_tolerance = 64.0 * np.finfo(float).eps * coordinate_scale
    unique_len = len(coords)
    while unique_len > 1 and np.linalg.norm(coords[unique_len - 1] - coords[0]) <= duplicate_tolerance:
        unique_len -= 1
    while unique_len > 1 and np.linalg.norm(coords[unique_len - 1] - coords[unique_len - 2]) <= duplicate_tolerance:
        unique_len -= 1
    if unique_len <= 0:
        return []
    count = max(1, min(int(count), unique_len))
    raw = (np.linspace(0, unique_len, count, endpoint=False) + offset) % unique_len
    return sorted({int(round(value)) % unique_len for value in raw})


def draw_ck_size_inset(parent_ax: plt.Axes) -> None:
    df = pd.read_csv(DATA_DIR / "thresholds_vs_size.csv").sort_values("n")
    y = df["log10_epsilon_threshold_theory"]
    y_min = float(y.min())
    y_max = float(y.max())

    ax = parent_ax.inset_axes([0.150, 0.105, 0.47, 0.38])
    ax.set_facecolor("white")
    ax.plot(
        df["n"],
        y,
        color="black",
        marker="o",
        markerfacecolor="black",
        markeredgecolor="black",
        lw=1.50,
        ms=3.6,
    )
    ax.set_title("")
    ax.set_xlabel(r"$n$", fontsize=13.5, labelpad=0.6)
    ax.set_ylabel(r"$\log_{10}\varepsilon$", fontsize=13.5, labelpad=0.6)
    ax.set_xlim(float(df["n"].min()) - 1.2, float(df["n"].max()) + 1.2)
    ax.set_xticks([5, 25, 45])
    ax.set_ylim(y_min - 280.0, y_max + 760.0)
    ax.set_yticks([y_min, -3000.0, y_max])
    ax.set_yticklabels([f"{y_min:.0f}", "-3000", f"{y_max:.0f}"])
    ax.tick_params(axis="both", labelsize=12.0, length=MAJOR_TICK_LENGTH, width=MAJOR_TICK_WIDTH, pad=1.0, direction="out")
    ax.grid(False, which="both")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.72)


def draw_ck_size_panel(ax: plt.Axes) -> None:
    df = pd.read_csv(DATA_DIR / "thresholds_vs_size.csv").sort_values("n")
    y = df["log10_epsilon_threshold_theory"]
    y_min = float(y.min())
    y_max = float(y.max())

    line, = ax.plot(
        df["n"],
        y,
        color="black",
        marker="s",
        markerfacecolor="black",
        markeredgecolor="black",
        lw=1.65 * LINE_SCALE,
        linestyle="-",
        ms=16.8 * LINE_SCALE,
    )
    set_visual_dash(line, THRESHOLD_DASH_POINTS)
    ax.set_xlabel(r"$n$", fontsize=AXIS_LABEL_FS, labelpad=1.0)
    ax.set_ylabel(r"$\log_{10}\varepsilon_c$", fontsize=AXIS_LABEL_FS, labelpad=1.0)
    ax.set_xlim(float(df["n"].min()) - 1.2, float(df["n"].max()) + 1.2)
    ax.set_xticks([5, 13, 21, 29, 37, 45])
    ax.set_ylim(y_min - 280.0, y_max + 760.0)
    ax.set_yticks([y_min, -9000.0, -6000.0, -3000.0, y_max])
    ax.set_yticklabels([f"{y_min:.0f}", "-9000", "-6000", "-3000", f"{y_max:.0f}"])
    ax.tick_params(axis="both", labelsize=TICK_FS, length=MAJOR_TICK_LENGTH, width=MAJOR_TICK_WIDTH, pad=1.0, direction="out")
    ax.grid(False, which="both")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.90 * LINE_SCALE)
    ax.set_box_aspect(1.0)


def style_local_axis(
    ax: plt.Axes,
    half_width: float,
    *,
    show_ylabel: bool,
    show_xlabel: bool,
    energy_subscript: str = "j",
) -> None:
    axis_half, scale, ticks, labels = integer_axis_params(half_width)
    ax.set_xlim(-axis_half, axis_half)
    ax.set_ylim(-axis_half, axis_half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(ticks)
    if energy_subscript in {"0", "1"}:
        display_labels = []
        for tick, label in zip(ticks, labels):
            display_labels.append(label if abs(tick) < 1e-9 or abs(abs(tick) - axis_half) < 1e-9 else "")
    else:
        display_labels = labels
    if energy_subscript == "j":
        display_labels = [label if idx % 2 == 0 else "" for idx, label in enumerate(display_labels)]
    ax.set_xticklabels(display_labels, fontsize=TICK_FS, color="black")
    ax.set_yticks(ticks)
    y_display_labels = display_labels
    if energy_subscript == "j":
        y_display_labels = [label if idx % 2 == 0 else "" for idx, label in enumerate(display_labels)]
    ax.set_yticklabels(y_display_labels if show_ylabel else [], fontsize=TICK_FS, color="black")
    ax.tick_params(axis="both", length=MAJOR_TICK_LENGTH, width=MAJOR_TICK_WIDTH, pad=3.0, colors="black", direction="out")
    if show_xlabel:
        ax.set_xlabel(rf"$\mathrm{{Re}}(z-E_{energy_subscript})$", fontsize=AXIS_LABEL_FS, labelpad=5.0)
    if show_ylabel:
        ax.set_ylabel(rf"$\mathrm{{Im}}(z-E_{energy_subscript})$", fontsize=AXIS_LABEL_FS, labelpad=5.0)
    ax.grid(False, which="both")
    for spine in ax.spines.values():
        spine.set_linewidth(0.82 * LINE_SCALE)


def theory_energy_indices(energy: str) -> tuple[int, ...]:
    if energy == "E0":
        return (0,)
    if energy == "E1":
        return (1,)
    if energy == "both":
        return (0, 1)
    raise ValueError(f"Unsupported energy selection: {energy}")


def draw_fk_asinh_overlay(
    fig: plt.Figure,
    spec,
    rows: list[dict[str, str]],
    arrays: np.lib.npyio.NpzFile,
    energy: str = "both",
) -> plt.Axes:
    if energy not in {"both", "E0", "E1"}:
        raise ValueError(f"Unsupported energy selection: {energy}")
    ax = fig.add_subplot(spec)
    transformed: list[np.ndarray] = []
    r_label_positions: list[tuple[int, float, str]] = []

    for r in [1, 2, 3, 4]:
        row_e0 = row_for(rows, "NHMIS-FKQAA-full-direct-sum", "E0", r)
        row_e1 = row_for(rows, "NHMIS-FKQAA-full-direct-sum", "E1", r)
        raw0 = close_polygon(contour_xy(row_e0, arrays))
        raw1 = close_polygon(contour_xy(row_e1, arrays))
        true_rho = float(max(np.max(np.hypot(raw0[:, 0], raw0[:, 1])), np.max(np.hypot(raw1[:, 0], raw1[:, 1]))))
        xy0 = radial_asinh(raw0)
        xy1 = radial_asinh(raw1)
        if energy in {"both", "E0"}:
            transformed.append(xy0)
        if energy in {"both", "E1"}:
            transformed.append(xy1)
        marker_count = {1: 5, 2: 13, 3: 19, 4: 29}[r]
        marker_offset = {1: 0, 2: 4, 3: 6, 4: 8}[r]
        if energy in {"both", "E0"}:
            draw_contour(
                ax,
                xy0,
                BLUE,
                alpha=0.0,
                lw=NUMERIC_CONTOUR_LW * LINE_SCALE,
                line_alpha=0.54,
                marker="o",
                markevery=nonduplicate_marker_indices(xy0, marker_count, offset=0),
                markersize=20.0,
                zorder=8,
            )
        if energy in {"both", "E1"}:
            draw_contour(
                ax,
                xy1,
                BLUE,
                alpha=0.0,
                lw=NUMERIC_CONTOUR_LW * LINE_SCALE,
                line_alpha=0.44,
                zorder=8,
            )
            draw_contour(
                ax,
                xy1,
                BLUE,
                alpha=0.0,
                lw=0.0,
                line_alpha=0.82,
                marker="x",
                markevery=nonduplicate_marker_indices(xy1, marker_count, offset=marker_offset),
                markersize=23.2,
                markeredgewidth=3.0,
                zorder=13,
            )
        sectors = history_sector_data_for_r(r, n=5, p=2.0, q=4.0)
        condition_scales = local_condition_scales_for_r(sectors, len(sectors[0]["v"]) - 1)
        for k in theory_energy_indices(energy):
            radius = float(EPSILON_VIS * condition_scales[k])
            estimate_xy = radial_asinh(circle_xy(radius))
            draw_condition_circle(
                ax,
                estimate_xy,
                color=BLUE,
                linestyle=THEORY_DASH_BY_LEVEL[r],
                marker=None,
                markevery=None,
                lw=5.655,
                alpha=1.0,
                zorder=22,
            )
        r_label_positions.append(
            (
                r,
                float(max(np.max(np.hypot(xy0[:, 0], xy0[:, 1])), np.max(np.hypot(xy1[:, 0], xy1[:, 1])))),
                true_scale_label(true_rho),
            )
        )

    all_xy = np.vstack(transformed)
    max_abs = float(np.max(np.abs(all_xy)))
    lim = math.ceil((max_abs + 2.0) / 10.0) * 10.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ticks = np.arange(-lim, lim + 0.1, 12.5)
    tick_labels = [f"{tick:.1f}".rstrip("0").rstrip(".") for tick in ticks]
    sparse_tick_labels = [label if idx % 2 == 0 else "" for idx, label in enumerate(tick_labels)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(sparse_tick_labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels(sparse_tick_labels)
    ax.tick_params(axis="both", labelsize=TICK_FS, length=MAJOR_TICK_LENGTH, width=MAJOR_TICK_WIDTH, pad=8.0)
    energy_subscript = "0" if energy == "E0" else "1"
    if energy == "both":
        energy_subscript = "j"
    ax.set_xlabel(rf"asinh-scaled $\mathrm{{Re}}(z-E_{energy_subscript})$", fontsize=AXIS_LABEL_FS, labelpad=14.0)
    ax.set_ylabel(rf"asinh-scaled $\mathrm{{Im}}(z-E_{energy_subscript})$", fontsize=AXIS_LABEL_FS, labelpad=14.0)
    ax.set_title("")
    ax.set_title("", loc="left")
    for r, radius, scale_text in r_label_positions:
        label_y = min(radius - 2.2, lim - 4.5)
        add_miktex_label(ax, 0.0, label_y, r, scale_text)

    ax.grid(False, which="both")
    for spine in ax.spines.values():
        spine.set_linewidth(0.90 * LINE_SCALE)
    return ax


def draw_r5_group(fig: plt.Figure, spec, rows: list[dict[str, str]], arrays: np.lib.npyio.NpzFile) -> plt.Axes:
    row = row_for(rows, "NHMIS-FKQAA-full-direct-sum", "connected_E0_E1", 5)
    coords = close_polygon(contour_xy(row, arrays))
    ax = fig.add_subplot(spec)
    draw_contour(ax, coords, BLUE, alpha=0.13, lw=NUMERIC_CONTOUR_LW * LINE_SCALE)
    e0 = float(row["E0"])
    e1 = float(row["E1"])
    ax.scatter([e0], [0.0], marker="o", s=ENERGY_POINT_S, facecolor="black", edgecolor="black", linewidth=0.0, zorder=6)
    ax.scatter([e1], [0.0], marker="o", s=ENERGY_POINT_S, facecolor="black", edgecolor="black", linewidth=0.0, zorder=6)
    xmin, xmax = float(coords[:, 0].min()), float(coords[:, 0].max())
    ymin, ymax = float(coords[:, 1].min()), float(coords[:, 1].max())
    scale = 1e-3
    xlim, ylim = square_limits(padded_limits(xmin, xmax, 0.006), padded_limits(ymin, ymax, 0.070))
    x_ticks, x_labels = r5_x_ticks(xlim[0], xlim[1], scale, e0, e1)
    y_ticks, y_labels = decimal_ticks_for_range(ylim[0], ylim[1], 0.50 * scale, scale)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("N")
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=TICK_FS)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=TICK_FS)
    ax.tick_params(axis="both", length=MAJOR_TICK_LENGTH, width=MAJOR_TICK_WIDTH, pad=2.0, colors="black", direction="out")
    ax.xaxis.set_minor_locator(MultipleLocator(0.125 * scale))
    ax.yaxis.set_minor_locator(MultipleLocator(0.10 * scale))
    ax.tick_params(axis="both", which="minor", length=MINOR_TICK_LENGTH, width=MINOR_TICK_WIDTH, colors="black", direction="out")
    energy_label_y = 0.16 * max(abs(ylim[0]), abs(ylim[1]))
    e0_label_x = e0
    e0_label_y = energy_label_y
    ax.text(e0_label_x, e0_label_y, r"$E_0$", ha="center", va="center", fontsize=1.70 * ENERGY_LABEL_FS, color="black")
    ax.text(e1, energy_label_y, r"$E_1$", ha="center", va="center", fontsize=1.70 * ENERGY_LABEL_FS, color="black")
    ax.text(
        0.96,
        0.06,
        r"$10^{-3}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=1.5 * ENERGY_LABEL_FS,
        color="black",
    )
    ax.set_title("")
    ax.set_title("", loc="left")
    ax.set_xlabel(r"$\mathrm{Re}\,z$", fontsize=AXIS_LABEL_FS, labelpad=1.0)
    ax.set_ylabel(r"$\mathrm{Im}\,z$", fontsize=AXIS_LABEL_FS, labelpad=1.0)
    ax.grid(False, which="both")
    for spine in ax.spines.values():
        spine.set_linewidth(0.90 * LINE_SCALE)
    return ax


def draw_hhd_group(fig: plt.Figure, spec, rows: list[dict[str, str]], arrays: np.lib.npyio.NpzFile) -> dict[str, plt.Axes]:
    sub = spec.subgridspec(
        2,
        2,
        height_ratios=[0.20, 1.0],
        width_ratios=[1.0, 1.0],
        wspace=0.12,
        hspace=0.03,
    )
    title_ax = fig.add_subplot(sub[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.78, "HM QAA + HD QAA r=1,2,3,4,5", ha="left", va="center", fontsize=13.0, fontweight="semibold")
    title_ax.legend(
        handles=[
            Line2D([0], [0], color="0.20", lw=2.00, linestyle="-", label="HD QAA"),
            Line2D([0], [0], color="0.20", lw=2.35, linestyle=H_DASH, label="HM QAA"),
        ],
        loc="center right",
        bbox_to_anchor=(1.0, 0.20),
        frameon=False,
        fontsize=10.8,
        ncol=2,
        handlelength=2.5,
        columnspacing=1.0,
        borderaxespad=0.0,
    )

    hd0 = close_polygon(contour_xy(row_for(rows, "NHMIS-HDQAA", "E0"), arrays))
    h0 = close_polygon(contour_xy(row_for(rows, "HMIS-HDQAA", "E0"), arrays))
    hd1 = close_polygon(contour_xy(row_for(rows, "NHMIS-HDQAA", "E1"), arrays))
    h1 = close_polygon(contour_xy(row_for(rows, "HMIS-HDQAA", "E1"), arrays))
    half = contour_half_width([hd0, h0, hd1, h1], pad=0.05)

    ax0 = fig.add_subplot(sub[1, 0])
    draw_contour(ax0, hd0, BLUE, alpha=0.14, lw=1.25, linestyle="-", line_alpha=1.0)
    draw_contour(ax0, h0, BLUE, alpha=0.06, lw=1.45, linestyle=H_DASH, line_alpha=0.95)
    style_local_axis(ax0, half, show_ylabel=True, show_xlabel=True)
    ax0.scatter([0.0], [0.0], marker="o", s=30, facecolor="black", edgecolor="black", linewidth=0.0, zorder=7)
    ax0.text(0.04, 0.96, r"$E_0=0$", transform=ax0.transAxes, ha="left", va="top", fontsize=1.5 * ENERGY_LABEL_FS, color="black", fontweight="semibold")

    ax1 = fig.add_subplot(sub[1, 1])
    draw_contour(ax1, hd1, ORANGE, alpha=0.14, lw=1.25, linestyle="-", line_alpha=1.0)
    draw_contour(ax1, h1, ORANGE, alpha=0.06, lw=1.45, linestyle=H_DASH, line_alpha=0.95)
    style_local_axis(ax1, half, show_ylabel=False, show_xlabel=True)
    ax1.scatter([0.0], [0.0], marker="o", s=30, facecolor="black", edgecolor="black", linewidth=0.0, zorder=7)
    ax1.text(0.04, 0.96, r"$E_1=1$", transform=ax1.transAxes, ha="left", va="top", fontsize=1.5 * ENERGY_LABEL_FS, color="black", fontweight="semibold")
    return {"title": title_ax, "e0": ax0, "e1": ax1}


def draw_hhd_axis(
    fig: plt.Figure,
    spec,
    hd: np.ndarray,
    h: np.ndarray,
    half: float,
    *,
    title: str,
    center_text: str,
    color: str,
    show_ylabel: bool,
    show_legend: bool = False,
) -> plt.Axes:
    ax = fig.add_subplot(spec)
    draw_contour(ax, hd, HD_COLOR, alpha=0.14, lw=NUMERIC_CONTOUR_LW * LINE_SCALE, linestyle="-", line_alpha=0.55)
    draw_contour(ax, h, H_COLOR, alpha=0.06, lw=NUMERIC_DASHED_CONTOUR_LW * LINE_SCALE, linestyle="-", line_alpha=0.55)
    center_type = "E0" if "E_0" in center_text else "E1"
    draw_condition_circle(ax, circle_xy(hd_condition_radius(4.0, center_type)), color=HD_COLOR, linestyle=H_DASH, lw=1.10)
    draw_condition_circle(ax, circle_xy(hd_condition_radius(1.0, center_type)), color=H_COLOR, linestyle=H_DASH, lw=1.10)
    energy_subscript = "0" if center_type == "E0" else "1"
    style_local_axis(ax, half, show_ylabel=show_ylabel, show_xlabel=True, energy_subscript=energy_subscript)
    ax.scatter([0.0], [0.0], marker="o", s=36 * LINE_SCALE, facecolor="black", edgecolor="black", linewidth=0.0, zorder=7)
    ax.text(0.04, 0.96, center_text, transform=ax.transAxes, ha="left", va="top", fontsize=1.5 * ENERGY_LABEL_FS, color="black", fontweight="semibold")
    ax.text(
        0.96,
        0.06,
        r"$10^{-32}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=1.5 * ENERGY_LABEL_FS,
        color="black",
    )
    ax.set_title(title, loc="left", fontsize=AXIS_LABEL_FS, fontweight="semibold")
    if show_legend:
        ax.legend(
            handles=hhd_legend_handles(color),
            loc="upper right",
            frameon=False,
            fontsize=LEGEND_FS * 0.72,
            handlelength=4.6,
            handletextpad=0.45,
            labelspacing=0.22,
            borderaxespad=0.15,
        )
    return ax


def draw_hhd_merged_axis(
    fig: plt.Figure,
    spec,
    hd0: np.ndarray,
    h0: np.ndarray,
    hd1: np.ndarray,
    h1: np.ndarray,
    half: float,
) -> plt.Axes:
    ax = fig.add_subplot(spec)
    draw_contour(
        ax,
        hd0,
        HD_COLOR,
        alpha=0.10,
        lw=NUMERIC_CONTOUR_LW * LINE_SCALE,
        linestyle="-",
        line_alpha=0.34,
        zorder=7,
    )
    draw_contour(
        ax,
        hd0,
        HD_COLOR,
        alpha=0.0,
        lw=0.0,
        linestyle="-",
        line_alpha=0.62,
        marker="o",
        markevery=nonduplicate_marker_indices(hd0, 11, offset=0),
        markersize=19.0,
        zorder=12,
    )
    draw_contour(
        ax,
        h0,
        H_COLOR,
        alpha=0.05,
        lw=NUMERIC_DASHED_CONTOUR_LW * LINE_SCALE,
        linestyle="-",
        line_alpha=0.34,
        zorder=7,
    )
    draw_contour(
        ax,
        h0,
        H_COLOR,
        alpha=0.0,
        lw=0.0,
        linestyle="-",
        line_alpha=0.62,
        marker="o",
        markevery=nonduplicate_marker_indices(h0, 11, offset=16),
        markersize=19.0,
        zorder=12,
    )
    draw_contour(
        ax,
        hd1,
        HD_COLOR,
        alpha=0.0,
        lw=NUMERIC_CONTOUR_LW * LINE_SCALE,
        linestyle="-",
        line_alpha=0.30,
        zorder=7,
    )
    draw_contour(
        ax,
        hd1,
        HD_COLOR,
        alpha=0.0,
        lw=0.0,
        linestyle="-",
        line_alpha=0.78,
        marker="x",
        markevery=(8, 34),
        markersize=22.0,
        markeredgewidth=3.0,
        zorder=13,
    )
    draw_contour(
        ax,
        h1,
        H_COLOR,
        alpha=0.0,
        lw=NUMERIC_DASHED_CONTOUR_LW * LINE_SCALE,
        linestyle="-",
        line_alpha=0.30,
        zorder=7,
    )
    draw_contour(
        ax,
        h1,
        H_COLOR,
        alpha=0.0,
        lw=0.0,
        linestyle="-",
        line_alpha=0.78,
        marker="x",
        markevery=(25, 34),
        markersize=22.0,
        markeredgewidth=3.0,
        zorder=13,
    )
    for center_type in ["E0", "E1"]:
        draw_condition_circle(
            ax,
            circle_xy(hd_condition_radius(4.0, center_type)),
            color=HD_COLOR,
            linestyle=THEORY_DASH_BY_LEVEL[4],
            lw=6.045,
        )
        draw_condition_circle(
            ax,
            circle_xy(hd_condition_radius(1.0, center_type)),
            color=H_COLOR,
            linestyle=THEORY_DASH_BY_LEVEL[1],
            lw=6.045,
        )
    style_local_axis(ax, half, show_ylabel=True, show_xlabel=True, energy_subscript="j")
    ax.scatter([0.0], [0.0], marker="o", s=ENERGY_POINT_S, facecolor="black", edgecolor="black", linewidth=0.0, zorder=12)
    ax.text(
        0.96,
        0.06,
        r"$10^{-32}$",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=1.5 * ENERGY_LABEL_FS,
        color="black",
    )
    return ax


def draw_asinh_left_panel(fig: plt.Figure, spec, rows: list[dict[str, str]]) -> dict[str, object]:
    arrays = contour_arrays()
    outer = spec.subgridspec(
        2,
        2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=-0.12,
        hspace=0.24,
    )
    fk_ax = draw_fk_asinh_overlay(fig, outer[0, 0], rows, arrays)
    r5_ax = draw_r5_group(fig, outer[0, 1], rows, arrays)
    hd0 = close_polygon(contour_xy(row_for(rows, "NHMIS-HDQAA", "E0"), arrays))
    h0 = close_polygon(contour_xy(row_for(rows, "HMIS-HDQAA", "E0"), arrays))
    hd1 = close_polygon(contour_xy(row_for(rows, "NHMIS-HDQAA", "E1"), arrays))
    h1 = close_polygon(contour_xy(row_for(rows, "HMIS-HDQAA", "E1"), arrays))
    half = contour_half_width([hd0, h0, hd1, h1], pad=0.05)
    hhd_e0 = draw_hhd_axis(
        fig,
        outer[1, 0],
        hd0,
        h0,
        half,
        title="HM QAA + HD QAA E0",
        center_text=r"$E_0=0$",
        color=BLUE,
        show_ylabel=True,
        show_legend=False,
    )
    hhd_e1 = draw_hhd_axis(
        fig,
        outer[1, 1],
        hd1,
        h1,
        half,
        title="HM QAA + HD QAA E1",
        center_text=r"$E_1=1$",
        color=ORANGE,
        show_ylabel=True,
        show_legend=False,
    )
    arrays.close()
    return {"fk": fk_ax, "r5": r5_ax, "hhd": {"e0": hhd_e0, "e1": hhd_e1}}


def shift_axes_y(axes: list[plt.Axes], dy: float) -> None:
    for ax in axes:
        pos = ax.get_position()
        ax.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height])


def align_left_geometry(fig: plt.Figure, left_axes: dict[str, object], ax_right: plt.Axes) -> None:
    fig.canvas.draw()
    right_pos = ax_right.get_position()
    fig_w, fig_h = fig.get_size_inches()

    fk_ax = left_axes["fk"]
    if isinstance(fk_ax, plt.Axes):
        fk_pos = fk_ax.get_position()
        fk_ax.set_position([fk_pos.x0, right_pos.y0, right_pos.width, right_pos.height])

    r5_ax = left_axes["r5"]
    if isinstance(r5_ax, plt.Axes):
        fk_pos = fk_ax.get_position() if isinstance(fk_ax, plt.Axes) else None
        old = r5_ax.get_position()
        old_aspect = (old.height * fig_h) / (old.width * fig_w)
        middle_left = (fk_pos.x1 + 0.026) if fk_pos is not None else old.x0
        middle_right = right_pos.x0 - 0.024
        width = max(old.width, middle_right - middle_left)
        height = width * fig_w / fig_h * old_aspect
        r5_ax.set_position([middle_left, right_pos.y1 - height, width, height])

    hhd = left_axes["hhd"]
    if isinstance(hhd, dict):
        hhd_axes = [hhd["title"], hhd["e0"], hhd["e1"]]
        hhd_top = max(ax.get_position().y1 for ax in hhd_axes)
        if isinstance(r5_ax, plt.Axes):
            target_top = r5_ax.get_position().y0 - 0.036
        else:
            target_top = hhd_top
        shift_axes_y(hhd_axes, target_top - hhd_top)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.065,
        1.045,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=PANEL_LABEL_FS,
        fontweight="semibold",
        color="black",
        zorder=30,
        clip_on=False,
    )


def scale_threshold_panel(ax: plt.Axes) -> None:
    ax.set_xlabel(ax.get_xlabel(), fontsize=AXIS_LABEL_FS, labelpad=1.0)
    ax.set_ylabel(ax.get_ylabel(), fontsize=AXIS_LABEL_FS, labelpad=1.0)
    ax.tick_params(axis="both", labelsize=TICK_FS, length=MAJOR_TICK_LENGTH, width=MAJOR_TICK_WIDTH, pad=2.0, direction="out")
    for text in ax.texts:
        text.set_fontsize(THRESHOLD_ANNOT_FS)
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    for line in ax.lines:
        line.set_linewidth(line.get_linewidth() * 1.80)
        line.set_markersize(line.get_markersize() * 4.0 * LINE_SCALE)
    ax.lines[0].set_linestyle("-")
    set_visual_dash(ax.lines[1], THRESHOLD_DASH_POINTS)
    for spine in ax.spines.values():
        spine.set_linewidth(0.95 * LINE_SCALE)


def set_threshold_numerical_marker(ax: plt.Axes, marker: str) -> None:
    if ax.lines:
        ax.lines[0].set_marker(marker)


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

    fig = plt.figure(figsize=(24.0, 32.0), constrained_layout=False)
    outer = GridSpec(
        3,
        2,
        figure=fig,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0, 1.0],
        left=0.040,
        right=0.995,
        bottom=0.105,
        top=0.975,
        wspace=-0.050,
        hspace=0.370,
    )

    arrays = contour_arrays()
    hd0 = close_polygon(contour_xy(row_for(rows, "NHMIS-HDQAA", "E0"), arrays))
    h0 = close_polygon(contour_xy(row_for(rows, "HMIS-HDQAA", "E0"), arrays))
    hd1 = close_polygon(contour_xy(row_for(rows, "NHMIS-HDQAA", "E1"), arrays))
    h1 = close_polygon(contour_xy(row_for(rows, "HMIS-HDQAA", "E1"), arrays))
    half = contour_half_width([hd0, h0, hd1, h1], pad=0.05)
    ax_a = draw_hhd_merged_axis(fig, outer[0, 0], hd0, h0, hd1, h1, half)
    ax_b = draw_fk_asinh_overlay(fig, outer[0, 1], rows, arrays, energy="E0")
    ax_c = draw_fk_asinh_overlay(fig, outer[1, 0], rows, arrays, energy="E1")
    ax_d = draw_r5_group(fig, outer[1, 1], rows, arrays)
    arrays.close()

    ax_e = fig.add_subplot(outer[2, 0])
    draw_threshold_panel(ax_e)
    scale_threshold_panel(ax_e)
    set_threshold_numerical_marker(ax_e, marker="s")
    ax_e.set_ylabel(r"$\log_{10}\varepsilon_c$", fontsize=AXIS_LABEL_FS, labelpad=1.0)
    ax_f = fig.add_subplot(outer[2, 1])
    draw_ck_size_panel(ax_f)

    labeled_axes = [
        ("(a)", ax_a),
        ("(b)", ax_b),
        ("(c)", ax_c),
        ("(d)", ax_d),
        ("(e)", ax_e),
        ("(f)", ax_f),
    ]
    for label, ax in labeled_axes:
        ax.set_title("")
        ax.set_title("", loc="left")
        ax.set_title("", loc="right")
        ax.set_box_aspect(1.0)
        ax.set_anchor("C")
        add_panel_label(ax, label)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / "exp_two_psedospectrum.png"
    pdf = FIG_DIR / "exp_two_psedospectrum.pdf"
    fig.savefig(png, dpi=140, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
