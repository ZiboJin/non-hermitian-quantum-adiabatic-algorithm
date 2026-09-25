from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_two_resolvent_pseudospectrum import (
    _fk_data_for_sector,
    eigenvalue_condition_estimate_for_sector,
    selected_sector_for_n,
)


OUT_DIR = Path("outputs") / "experiment_two_resolvent_paper_final_R2"
FIG_DIR = OUT_DIR / "figures"
DATA_DIR = OUT_DIR / "data"

BLUE = "#4C72B0"
N_VALUES = [5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]


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


def compute_thresholds() -> pd.DataFrame:
    rows: list[dict] = []
    for n in N_VALUES:
        start = time.perf_counter()
        sector, rule = selected_sector_for_n(n)
        fk = _fk_data_for_sector(n, sector, p=2.0, q=4.0)
        est = eigenvalue_condition_estimate_for_sector(fk["logW"], chunk_size=128)
        rows.append(
            {
                "n": int(n),
                "m": int(fk["graph"]["m"]),
                "r": int(n),
                "L": int(fk["L"]),
                "sector_used": "".join(str(int(bit)) for bit in sector),
                "sector_selection_rule": rule,
                "log10_epsilon_threshold_theory": float(est["log10_epsilon_est"]),
                "minus_log10_epsilon_threshold_theory": float(-est["log10_epsilon_est"]),
                "epsilon_threshold_theory": float(10.0 ** float(est["log10_epsilon_est"])),
                "k_star_est": int(est["k_star_est"]),
                "log10_chi_0": float(est["log10_chi_0"]),
                "log10_chi_kstar": float(est["log10_chi_kstar"]),
                "runtime_seconds": float(time.perf_counter() - start),
            }
        )
        print(
            f"n={n} m={fk['graph']['m']} L={fk['L']} "
            f"log10eps={est['log10_epsilon_est']:.6g}"
        )
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame) -> tuple[Path, Path]:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "font.size": 9,
            "axes.labelsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
    ax.plot(
        df["n"],
        df["log10_epsilon_threshold_theory"],
        color=BLUE,
        marker="o",
        markerfacecolor=BLUE,
        markeredgecolor=BLUE,
        lw=1.8,
        ms=4.8,
    )
    ax.set_title(r"CK graph theoretical noise strength $\varepsilon$ threshold", loc="left", fontweight="normal")
    ax.set_xlabel(r"CK graph size $n$")
    ax.set_ylabel(r"$\log_{10}\varepsilon$")
    ax.set_xlim(float(df["n"].min()) - 1.0, float(df["n"].max()) + 1.0)
    ax.set_xticks(df["n"])
    ymin = np.floor(float(df["log10_epsilon_threshold_theory"].min()) / 1000.0) * 1000.0
    ymax = np.ceil(float(df["log10_epsilon_threshold_theory"].max()) / 100.0) * 100.0
    ax.set_ylim(ymin - 80.0, ymax + 8.0)
    ax.set_yticks(endpoint_yticks(df["log10_epsilon_threshold_theory"], step=1000.0))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.85)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / "fig2_ck_size_theory_threshold.png"
    pdf = FIG_DIR / "fig2_ck_size_theory_threshold.pdf"
    fig.savefig(png, dpi=280)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = compute_thresholds()
    csv_path = DATA_DIR / "thresholds_vs_size.csv"
    df.to_csv(csv_path, index=False)
    png, pdf = plot(df)
    print(csv_path)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
