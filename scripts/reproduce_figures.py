"""Redraw manuscript Figures 5--7 using the archived numerical data."""
from pathlib import Path
import argparse
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "reproduced" / ".matplotlib"))
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(key, "1")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", type=int, choices=[5, 6, 7], default=[5, 6, 7])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reproduced")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    for figure in args.figures:
        plt.rcdefaults()
        if figure == 5:
            from plot_ck_vs_ck_like_gamma10 import plot_ck_vs_ck_like, set_palette
            set_palette("prl_muted")
            fig, _, _ = plot_ck_vs_ck_like(
                str(ROOT / "data/figure5/ck_results.csv"),
                str(ROOT / "data/figure5/ck_like_summary.csv"),
                str(out / "figure5.png"), pdf=str(out / "figure5.pdf"), n_max=45,
            )
            plt.close(fig)
        elif figure == 6:
            import plot_figure6 as module
            module.FIG_DIR = out
            module.main()
            for suffix in [".pdf", ".png"]:
                (out / ("exp_two_psedospectrum" + suffix)).replace(out / ("figure6" + suffix))
        else:
            from experiment_three_noisy_dynamics import plot_experiment_three_from_summary
            fig = plot_experiment_three_from_summary(
                str(ROOT / "data/figure7/summary.csv"), str(out),
                png_filename=str(out / "figure7.png"), pdf_filename=str(out / "figure7.pdf"),
            )
            plt.close(fig)
        print(f"Figure {figure}: {out / ('figure' + str(figure) + '.pdf')}", flush=True)

if __name__ == "__main__":
    main()
