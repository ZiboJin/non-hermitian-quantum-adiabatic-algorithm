"""CLI for experiment three."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse

try:
    from .experiment_three_noisy_dynamics import (
        ALGORITHMS,
        DEFAULT_EXP3_EPS_LIST,
        plot_experiment_three_from_summary,
        run_experiment_three,
    )
except ImportError:  # pragma: no cover
    from experiment_three_noisy_dynamics import (
        ALGORITHMS,
        DEFAULT_EXP3_EPS_LIST,
        plot_experiment_three_from_summary,
        run_experiment_three,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run experiment three noisy dynamics.")
    parser.add_argument(
        "--eps-list",
        nargs="+",
        type=float,
        default=DEFAULT_EXP3_EPS_LIST,
    )
    parser.add_argument("--n-real", type=int, default=20)
    parser.add_argument(
        "--instance",
        choices=["paper_m2", "ck_m2_r1", "toy_star3_r1", "toy_star3_r2"],
        default="paper_m2",
    )
    parser.add_argument("--n-noise", type=int, default=70)
    parser.add_argument("--steps-per-segment", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--out-dir", default="outputs/experiment_three_noisy")
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHMS, default=ALGORITHMS)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--q", type=float, default=4.0)
    parser.add_argument("--Omega", type=float, default=1.0)
    parser.add_argument("--integrator", choices=["split", "combined"], default="combined")
    parser.add_argument("--norm-method", choices=["power", "exact"], default="power")
    parser.add_argument("--norm-iters", type=int, default=32)
    parser.add_argument(
        "--seed-mode",
        choices=["per_algorithm", "paired_noise"],
        default="per_algorithm",
    )
    parser.add_argument(
        "--noise-model",
        choices=["ginibre", "scalar_identity", "none"],
        default="ginibre",
    )
    parser.add_argument("--baseline-mode", choices=["midpoint", "reference"], default="midpoint")
    parser.add_argument("--skip-baseline-validation", action="store_true")
    parser.add_argument("--renormalize-each-step", dest="renormalize_each_step", action="store_true", default=True)
    parser.add_argument("--no-renormalize-each-step", dest="renormalize_each_step", action="store_false")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--reserve-cores", type=int, default=1)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fk-include-hx", dest="fk_include_hx", action="store_true", default=True)
    parser.add_argument("--fk-no-include-hx", dest="fk_include_hx", action="store_false")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--summary-csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_only:
        summary_csv = args.summary_csv or os.path.join(args.out_dir, "summary.csv")
        fig = plot_experiment_three_from_summary(summary_csv, args.out_dir)
        fig.clf()
        return

    run_experiment_three(
        eps_list=args.eps_list,
        n_real=args.n_real,
        instance_name=args.instance,
        n_noise=args.n_noise,
        steps_per_segment=args.steps_per_segment,
        seed=args.seed,
        out_dir=args.out_dir,
        algorithms=args.algorithms,
        gamma=args.gamma,
        p=args.p,
        q=args.q,
        Omega=args.Omega,
        renormalize_each_step=args.renormalize_each_step,
        noise_model=args.noise_model,
        integrator=args.integrator,
        norm_method=args.norm_method,
        norm_iters=args.norm_iters,
        seed_mode=args.seed_mode,
        baseline_mode=args.baseline_mode,
        validate_baseline=not args.skip_baseline_validation,
        max_workers=args.max_workers,
        reserve_cores=args.reserve_cores,
        serial=args.serial,
        force=args.force,
        fk_include_hx=args.fk_include_hx,
    )


if __name__ == "__main__":
    main()
