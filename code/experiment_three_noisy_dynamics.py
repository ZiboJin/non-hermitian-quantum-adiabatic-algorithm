"""Experiment three: full-space noisy finite-time dynamics on CK m=2."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Callable

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.sparse.linalg import LinearOperator, expm_multiply

try:
    from .experiment_one_ck import (
        build_ck_gate_sequence,
        ck_graph_info,
        compute_prefix_weights_ck,
        default_max_workers,
        solve_empty_fk_clock_ivp,
        success_nhmis_fk,
        theta_schedule,
    )
    from .full_space_tools import (
        all_gate_eigenvalues,
        apply_full_fk_path_hamiltonian,
        apply_full_hd_active_link_hamiltonian,
        bitstrings_array,
        build_grover_unitary,
        build_full_space_instance,
        build_hx_plus_projector,
        build_m2_instance,
        dense_matrix_from_matvec,
        estimate_spectral_norm,
        initial_state_full,
        random_complex_matrix,
    )
    from .validate_full_space_m2 import (
        simulate_full_hmis_hd,
        simulate_full_nhmis_fk,
        simulate_full_nhmis_hd,
    )
except ImportError:  # pragma: no cover
    from experiment_one_ck import (
        build_ck_gate_sequence,
        ck_graph_info,
        compute_prefix_weights_ck,
        default_max_workers,
        solve_empty_fk_clock_ivp,
        success_nhmis_fk,
        theta_schedule,
    )
    from full_space_tools import (
        all_gate_eigenvalues,
        apply_full_fk_path_hamiltonian,
        apply_full_hd_active_link_hamiltonian,
        bitstrings_array,
        build_grover_unitary,
        build_full_space_instance,
        build_hx_plus_projector,
        build_m2_instance,
        dense_matrix_from_matvec,
        estimate_spectral_norm,
        initial_state_full,
        random_complex_matrix,
    )
    from validate_full_space_m2 import (
        simulate_full_hmis_hd,
        simulate_full_nhmis_fk,
        simulate_full_nhmis_hd,
    )


ALGORITHMS = ["NHMIS-FKQAA", "NHMIS-HDQAA", "HMIS-HDQAA"]
ALGORITHM_DISPLAY = {
    "NHMIS-FKQAA": "FK QAA",
    "NHMIS-HDQAA": "HD QAA",
    "HMIS-HDQAA": "H QAA",
}
ALGORITHM_COLORS_PRL_MUTED = {
    "NHMIS-FKQAA": "#4C72B0",
    "NHMIS-HDQAA": "#DD8452",
    "HMIS-HDQAA": "#55A868",
}
DEFAULT_EXP3_EPS_LIST = [
    1e-15,
    1e-14,
    1e-13,
    1e-12,
    1e-11,
    1e-10,
    1e-9,
    1e-8,
    1e-7,
    1e-6,
]
INTEGRATOR_VERSION = "segment_substeps_v2"
H0_MODEL_VERSION = "full_fk_input_v1"
H0_MODEL_VERSION_FK_NO_INPUT = "full_fk_no_input_v1"
FK_INPUT_TERM_WITH_HX = "hx_clock0"
FK_INPUT_TERM_NONE = "none"
FK_INPUT_TERM_NOT_APPLICABLE = "not_applicable"


def _enable_miktex_text_rendering() -> None:
    """Use the manuscript's Matplotlib text settings without machine-specific caches."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "text.usetex": False,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "axes.unicode_minus": False,
    })



def fk_input_term_for_algorithm(algorithm: str, fk_include_hx: bool = True) -> str:
    if algorithm == "NHMIS-FKQAA":
        return FK_INPUT_TERM_WITH_HX if fk_include_hx else FK_INPUT_TERM_NONE
    return FK_INPUT_TERM_NOT_APPLICABLE


def normalize_fk_input_term(algorithm: str, value) -> str:
    if algorithm != "NHMIS-FKQAA":
        return FK_INPUT_TERM_NOT_APPLICABLE
    if value is None:
        return FK_INPUT_TERM_WITH_HX
    try:
        if pd.isna(value):
            return FK_INPUT_TERM_WITH_HX
    except TypeError:
        pass
    text = str(value)
    return text if text else FK_INPUT_TERM_WITH_HX


def h0_model_version_for_algorithm(algorithm: str, fk_include_hx: bool = True) -> str:
    if algorithm == "NHMIS-FKQAA" and not fk_include_hx:
        return H0_MODEL_VERSION_FK_NO_INPUT
    return H0_MODEL_VERSION


REALIZATION_COLUMNS = [
    "instance_name",
    "algorithm",
    "epsilon",
    "realization",
    "seed",
    "seed_mode",
    "n_noise",
    "steps_per_segment",
    "n_time_steps",
    "n_real",
    "n",
    "r",
    "L",
    "full_dim",
    "p_raw",
    "p0",
    "p0_reference",
    "p0_midpoint",
    "baseline_abs_error",
    "baseline_rel_error",
    "baseline_mode",
    "integrator_version",
    "h0_model_version",
    "fk_input_term",
    "p0_reference_source",
    "integrator",
    "integrator_note",
    "norm_method",
    "norm_iters",
    "p_normalized",
    "final_norm",
    "runtime_seconds",
    "status",
    "error_message",
    "cache_hit",
]
MAIN_AXES_RECT = (0.12, 0.20, 0.84, 0.92)
SUMMARY_COLUMNS = [
    "instance_name",
    "algorithm",
    "epsilon",
    "n_success",
    "seed_mode",
    "n",
    "r",
    "L",
    "full_dim",
    "p0",
    "p0_reference",
    "p0_midpoint",
    "baseline_abs_error",
    "baseline_rel_error",
    "baseline_mode",
    "integrator_version",
    "h0_model_version",
    "fk_input_term",
    "p0_reference_source",
    "integrator",
    "integrator_note",
    "norm_method",
    "norm_iters",
    "median_p_raw",
    "min_p_raw",
    "q1_p_raw",
    "q3_p_raw",
    "max_p_raw",
    "median_p_normalized",
    "min_p_normalized",
    "q1_p_normalized",
    "q3_p_normalized",
    "max_p_normalized",
]


def _threadpool_limit_context():
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        return nullcontext()
    return threadpool_limits(limits=1)


def default_exp3_max_workers(reserve_cores: int = 1) -> int:
    """Memory-conscious default worker count for dense-noise propagation."""
    return min(default_max_workers(reserve_cores), 4)


def now_iso() -> str:
    """Return a stable local timestamp string."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def eps_token(epsilon: float) -> str:
    """Return a filename-safe epsilon token."""
    return f"{float(epsilon):.0e}".replace("-", "m")


def _algorithm_token(algorithm: str) -> str:
    return algorithm.replace("-", "").replace("_", "")


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(row: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(row, f, indent=2, sort_keys=True)


def exp3_baseline_cache_path(
    out_dir: str,
    algorithm: str,
    instance_name: str = "paper_m2",
    steps_per_segment: int = 8,
    integrator: str = "split",
    fk_include_hx: bool = True,
) -> str:
    h0_model_version = h0_model_version_for_algorithm(algorithm, fk_include_hx)
    return os.path.join(
        out_dir,
        "cache",
        (
            f"exp3_baseline_{instance_name}_{_algorithm_token(algorithm)}"
            f"_sps_{steps_per_segment}_{integrator}_{INTEGRATOR_VERSION}"
            f"_{h0_model_version}.json"
        ),
    )


def exp3_task_cache_path(
    out_dir: str,
    algorithm: str,
    epsilon: float,
    realization: int,
    n_noise: int,
    instance_name: str = "paper_m2",
    steps_per_segment: int = 8,
    noise_model: str = "ginibre",
    integrator: str = "split",
    norm_method: str = "power",
    norm_iters: int = 8,
    seed_mode: str = "per_algorithm",
    fk_include_hx: bool = True,
) -> str:
    seed_suffix = "" if seed_mode == "per_algorithm" else f"_seed_{seed_mode}"
    h0_model_version = h0_model_version_for_algorithm(algorithm, fk_include_hx)
    return os.path.join(
        out_dir,
        "cache",
        (
            f"exp3_{instance_name}_{_algorithm_token(algorithm)}_eps_{eps_token(epsilon)}"
            f"_real_{realization}_nnoise_{n_noise}_sps_{steps_per_segment}"
            f"_{noise_model}_{integrator}_{norm_method}_niters_{norm_iters}"
            f"{seed_suffix}_{INTEGRATOR_VERSION}_{h0_model_version}.json"
        ),
    )


def cached_success(path: str, force: bool = False) -> dict | None:
    if force or not os.path.exists(path):
        return None
    row = load_json(path)
    if row.get("status") == "success":
        row["cache_hit"] = True
        return row
    return None


def derive_seed(
    seed: int,
    algorithm: str,
    epsilon: float,
    realization: int,
    seed_mode: str = "per_algorithm",
) -> int:
    """Derive a reproducible per-task seed."""
    if seed_mode not in {"per_algorithm", "paired_noise"}:
        raise ValueError("seed_mode must be 'per_algorithm' or 'paired_noise'")
    eps_int = int(round(-np.log10(float(epsilon)) * 1000)) if epsilon > 0 else 0
    if seed_mode == "paired_noise":
        ss = np.random.SeedSequence([seed, eps_int, realization])
    else:
        alg_idx = ALGORITHMS.index(algorithm) if algorithm in ALGORITHMS else 99
        ss = np.random.SeedSequence([seed, alg_idx, eps_int, realization])
    return int(ss.generate_state(1)[0])


def compute_time_grid(
    L: int,
    steps_per_segment: int = 8,
    n_noise: int = 70,
    total_time: float | None = None,
) -> dict:
    """Return midpoint time grid and piecewise-constant noise indices.

    ``n_noise`` controls only noise resampling.  The deterministic schedule is
    always propagated over all ``L`` active segments using
    ``L * steps_per_segment`` midpoint steps.
    """
    if L <= 0:
        raise ValueError("L must be positive")
    if steps_per_segment <= 0:
        raise ValueError("steps_per_segment must be positive")
    if n_noise <= 0:
        raise ValueError("n_noise must be positive")
    n_time_steps = int(L * steps_per_segment)
    if total_time is None:
        total_time = float(L)
    dt = float(total_time) / n_time_steps
    steps = np.arange(n_time_steps, dtype=np.int64)
    t_mid = (steps + 0.5) * dt
    segment_indices = np.minimum(steps // steps_per_segment, L - 1).astype(np.int64)
    local_step = steps % steps_per_segment
    local_s = (local_step + 0.5) / steps_per_segment
    noise_indices = np.minimum(
        (steps * n_noise) // n_time_steps, n_noise - 1
    ).astype(np.int64)
    return {
        "L": int(L),
        "steps_per_segment": int(steps_per_segment),
        "n_noise": int(n_noise),
        "n_time_steps": int(n_time_steps),
        "dt": float(dt),
        "t_mid": t_mid,
        "segment_indices": segment_indices,
        "local_s": local_s,
        "noise_indices": noise_indices,
    }


def _success_from_flat_state(psi: np.ndarray, success_index: int) -> tuple[float, float]:
    norm = float(np.sum(np.abs(psi) ** 2))
    if norm <= 0.0 or not np.isfinite(norm):
        raise RuntimeError("state norm is not finite and positive")
    return float(abs(psi[success_index]) ** 2 / norm), norm


def apply_full_fk_path_hamiltonian_adjoint(
    psi: np.ndarray, s: float, V_diag: list[np.ndarray], include_hx: bool = False, hx=None
) -> np.ndarray:
    """Apply the adjoint of the full FK interpolation Hamiltonian."""
    work_dim, clock_dim = psi.shape
    init_diag = np.ones(clock_dim, dtype=float)
    init_diag[0] = 0.0
    prop = np.zeros_like(psi, dtype=complex)
    for idx, v in enumerate(V_diag):
        left = idx
        right = idx + 1
        prop[:, left] += 0.5 * psi[:, left]
        prop[:, right] += 0.5 * psi[:, right]
        prop[:, right] += -0.5 * (1.0 / v) * psi[:, left]
        prop[:, left] += -0.5 * v * psi[:, right]
    out = (1.0 - s) * psi * init_diag[None, :] + s * prop
    if include_hx:
        if hx is None:
            n = int(round(np.log2(work_dim)))
            if (1 << n) != work_dim:
                raise ValueError("work_dim must be a power of two when hx is not provided")
            hx = build_hx_plus_projector(n)
        out[:, 0] += hx @ psi[:, 0]
    return out


def apply_full_hd_active_link_hamiltonian_adjoint(
    psi: np.ndarray,
    l: int,
    theta: float,
    gate_type: str,
    Omega: float = 1.0,
    v: np.ndarray | None = None,
    U: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the adjoint active-link HD Hamiltonian."""
    if gate_type == "HMIS":
        return apply_full_hd_active_link_hamiltonian(
            psi, l, theta, gate_type, Omega=Omega, U=U
        )
    if gate_type != "NHMIS":
        raise ValueError("gate_type must be 'NHMIS' or 'HMIS'")
    if v is None:
        raise ValueError("v is required for NHMIS active link")
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    out = Omega * psi.astype(complex, copy=True)
    left = l - 1
    right = l
    a = psi[:, left]
    b = psi[:, right]
    out[:, left] = Omega * (sin_theta * sin_theta * a - sin_theta * cos_theta * v * b)
    out[:, right] = Omega * (
        -sin_theta * cos_theta * (a / v) + cos_theta * cos_theta * b
    )
    return out


def _full_h0_factory(
    algorithm: str,
    instance: dict,
    V_diag: list[np.ndarray],
    U: np.ndarray | None,
    gamma: float,
    Omega: float,
    fk_include_hx: bool = True,
) -> tuple[float, Callable[[float], tuple[Callable, Callable, complex]]]:
    """Return total time and a midpoint H0 matvec factory."""
    work_dim = instance["work_dim"]
    clock_dim = instance["clock_dim"]
    full_dim = instance["full_dim"]
    L = instance["L"]

    if algorithm == "NHMIS-FKQAA":
        total_time = gamma * L
    elif algorithm in {"NHMIS-HDQAA", "HMIS-HDQAA"}:
        total_time = gamma * L / Omega
    else:
        raise ValueError(f"unknown algorithm {algorithm}")

    hx = (
        build_hx_plus_projector(instance["n"])
        if algorithm == "NHMIS-FKQAA" and fk_include_hx
        else None
    )
    hx_trace = (
        0.5 * instance["n"] * work_dim
        if algorithm == "NHMIS-FKQAA" and fk_include_hx
        else 0.0
    )
    h0_trace = complex(
        work_dim * L + hx_trace if algorithm == "NHMIS-FKQAA" else Omega * work_dim * L
    )

    def factory(t_mid: float):
        if algorithm == "NHMIS-FKQAA":
            s = np.clip(t_mid / total_time, 0.0, 1.0)

            def matvec(x: np.ndarray) -> np.ndarray:
                psi = x.reshape(work_dim, clock_dim)
                return apply_full_fk_path_hamiltonian(
                    psi, float(s), V_diag, include_hx=fk_include_hx, hx=hx
                ).reshape(full_dim)

            def rmatvec(x: np.ndarray) -> np.ndarray:
                psi = x.reshape(work_dim, clock_dim)
                return apply_full_fk_path_hamiltonian_adjoint(
                    psi, float(s), V_diag, include_hx=fk_include_hx, hx=hx
                ).reshape(full_dim)

            return matvec, rmatvec, h0_trace

        tau = gamma / Omega
        segment = min(int(t_mid / tau), L - 1)
        local_s = np.clip((t_mid - segment * tau) / tau, 0.0, 1.0)
        theta = theta_schedule(local_s)
        l = segment + 1
        if algorithm == "NHMIS-HDQAA":
            v = V_diag[segment]

            def matvec(x: np.ndarray) -> np.ndarray:
                psi = x.reshape(work_dim, clock_dim)
                return apply_full_hd_active_link_hamiltonian(
                    psi, l, theta, "NHMIS", Omega=Omega, v=v
                ).reshape(full_dim)

            def rmatvec(x: np.ndarray) -> np.ndarray:
                psi = x.reshape(work_dim, clock_dim)
                return apply_full_hd_active_link_hamiltonian_adjoint(
                    psi, l, theta, "NHMIS", Omega=Omega, v=v
                ).reshape(full_dim)

            return matvec, rmatvec, h0_trace

        def matvec(x: np.ndarray) -> np.ndarray:
            psi = x.reshape(work_dim, clock_dim)
            return apply_full_hd_active_link_hamiltonian(
                psi, l, theta, "HMIS", Omega=Omega, U=U
            ).reshape(full_dim)

        def rmatvec(x: np.ndarray) -> np.ndarray:
            psi = x.reshape(work_dim, clock_dim)
            return apply_full_hd_active_link_hamiltonian_adjoint(
                psi, l, theta, "HMIS", Omega=Omega, U=U
            ).reshape(full_dim)

        return matvec, rmatvec, h0_trace

    return total_time, factory


class PiecewiseConstantNoise:
    """Generate and reuse one dense noise matrix per noise interval."""

    def __init__(
        self,
        dim: int,
        epsilon: float,
        rng: np.random.Generator,
        noise_model: str = "ginibre",
        norm_method: str = "power",
        norm_iters: int = 8,
        scalar_noise: complex = 1.0 + 0.25j,
    ) -> None:
        if epsilon < 0.0:
            raise ValueError("epsilon must be nonnegative")
        if noise_model not in {"ginibre", "scalar_identity", "none"}:
            raise ValueError("noise_model must be 'ginibre', 'scalar_identity', or 'none'")
        self.dim = int(dim)
        self.epsilon = float(epsilon)
        self.rng = rng
        self.noise_model = noise_model
        self.norm_method = norm_method
        self.norm_iters = int(norm_iters)
        self.scalar_noise = complex(scalar_noise)
        self.current_idx: int | None = None
        self.R: np.ndarray | None = None
        self.scaled = 0.0
        self.scalar = 0.0 + 0.0j
        self.noise_trace = 0.0 + 0.0j
        self.norm_R = 0.0

    def get(self, noise_idx: int) -> "PiecewiseConstantNoise":
        if self.epsilon == 0.0 or self.noise_model == "none":
            self.current_idx = int(noise_idx)
            self.R = None
            self.scaled = 0.0
            self.scalar = 0.0 + 0.0j
            self.noise_trace = 0.0 + 0.0j
            self.norm_R = 0.0
            return self
        if self.current_idx == int(noise_idx):
            return self

        self.current_idx = int(noise_idx)
        self.R = None
        self.scaled = 0.0
        self.scalar = 0.0 + 0.0j
        self.noise_trace = 0.0 + 0.0j
        self.norm_R = 0.0
        if self.noise_model == "ginibre":
            self.R = random_complex_matrix(self.dim, self.rng)
            self.norm_R = estimate_spectral_norm(
                self.R,
                method=self.norm_method,
                n_iter=self.norm_iters,
                rng=self.rng,
            )
            if self.norm_R <= 0.0 or not np.isfinite(self.norm_R):
                raise RuntimeError("random noise norm is not finite and positive")
            self.scaled = self.epsilon / self.norm_R
            self.noise_trace = self.scaled * np.trace(self.R)
        elif self.noise_model == "scalar_identity":
            self.scalar = self.epsilon * self.scalar_noise
            self.noise_trace = self.dim * self.scalar
        return self

    def matvec(self, x: np.ndarray) -> np.ndarray:
        if self.R is not None:
            return self.scaled * (self.R @ x)
        if self.scalar != 0.0:
            return self.scalar * x
        return np.zeros_like(x, dtype=complex)

    def rmatvec(self, x: np.ndarray) -> np.ndarray:
        if self.R is not None:
            return np.conj(self.scaled) * (self.R.conj().T @ x)
        if self.scalar != 0.0:
            return np.conj(self.scalar) * x
        return np.zeros_like(x, dtype=complex)

    def matmat(self, X: np.ndarray) -> np.ndarray:
        if self.R is not None:
            return self.scaled * (self.R @ X)
        if self.scalar != 0.0:
            return self.scalar * X
        return np.zeros_like(X, dtype=complex)

    def rmatmat(self, X: np.ndarray) -> np.ndarray:
        if self.R is not None:
            return np.conj(self.scaled) * (self.R.conj().T @ X)
        if self.scalar != 0.0:
            return np.conj(self.scalar) * X
        return np.zeros_like(X, dtype=complex)


def _expm_midpoint_step(
    psi: np.ndarray,
    dt: float,
    h0_matvec: Callable[[np.ndarray], np.ndarray],
    h0_rmatvec: Callable[[np.ndarray], np.ndarray],
    h0_trace: complex,
    noise: PiecewiseConstantNoise,
) -> tuple[np.ndarray, float]:
    """Apply one noisy midpoint exponential step."""
    dim = psi.size

    def _matvec_one(x: np.ndarray) -> np.ndarray:
        y = h0_matvec(x) + noise.matvec(x)
        return (-1j * dt) * y

    def _rmatvec_one(x: np.ndarray) -> np.ndarray:
        y = h0_rmatvec(x) + noise.rmatvec(x)
        return (1j * dt) * y

    def matvec(x: np.ndarray) -> np.ndarray:
        return _matvec_one(np.asarray(x).reshape(-1))

    def rmatvec(x: np.ndarray) -> np.ndarray:
        return _rmatvec_one(np.asarray(x).reshape(-1))

    def matmat(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        h0_part = np.column_stack([h0_matvec(X[:, j]) for j in range(X.shape[1])])
        h0_part = h0_part + noise.matmat(X)
        return (-1j * dt) * h0_part

    def rmatmat(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        h0_part = np.column_stack([h0_rmatvec(X[:, j]) for j in range(X.shape[1])])
        h0_part = h0_part + noise.rmatmat(X)
        return (1j * dt) * h0_part

    operator = LinearOperator(
        (dim, dim),
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
        dtype=complex,
    )
    trace = (-1j * dt) * (h0_trace + noise.noise_trace)
    return expm_multiply(operator, psi, traceA=trace), noise.norm_R


def propagate_state_midpoint(
    initial_state: np.ndarray,
    h0_factory: Callable[[float], tuple[Callable, Callable, complex]],
    total_time: float,
    n_steps: int,
    epsilon: float,
    seed: int,
    success_index: int | None = None,
    n_noise: int | None = None,
    renormalize_each_step: bool = True,
    noise_model: str = "ginibre",
    norm_method: str = "power",
    norm_iters: int = 8,
    scalar_noise: complex = 1.0 + 0.25j,
) -> dict:
    """Generic midpoint exponential propagation with piecewise-constant noise."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if n_noise is None:
        n_noise = n_steps
    if n_noise <= 0:
        raise ValueError("n_noise must be positive")
    if total_time < 0.0:
        raise ValueError("total_time must be nonnegative")
    if epsilon < 0.0:
        raise ValueError("epsilon must be nonnegative")
    psi = np.asarray(initial_state, dtype=complex).reshape(-1).copy()
    rng = np.random.default_rng(seed)
    dt = total_time / n_steps
    noise_indices = np.minimum(
        (np.arange(n_steps, dtype=np.int64) * n_noise) // n_steps, n_noise - 1
    ).astype(np.int64)
    noise = PiecewiseConstantNoise(
        psi.size,
        epsilon,
        rng,
        noise_model=noise_model,
        norm_method=norm_method,
        norm_iters=norm_iters,
        scalar_noise=scalar_noise,
    )
    norm_values = []

    for step in range(n_steps):
        t_mid = (step + 0.5) * dt
        h0_matvec, h0_rmatvec, h0_trace = h0_factory(t_mid)
        noise.get(int(noise_indices[step]))
        psi, norm_R = _expm_midpoint_step(
            psi,
            dt,
            h0_matvec,
            h0_rmatvec,
            h0_trace,
            noise,
        )
        if norm_R:
            norm_values.append(norm_R)
        if renormalize_each_step:
            norm = np.linalg.norm(psi)
            if norm <= 0.0 or not np.isfinite(norm):
                raise RuntimeError("state norm is not finite and positive")
            psi = psi / norm

    p_succ = np.nan
    final_norm = float(np.sum(np.abs(psi) ** 2))
    if success_index is not None:
        p_succ, final_norm = _success_from_flat_state(psi, success_index)
    return {
        "state": psi,
        "p_success": p_succ,
        "final_norm": final_norm,
        "mean_norm_R": float(np.mean(norm_values)) if norm_values else 0.0,
    }


def propagate_noiseless_hd_midpoint(
    algorithm: str,
    instance: dict,
    V_diag: list[np.ndarray],
    U: np.ndarray | None,
    steps_per_segment: int,
    gamma: float,
    Omega: float,
    renormalize_each_step: bool = True,
) -> dict:
    """Fast full-space midpoint propagation for zero-noise HDQAA."""
    if algorithm not in {"NHMIS-HDQAA", "HMIS-HDQAA"}:
        raise ValueError("fast HD midpoint supports only HDQAA algorithms")
    grid = compute_time_grid(
        instance["L"],
        steps_per_segment=steps_per_segment,
        n_noise=1,
        total_time=gamma * instance["L"] / Omega,
    )
    psi = initial_state_full(instance["work_dim"], instance["clock_dim"])
    phase = np.exp(-1j * Omega * grid["dt"])
    coeff = phase - 1.0
    for segment, local_s in zip(grid["segment_indices"], grid["local_s"]):
        l = int(segment) + 1
        left = l - 1
        right = l
        theta = theta_schedule(local_s)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        a = psi[:, left].copy()
        b = psi[:, right].copy()
        psi *= phase
        if algorithm == "NHMIS-HDQAA":
            v = V_diag[int(segment)]
            h_a = sin_theta * sin_theta * a - sin_theta * cos_theta * b / v
            h_b = -sin_theta * cos_theta * v * a + cos_theta * cos_theta * b
        else:
            if U is None:
                raise ValueError("U is required for HMIS-HDQAA")
            h_a = sin_theta * sin_theta * a - sin_theta * cos_theta * (U.conj().T @ b)
            h_b = -sin_theta * cos_theta * (U @ a) + cos_theta * cos_theta * b
        psi[:, left] = a + coeff * h_a
        psi[:, right] = b + coeff * h_b
        if renormalize_each_step:
            norm = np.linalg.norm(psi.reshape(-1))
            if norm <= 0.0 or not np.isfinite(norm):
                raise RuntimeError("state norm is not finite and positive")
            psi /= norm
    success_index = instance["x_star_index"] * instance["clock_dim"] + instance["L"]
    p_success, final_norm = _success_from_flat_state(psi.reshape(-1), success_index)
    return {
        "state": psi.reshape(-1),
        "p_success": p_success,
        "final_norm": final_norm,
        "mean_norm_R": 0.0,
        "n_time_steps": grid["n_time_steps"],
    }


def _apply_noise_only_exponential(
    psi_flat: np.ndarray,
    duration: float,
    noise: PiecewiseConstantNoise,
) -> np.ndarray:
    """Apply exp(-i duration DeltaH) using the current dense noise block."""
    if noise.epsilon == 0.0 or noise.noise_model == "none":
        return psi_flat
    if noise.noise_model == "scalar_identity":
        return np.exp(-1j * duration * noise.scalar) * psi_flat
    if noise.R is None:
        return psi_flat
    operator = (-1j * duration * noise.scaled) * noise.R
    trace = (-1j * duration) * noise.noise_trace
    return expm_multiply(operator, psi_flat, traceA=trace)


def _apply_h0_exponential_step(
    psi_flat: np.ndarray,
    dt: float,
    h0_matvec: Callable[[np.ndarray], np.ndarray],
    h0_rmatvec: Callable[[np.ndarray], np.ndarray],
    h0_trace: complex,
) -> np.ndarray:
    """Apply exp(-i dt H0_mid) for a deterministic full-space step."""
    dim = psi_flat.size

    def matvec(x: np.ndarray) -> np.ndarray:
        return (-1j * dt) * h0_matvec(np.asarray(x).reshape(-1))

    def rmatvec(x: np.ndarray) -> np.ndarray:
        return (1j * dt) * h0_rmatvec(np.asarray(x).reshape(-1))

    def matmat(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return (-1j * dt) * np.column_stack(
            [h0_matvec(X[:, j]) for j in range(X.shape[1])]
        )

    def rmatmat(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        return (1j * dt) * np.column_stack(
            [h0_rmatvec(X[:, j]) for j in range(X.shape[1])]
        )

    operator = LinearOperator(
        (dim, dim),
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
        dtype=complex,
    )
    return expm_multiply(operator, psi_flat, traceA=(-1j * dt) * h0_trace)


def propagate_noisy_fk_split_midpoint(
    instance: dict,
    V_diag: list[np.ndarray],
    epsilon: float,
    seed: int,
    n_noise: int,
    steps_per_segment: int,
    gamma: float,
    Omega: float,
    renormalize_each_step: bool,
    noise_model: str,
    norm_method: str,
    norm_iters: int,
    scalar_noise: complex,
    fk_include_hx: bool = True,
) -> dict:
    """Full-space FK midpoint H0 steps with piecewise dense-noise exponentials."""
    total_time, h0_factory = _full_h0_factory(
        "NHMIS-FKQAA", instance, V_diag, None, gamma, Omega, fk_include_hx=fk_include_hx
    )
    grid = compute_time_grid(
        instance["L"],
        steps_per_segment=steps_per_segment,
        n_noise=n_noise,
        total_time=total_time,
    )
    psi_flat = initial_state_full(instance["work_dim"], instance["clock_dim"]).reshape(-1)
    rng = np.random.default_rng(seed)
    noise = PiecewiseConstantNoise(
        instance["full_dim"],
        epsilon,
        rng,
        noise_model=noise_model,
        norm_method=norm_method,
        norm_iters=norm_iters,
        scalar_noise=scalar_noise,
    )
    current_noise_idx: int | None = None
    interval_steps = 0
    norm_values = []

    def flush_noise_block() -> None:
        nonlocal psi_flat, interval_steps
        if current_noise_idx is None or interval_steps == 0:
            return
        psi_flat = _apply_noise_only_exponential(psi_flat, grid["dt"] * interval_steps, noise)
        if noise.norm_R:
            norm_values.append(noise.norm_R)
        if renormalize_each_step:
            norm = np.linalg.norm(psi_flat)
            if norm <= 0.0 or not np.isfinite(norm):
                raise RuntimeError("state norm is not finite and positive")
            psi_flat = psi_flat / norm
        interval_steps = 0

    for t_mid, noise_idx in zip(grid["t_mid"], grid["noise_indices"]):
        noise_idx = int(noise_idx)
        if current_noise_idx is None:
            current_noise_idx = noise_idx
            noise.get(noise_idx)
        elif noise_idx != current_noise_idx:
            flush_noise_block()
            current_noise_idx = noise_idx
            noise.get(noise_idx)
        h0_matvec, h0_rmatvec, h0_trace = h0_factory(float(t_mid))
        psi_flat = _apply_h0_exponential_step(
            psi_flat, grid["dt"], h0_matvec, h0_rmatvec, h0_trace
        )
        if renormalize_each_step:
            norm = np.linalg.norm(psi_flat)
            if norm <= 0.0 or not np.isfinite(norm):
                raise RuntimeError("state norm is not finite and positive")
            psi_flat = psi_flat / norm
        interval_steps += 1
    flush_noise_block()
    success_index = instance["x_star_index"] * instance["clock_dim"] + instance["L"]
    p_success, final_norm = _success_from_flat_state(psi_flat, success_index)
    return {
        "state": psi_flat,
        "p_success": p_success,
        "final_norm": final_norm,
        "mean_norm_R": float(np.mean(norm_values)) if norm_values else 0.0,
        "n_time_steps": grid["n_time_steps"],
    }


def propagate_noisy_hd_split_midpoint(
    algorithm: str,
    instance: dict,
    V_diag: list[np.ndarray],
    U: np.ndarray | None,
    epsilon: float,
    seed: int,
    n_noise: int,
    steps_per_segment: int,
    gamma: float,
    Omega: float,
    renormalize_each_step: bool,
    noise_model: str,
    norm_method: str,
    norm_iters: int,
    scalar_noise: complex,
) -> dict:
    """Full-space HDQAA midpoint H0 steps with piecewise dense-noise exponentials."""
    grid = compute_time_grid(
        instance["L"],
        steps_per_segment=steps_per_segment,
        n_noise=n_noise,
        total_time=gamma * instance["L"] / Omega,
    )
    psi = initial_state_full(instance["work_dim"], instance["clock_dim"])
    phase = np.exp(-1j * Omega * grid["dt"])
    coeff = phase - 1.0
    rng = np.random.default_rng(seed)
    noise = PiecewiseConstantNoise(
        instance["full_dim"],
        epsilon,
        rng,
        noise_model=noise_model,
        norm_method=norm_method,
        norm_iters=norm_iters,
        scalar_noise=scalar_noise,
    )
    current_noise_idx: int | None = None
    interval_steps = 0
    norm_values = []

    def flush_noise_block() -> None:
        nonlocal psi, interval_steps
        if current_noise_idx is None or interval_steps == 0:
            return
        flat = psi.reshape(-1)
        flat = _apply_noise_only_exponential(flat, grid["dt"] * interval_steps, noise)
        psi = flat.reshape(instance["work_dim"], instance["clock_dim"])
        if noise.norm_R:
            norm_values.append(noise.norm_R)
        if renormalize_each_step:
            norm = np.linalg.norm(flat)
            if norm <= 0.0 or not np.isfinite(norm):
                raise RuntimeError("state norm is not finite and positive")
            psi = psi / norm
        interval_steps = 0

    for step, (segment, local_s, noise_idx) in enumerate(
        zip(grid["segment_indices"], grid["local_s"], grid["noise_indices"])
    ):
        del step
        noise_idx = int(noise_idx)
        if current_noise_idx is None:
            current_noise_idx = noise_idx
            noise.get(noise_idx)
        elif noise_idx != current_noise_idx:
            flush_noise_block()
            current_noise_idx = noise_idx
            noise.get(noise_idx)

        l = int(segment) + 1
        left = l - 1
        right = l
        theta = theta_schedule(local_s)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        a = psi[:, left].copy()
        b = psi[:, right].copy()
        psi *= phase
        if algorithm == "NHMIS-HDQAA":
            v = V_diag[int(segment)]
            h_a = sin_theta * sin_theta * a - sin_theta * cos_theta * b / v
            h_b = -sin_theta * cos_theta * v * a + cos_theta * cos_theta * b
        else:
            if U is None:
                raise ValueError("U is required for HMIS-HDQAA")
            h_a = sin_theta * sin_theta * a - sin_theta * cos_theta * (U.conj().T @ b)
            h_b = -sin_theta * cos_theta * (U @ a) + cos_theta * cos_theta * b
        psi[:, left] = a + coeff * h_a
        psi[:, right] = b + coeff * h_b
        interval_steps += 1
        if renormalize_each_step:
            norm = np.linalg.norm(psi.reshape(-1))
            if norm <= 0.0 or not np.isfinite(norm):
                raise RuntimeError("state norm is not finite and positive")
            psi /= norm
    flush_noise_block()
    success_index = instance["x_star_index"] * instance["clock_dim"] + instance["L"]
    p_success, final_norm = _success_from_flat_state(psi.reshape(-1), success_index)
    return {
        "state": psi.reshape(-1),
        "p_success": p_success,
        "final_norm": final_norm,
        "mean_norm_R": float(np.mean(norm_values)) if norm_values else 0.0,
        "n_time_steps": grid["n_time_steps"],
    }


def propagate_noisy_dynamics_midpoint(
    algorithm: str,
    epsilon: float,
    seed: int,
    instance_name: str = "paper_m2",
    n_noise: int = 70,
    steps_per_segment: int = 8,
    integrator: str = "split",
    gamma: float = 10.0,
    p: float = 2.0,
    q: float = 4.0,
    Omega: float = 1.0,
    renormalize_each_step: bool = True,
    noise_model: str = "ginibre",
    norm_method: str = "power",
    norm_iters: int = 8,
    scalar_noise: complex = 1.0 + 0.25j,
    fk_include_hx: bool = True,
) -> dict:
    """Run one full-space noisy realization with midpoint exponentials."""
    if integrator not in {"split", "combined"}:
        raise ValueError("integrator must be 'split' or 'combined'")
    instance = build_full_space_instance(instance_name, p=p, q=q, gamma=gamma)
    bits = bitstrings_array(instance["n"])
    V_diag = all_gate_eigenvalues(instance["gates"], bits, p=p, q=q)
    U = (
        build_grover_unitary(instance["work_dim"], instance["x_star_index"])
        if algorithm == "HMIS-HDQAA"
        else None
    )
    grid = compute_time_grid(
        instance["L"],
        steps_per_segment=steps_per_segment,
        n_noise=n_noise,
        total_time=(
            gamma * instance["L"]
            if algorithm == "NHMIS-FKQAA"
            else gamma * instance["L"] / Omega
        ),
    )
    if epsilon == 0.0 and algorithm in {"NHMIS-HDQAA", "HMIS-HDQAA"}:
        result = propagate_noiseless_hd_midpoint(
            algorithm,
            instance,
            V_diag,
            U,
            steps_per_segment=steps_per_segment,
            gamma=gamma,
            Omega=Omega,
            renormalize_each_step=renormalize_each_step,
        )
        result["instance"] = instance
        return result
    if epsilon > 0.0 and integrator == "split" and algorithm in {"NHMIS-HDQAA", "HMIS-HDQAA"}:
        result = propagate_noisy_hd_split_midpoint(
            algorithm,
            instance,
            V_diag,
            U,
            epsilon=epsilon,
            seed=seed,
            n_noise=n_noise,
            steps_per_segment=steps_per_segment,
            gamma=gamma,
            Omega=Omega,
            renormalize_each_step=renormalize_each_step,
            noise_model=noise_model,
            norm_method=norm_method,
            norm_iters=norm_iters,
            scalar_noise=scalar_noise,
        )
        result["instance"] = instance
        return result
    if epsilon > 0.0 and integrator == "split" and algorithm == "NHMIS-FKQAA":
        result = propagate_noisy_fk_split_midpoint(
            instance,
            V_diag,
            epsilon=epsilon,
            seed=seed,
            n_noise=n_noise,
            steps_per_segment=steps_per_segment,
            gamma=gamma,
            Omega=Omega,
            renormalize_each_step=renormalize_each_step,
            noise_model=noise_model,
            norm_method=norm_method,
            norm_iters=norm_iters,
            scalar_noise=scalar_noise,
            fk_include_hx=fk_include_hx,
        )
        result["instance"] = instance
        return result
    total_time, h0_factory = _full_h0_factory(
        algorithm, instance, V_diag, U, gamma, Omega, fk_include_hx=fk_include_hx
    )
    psi0 = initial_state_full(instance["work_dim"], instance["clock_dim"]).reshape(-1)
    success_index = instance["x_star_index"] * instance["clock_dim"] + instance["L"]
    result = propagate_state_midpoint(
        psi0,
        h0_factory,
        total_time,
        grid["n_time_steps"],
        epsilon,
        seed,
        success_index=success_index,
        n_noise=n_noise,
        renormalize_each_step=renormalize_each_step,
        noise_model=noise_model,
        norm_method=norm_method,
        norm_iters=norm_iters,
        scalar_noise=scalar_noise,
    )
    result["n_time_steps"] = grid["n_time_steps"]
    result["instance"] = instance
    return result


def propagate_noisy_dynamics_dense_exact(
    algorithm: str,
    instance_name: str,
    epsilon: float,
    seed: int,
    n_noise: int,
    steps_per_segment: int,
    gamma: float = 10.0,
    p: float = 2.0,
    q: float = 4.0,
    Omega: float = 1.0,
    noise_model: str = "ginibre",
    norm_method: str = "exact",
    norm_iters: int = 64,
    integrator: str = "combined",
    renormalize_each_step: bool = True,
    scalar_noise: complex = 1.0 + 0.25j,
    allow_large_dense_exact: bool = False,
    fk_include_hx: bool = True,
) -> dict:
    """Run a small-instance dense exact midpoint propagation."""
    if integrator != "combined":
        raise ValueError("dense exact diagnostic supports only combined integrator")
    instance = build_full_space_instance(instance_name, p=p, q=q, gamma=gamma)
    if instance["full_dim"] > 1000 and not allow_large_dense_exact:
        raise RuntimeError(
            f"dense exact is intended for small instances; full_dim={instance['full_dim']}"
        )
    bits = bitstrings_array(instance["n"])
    V_diag = all_gate_eigenvalues(instance["gates"], bits, p=p, q=q)
    U = (
        build_grover_unitary(instance["work_dim"], instance["x_star_index"])
        if algorithm == "HMIS-HDQAA"
        else None
    )
    total_time, h0_factory = _full_h0_factory(
        algorithm, instance, V_diag, U, gamma, Omega, fk_include_hx=fk_include_hx
    )
    grid = compute_time_grid(
        instance["L"],
        steps_per_segment=steps_per_segment,
        n_noise=n_noise,
        total_time=total_time,
    )
    psi = initial_state_full(instance["work_dim"], instance["clock_dim"]).reshape(-1)
    rng = np.random.default_rng(seed)
    noise = PiecewiseConstantNoise(
        instance["full_dim"],
        epsilon,
        rng,
        noise_model=noise_model,
        norm_method=norm_method,
        norm_iters=norm_iters,
        scalar_noise=scalar_noise,
    )
    norm_values = []
    start = time.perf_counter()
    for t_mid, noise_idx in zip(grid["t_mid"], grid["noise_indices"]):
        h0_matvec, _h0_rmatvec, _h0_trace = h0_factory(float(t_mid))
        H0 = dense_matrix_from_matvec(h0_matvec, instance["full_dim"])
        noise.get(int(noise_idx))
        if noise.R is not None:
            H_mid = H0 + noise.scaled * noise.R
        elif noise.scalar != 0.0:
            H_mid = H0 + noise.scalar * np.eye(instance["full_dim"], dtype=complex)
        else:
            H_mid = H0
        if noise.norm_R:
            norm_values.append(noise.norm_R)
        psi = expm((-1j * grid["dt"]) * H_mid) @ psi
        if renormalize_each_step:
            norm = np.linalg.norm(psi)
            if norm <= 0.0 or not np.isfinite(norm):
                raise RuntimeError("state norm is not finite and positive")
            psi = psi / norm

    success_index = instance["x_star_index"] * instance["clock_dim"] + instance["L"]
    p_success, final_norm = _success_from_flat_state(psi, success_index)
    return {
        "state": psi,
        "p_success": p_success,
        "final_norm": final_norm,
        "mean_norm_R": float(np.mean(norm_values)) if norm_values else 0.0,
        "n_time_steps": grid["n_time_steps"],
        "runtime_seconds": time.perf_counter() - start,
        "instance": instance,
    }


def baseline_tolerance(algorithm: str, p0_reference: float) -> float:
    """Tolerance for midpoint baseline compared with validation reference."""
    if algorithm == "NHMIS-FKQAA":
        return 5e-4
    if algorithm == "NHMIS-HDQAA":
        return 1e-4
    if algorithm == "HMIS-HDQAA":
        return max(1e-5, 1e-3 * abs(float(p0_reference)))
    raise ValueError(f"unknown algorithm {algorithm}")


def reduced_fk_reference_paper_m2(gamma: float = 10.0, p: float = 2.0, q: float = 4.0) -> float:
    """Experiment-one reduced finite-time FK reference for CK m=2."""
    info = ck_graph_info(2)
    r = int(info["n"])
    gates = build_ck_gate_sequence(2, r)
    logZ, logw_star = compute_prefix_weights_ck(2, gates, p=p, q=q)
    c = solve_empty_fk_clock_ivp(L=len(gates), gamma=gamma)
    return float(success_nhmis_fk(c, logZ, logw_star))


def compute_baseline(
    algorithm: str,
    instance_name: str = "paper_m2",
    gamma: float = 10.0,
    p: float = 2.0,
    q: float = 4.0,
    Omega: float = 1.0,
    n_noise: int = 70,
    steps_per_segment: int = 8,
    baseline_mode: str = "midpoint",
    integrator: str = "split",
    validate_tolerance: bool = True,
    out_dir: str | None = None,
    force: bool = False,
    fk_include_hx: bool = True,
) -> dict:
    """Compute/cache reference and midpoint no-noise baselines."""
    if baseline_mode not in {"midpoint", "reference"}:
        raise ValueError("baseline_mode must be 'midpoint' or 'reference'")
    h0_model_version = h0_model_version_for_algorithm(algorithm, fk_include_hx)
    fk_input_term = fk_input_term_for_algorithm(algorithm, fk_include_hx)
    path = (
        exp3_baseline_cache_path(
            out_dir,
            algorithm,
            instance_name=instance_name,
            steps_per_segment=steps_per_segment,
            integrator=integrator,
            fk_include_hx=fk_include_hx,
        )
        if out_dir is not None
        else None
    )
    if path is not None:
        cached = cached_success(path, force=force)
        if (
            cached is not None
            and float(cached.get("gamma", np.nan)) == float(gamma)
            and float(cached.get("p", np.nan)) == float(p)
            and float(cached.get("q", np.nan)) == float(q)
            and float(cached.get("Omega", np.nan)) == float(Omega)
            and cached.get("instance_name", "paper_m2") == instance_name
            and int(cached.get("steps_per_segment", -1)) == int(steps_per_segment)
            and cached.get("integrator_version") == INTEGRATOR_VERSION
            and cached.get("h0_model_version") == h0_model_version
            and normalize_fk_input_term(algorithm, cached.get("fk_input_term")) == fk_input_term
            and cached.get("integrator", "split") == integrator
        ):
            return cached
    start = time.perf_counter()
    row = {
        "instance_name": instance_name,
        "algorithm": algorithm,
        "gamma": gamma,
        "p": p,
        "q": q,
        "Omega": Omega,
        "n_noise": n_noise,
        "steps_per_segment": steps_per_segment,
        "integrator_version": INTEGRATOR_VERSION,
        "h0_model_version": h0_model_version,
        "fk_input_term": fk_input_term,
        "baseline_mode": baseline_mode,
        "p0": np.nan,
        "integrator": integrator,
        "p0_reference_source": "",
        "p0_reference": np.nan,
        "p0_midpoint": np.nan,
        "baseline_abs_error": np.nan,
        "baseline_rel_error": np.nan,
        "baseline_tolerance": np.nan,
        "runtime_seconds": 0.0,
        "status": "failed",
        "error_message": "",
        "timestamp": now_iso(),
        "cache_hit": False,
    }
    with _threadpool_limit_context():
        try:
            midpoint = propagate_noisy_dynamics_midpoint(
                algorithm,
                epsilon=0.0,
                seed=0,
                instance_name=instance_name,
                n_noise=n_noise,
                steps_per_segment=steps_per_segment,
                integrator=integrator,
                gamma=gamma,
                p=p,
                q=q,
                Omega=Omega,
                noise_model="none",
                fk_include_hx=fk_include_hx,
            )
            p0_midpoint = float(midpoint["p_success"])
            if instance_name == "paper_m2":
                if algorithm == "NHMIS-FKQAA":
                    if fk_include_hx:
                        p0_reference = reduced_fk_reference_paper_m2(gamma=gamma, p=p, q=q)
                        p0_reference_source = "experiment_one_reduced_fk_ivp"
                    else:
                        p0_reference = p0_midpoint
                        p0_reference_source = "midpoint_self_reference_no_fk_input"
                elif algorithm == "NHMIS-HDQAA":
                    result = simulate_full_nhmis_hd(gamma=gamma, p=p, q=q, Omega=Omega)
                    p0_reference = result["p_NHHD_full"]
                    p0_reference_source = "full_space_validation"
                elif algorithm == "HMIS-HDQAA":
                    result = simulate_full_hmis_hd(gamma=gamma, Omega=Omega)
                    p0_reference = result["p_HHD_full"]
                    p0_reference_source = "full_space_validation"
                else:
                    raise ValueError(f"unknown algorithm {algorithm}")
            else:
                p0_reference = p0_midpoint
                p0_reference_source = "midpoint_self_reference"
            p0 = p0_midpoint if baseline_mode == "midpoint" else p0_reference
            if p0 <= 0.0 or not np.isfinite(p0):
                raise RuntimeError("baseline probability is not finite and positive")
            abs_error = abs(float(p0_midpoint) - float(p0_reference))
            rel_error = abs_error / max(abs(float(p0_reference)), 1e-300)
            tolerance = baseline_tolerance(algorithm, p0_reference)
            if validate_tolerance and abs_error > tolerance:
                raise RuntimeError(
                    f"midpoint baseline mismatch for {algorithm}: "
                    f"abs_error={abs_error:g}, tolerance={tolerance:g}"
                )
            row.update(
                {
                    "p0": float(p0),
                    "p0_reference": float(p0_reference),
                    "p0_midpoint": float(p0_midpoint),
                    "baseline_abs_error": float(abs_error),
                    "baseline_rel_error": float(rel_error),
                    "baseline_tolerance": float(tolerance),
                    "h0_model_version": h0_model_version,
                    "fk_input_term": fk_input_term,
                    "p0_reference_source": p0_reference_source,
                    "n_time_steps": int(midpoint.get("n_time_steps", 0)),
                    "runtime_seconds": time.perf_counter() - start,
                    "status": "success",
                }
            )
        except Exception as exc:
            row.update(
                {
                    "runtime_seconds": time.perf_counter() - start,
                    "error_message": repr(exc),
                }
            )
    if path is not None:
        save_json(row, path)
    return row


def _exp3_task(args: tuple) -> dict:
    (
        instance_name,
        algorithm,
        epsilon,
        realization,
        seed,
        n_noise,
        steps_per_segment,
        n_real,
        p0,
        p0_reference,
        p0_midpoint,
        baseline_abs_error,
        baseline_rel_error,
        baseline_mode,
        p0_reference_source,
        integrator,
        out_dir,
        gamma,
        p,
        q,
        Omega,
        renormalize_each_step,
        noise_model,
        norm_method,
        norm_iters,
        seed_mode,
        fk_include_hx,
        force,
    ) = args
    h0_model_version = h0_model_version_for_algorithm(algorithm, fk_include_hx)
    fk_input_term = fk_input_term_for_algorithm(algorithm, fk_include_hx)
    path = exp3_task_cache_path(
        out_dir,
        algorithm,
        epsilon,
        realization,
        n_noise,
        instance_name=instance_name,
        steps_per_segment=steps_per_segment,
        noise_model=noise_model,
        integrator=integrator,
        norm_method=norm_method,
        norm_iters=norm_iters,
        seed_mode=seed_mode,
        fk_include_hx=fk_include_hx,
    )
    cached = cached_success(path, force=force)
    if (
        cached is not None
        and cached.get("h0_model_version") == h0_model_version
        and normalize_fk_input_term(algorithm, cached.get("fk_input_term")) == fk_input_term
    ):
        return cached
    start = time.perf_counter()
    task_seed = derive_seed(seed, algorithm, epsilon, realization, seed_mode=seed_mode)
    instance = build_full_space_instance(instance_name, p=p, q=q, gamma=gamma)
    row = {
        "instance_name": instance_name,
        "algorithm": algorithm,
        "epsilon": float(epsilon),
        "realization": int(realization),
        "seed": task_seed,
        "seed_mode": seed_mode,
        "n_noise": int(n_noise),
        "steps_per_segment": int(steps_per_segment),
        "n_time_steps": np.nan,
        "n_real": int(n_real),
        "n": int(instance["n"]),
        "r": int(instance["r"]),
        "L": int(instance["L"]),
        "full_dim": int(instance["full_dim"]),
        "p_raw": np.nan,
        "p0": float(p0),
        "p0_reference": float(p0_reference),
        "p0_midpoint": float(p0_midpoint),
        "baseline_abs_error": float(baseline_abs_error),
        "baseline_rel_error": float(baseline_rel_error),
        "baseline_mode": baseline_mode,
        "integrator_version": INTEGRATOR_VERSION,
        "h0_model_version": h0_model_version,
        "fk_input_term": fk_input_term,
        "p0_reference_source": p0_reference_source,
        "integrator": integrator,
        "integrator_note": "diagnostic_only" if integrator == "split" else "formal",
        "norm_method": norm_method,
        "norm_iters": int(norm_iters),
        "p_normalized": np.nan,
        "final_norm": np.nan,
        "runtime_seconds": 0.0,
        "status": "failed",
        "error_message": "",
        "cache_hit": False,
    }
    with _threadpool_limit_context():
        try:
            result = propagate_noisy_dynamics_midpoint(
                algorithm,
                float(epsilon),
                task_seed,
                instance_name=instance_name,
                n_noise=n_noise,
                steps_per_segment=steps_per_segment,
                integrator=integrator,
                gamma=gamma,
                p=p,
                q=q,
                Omega=Omega,
                renormalize_each_step=renormalize_each_step,
                noise_model=noise_model,
                norm_method=norm_method,
                norm_iters=norm_iters,
                fk_include_hx=fk_include_hx,
            )
            p_raw = float(result["p_success"])
            p_normalized = p_raw / float(p0)
            if p_raw < 0.0 or p_normalized < 0.0 or not np.isfinite(p_normalized):
                raise RuntimeError("invalid noisy success probability")
            row.update(
                {
                    "p_raw": p_raw,
                    "p_normalized": p_normalized,
                    "final_norm": float(result["final_norm"]),
                    "n_time_steps": int(result.get("n_time_steps", 0)),
                    "runtime_seconds": time.perf_counter() - start,
                    "status": "success",
                }
            )
        except Exception as exc:
            row.update(
                {
                    "runtime_seconds": time.perf_counter() - start,
                    "error_message": repr(exc),
                }
            )
    save_json(row, path)
    return row


def build_exp3_tasks(
    instance_name: str,
    algorithms: list[str],
    eps_list: list[float],
    n_real: int,
    seed: int,
    n_noise: int,
    steps_per_segment: int,
    baseline_by_algorithm: dict[str, dict],
    out_dir: str,
    gamma: float,
    p: float,
    q: float,
    Omega: float,
    renormalize_each_step: bool,
    noise_model: str,
    integrator: str,
    norm_method: str,
    norm_iters: int,
    seed_mode: str = "per_algorithm",
    fk_include_hx: bool = True,
    force: bool = False,
) -> list[tuple]:
    """Return fine-grained (algorithm, epsilon, realization) task tuples."""
    return [
        (
            instance_name,
            algorithm,
            float(epsilon),
            realization,
            seed,
            n_noise,
            steps_per_segment,
            n_real,
            baseline_by_algorithm[algorithm]["p0"],
            baseline_by_algorithm[algorithm]["p0_reference"],
            baseline_by_algorithm[algorithm]["p0_midpoint"],
            baseline_by_algorithm[algorithm]["baseline_abs_error"],
            baseline_by_algorithm[algorithm]["baseline_rel_error"],
            baseline_by_algorithm[algorithm]["baseline_mode"],
            baseline_by_algorithm[algorithm].get("p0_reference_source", ""),
            integrator,
            out_dir,
            gamma,
            p,
            q,
            Omega,
            renormalize_each_step,
            noise_model,
            norm_method,
            norm_iters,
            seed_mode,
            fk_include_hx,
            force,
        )
        for algorithm in algorithms
        for epsilon in eps_list
        for realization in range(n_real)
    ]


def summarize_realizations(rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    """Compute median and IQR statistics for experiment-three rows."""
    df = pd.DataFrame(rows, columns=REALIZATION_COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    output = []
    success_df = df[df["status"] == "success"].copy()
    success_df["instance_name"] = success_df["instance_name"].fillna("paper_m2")
    success_df["seed_mode"] = success_df["seed_mode"].fillna("per_algorithm")
    success_df["n"] = success_df["n"].fillna(0).astype(int)
    success_df["r"] = success_df["r"].fillna(0).astype(int)
    success_df["L"] = success_df["L"].fillna(0).astype(int)
    success_df["full_dim"] = success_df["full_dim"].fillna(0).astype(int)
    success_df["integrator"] = success_df["integrator"].fillna("split")
    success_df["integrator_note"] = success_df["integrator_note"].fillna(
        success_df["integrator"].map(
            lambda value: "diagnostic_only" if value == "split" else "formal"
        )
    )
    success_df["norm_method"] = success_df["norm_method"].fillna("power")
    success_df["norm_iters"] = success_df["norm_iters"].fillna(8).astype(int)
    success_df["integrator_version"] = success_df["integrator_version"].fillna(
        INTEGRATOR_VERSION
    )
    success_df["h0_model_version"] = success_df["h0_model_version"].fillna(
        H0_MODEL_VERSION
    )
    success_df["fk_input_term"] = [
        normalize_fk_input_term(algorithm, value)
        for algorithm, value in zip(success_df["algorithm"], success_df["fk_input_term"])
    ]
    success_df["p0_reference_source"] = success_df["p0_reference_source"].fillna("")
    success_df["baseline_mode"] = success_df["baseline_mode"].fillna("midpoint")
    group_cols = [
        "algorithm",
        "epsilon",
        "instance_name",
        "seed_mode",
        "h0_model_version",
        "fk_input_term",
        "p0_reference_source",
        "integrator",
        "integrator_note",
        "norm_method",
        "norm_iters",
    ]
    for group_key, group in success_df.groupby(group_cols):
        (
            algorithm,
            epsilon,
            instance_name,
            seed_mode,
            h0_model_version,
            fk_input_term,
            p0_reference_source,
            integrator,
            integrator_note,
            norm_method,
            norm_iters,
        ) = group_key
        raw = group["p_raw"].to_numpy(dtype=float)
        normed = group["p_normalized"].to_numpy(dtype=float)
        raw = raw[np.isfinite(raw) & (raw >= 0.0)]
        normed = normed[np.isfinite(normed) & (normed >= 0.0)]
        if len(raw) == 0 or len(normed) == 0:
            continue
        output.append(
            {
                "algorithm": algorithm,
                "instance_name": instance_name,
                "epsilon": float(epsilon),
                "n_success": int(len(group)),
                "seed_mode": seed_mode,
                "n": int(group["n"].iloc[0]),
                "r": int(group["r"].iloc[0]),
                "L": int(group["L"].iloc[0]),
                "full_dim": int(group["full_dim"].iloc[0]),
                "p0": float(group["p0"].iloc[0]),
                "p0_reference": float(group["p0_reference"].iloc[0]),
                "p0_midpoint": float(group["p0_midpoint"].iloc[0]),
                "baseline_abs_error": float(group["baseline_abs_error"].iloc[0]),
                "baseline_rel_error": float(group["baseline_rel_error"].iloc[0]),
                "baseline_mode": group["baseline_mode"].iloc[0],
                "integrator_version": group["integrator_version"].iloc[0],
                "h0_model_version": h0_model_version,
                "fk_input_term": fk_input_term,
                "p0_reference_source": p0_reference_source,
                "integrator": integrator,
                "integrator_note": integrator_note,
                "norm_method": norm_method,
                "norm_iters": int(norm_iters),
                "median_p_raw": float(np.median(raw)),
                "min_p_raw": float(np.min(raw)),
                "q1_p_raw": float(np.quantile(raw, 0.25)),
                "q3_p_raw": float(np.quantile(raw, 0.75)),
                "max_p_raw": float(np.max(raw)),
                "median_p_normalized": float(np.median(normed)),
                "min_p_normalized": float(np.min(normed)),
                "q1_p_normalized": float(np.quantile(normed, 0.25)),
                "q3_p_normalized": float(np.quantile(normed, 0.75)),
                "max_p_normalized": float(np.max(normed)),
            }
        )
    return pd.DataFrame(output, columns=SUMMARY_COLUMNS).sort_values(
        ["instance_name", "algorithm", "epsilon", "integrator", "norm_method", "norm_iters"]
    )


BASELINE_COLUMNS = [
    "algorithm",
    "p0_reference",
    "p0_midpoint",
    "p0_reference_source",
    "baseline_abs_error",
    "baseline_rel_error",
    "baseline_status",
    "steps_per_segment",
    "integrator",
    "h0_model_version",
    "fk_input_term",
    "norm_iters",
]


def write_exp3_baselines(
    baseline_by_algorithm: dict[str, dict],
    out_dir: str,
    norm_iters: int,
) -> str:
    """Write experiment-three baseline comparison table."""
    rows = []
    for algorithm in ALGORITHMS:
        if algorithm not in baseline_by_algorithm:
            continue
        row = baseline_by_algorithm[algorithm]
        tolerance = baseline_tolerance(algorithm, float(row["p0_reference"]))
        status = "PASS" if float(row["baseline_abs_error"]) <= tolerance else "FAIL"
        rows.append(
            {
                "algorithm": algorithm,
                "p0_reference": float(row["p0_reference"]),
                "p0_midpoint": float(row["p0_midpoint"]),
                "p0_reference_source": row.get("p0_reference_source", ""),
                "baseline_abs_error": float(row["baseline_abs_error"]),
                "baseline_rel_error": float(row["baseline_rel_error"]),
                "baseline_status": status,
                "steps_per_segment": int(row["steps_per_segment"]),
                "integrator": row.get("integrator", "split"),
                "h0_model_version": row.get("h0_model_version", H0_MODEL_VERSION),
                "fk_input_term": normalize_fk_input_term(
                    algorithm, row.get("fk_input_term")
                ),
                "norm_iters": int(norm_iters),
            }
        )
    path = os.path.join(out_dir, "experiment_three_baselines.csv")
    pd.DataFrame(rows, columns=BASELINE_COLUMNS).to_csv(path, index=False)
    return path


def write_exp3_outputs(rows: list[dict], out_dir: str) -> tuple[str, str]:
    """Write realizations and summary CSVs."""
    os.makedirs(out_dir, exist_ok=True)
    real_path = os.path.join(out_dir, "realizations.csv")
    summary_path = os.path.join(out_dir, "summary.csv")
    df = pd.DataFrame(rows, columns=REALIZATION_COLUMNS)
    df.to_csv(real_path, index=False)
    summarize_realizations(df).to_csv(summary_path, index=False)
    return real_path, summary_path


def plot_experiment_three_from_summary(
    summary_csv: str,
    out_dir: str,
    png_filename: str | None = None,
    pdf_filename: str | None = None,
):
    """Plot raw success medians and min-max bands from summary CSV only."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogFormatterMathtext, LogLocator, NullFormatter

    _enable_miktex_text_rendering()
    df = pd.read_csv(summary_csv)
    os.makedirs(out_dir, exist_ok=True)
    png_path = png_filename or os.path.join(out_dir, "experiment_three_success.png")
    pdf_path = pdf_filename or os.path.join(out_dir, "experiment_three_success.pdf")
    marker = "o"
    tick_size = 28

    fig, ax = plt.subplots(figsize=(10.4, 7.4))
    positive_seen = False
    plotted_algorithms = []
    positive_eps_values: list[float] = []
    for algorithm in ALGORITHMS:
        sub = df[(df["algorithm"] == algorithm) & (df["epsilon"] > 0)].sort_values("epsilon")
        if sub.empty:
            continue
        x = sub["epsilon"].to_numpy(dtype=float)
        median = sub["median_p_raw"].to_numpy(dtype=float)
        band_low_column = "min_p_raw" if "min_p_raw" in sub.columns else "q1_p_raw"
        band_high_column = "max_p_raw" if "max_p_raw" in sub.columns else "q3_p_raw"
        band_low = sub[band_low_column].to_numpy(dtype=float)
        band_high = sub[band_high_column].to_numpy(dtype=float)
        if not np.any(median > 0.0):
            continue
        median_plot = np.where(median > 0.0, median, np.nan)
        band_low_plot = np.where(band_low > 0.0, band_low, np.nan)
        band_high_plot = np.where(band_high > 0.0, band_high, np.nan)
        positive_seen = True
        plotted_algorithms.append(algorithm)
        positive_eps_values.extend(x.tolist())
        color = ALGORITHM_COLORS_PRL_MUTED.get(algorithm)
        ax.fill_between(
            x,
            band_low_plot,
            band_high_plot,
            color=color,
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x,
            median_plot,
            marker=marker,
            color=color,
            linestyle="-",
            linewidth=4.0,
            markersize=9.6,
            label=ALGORITHM_DISPLAY.get(algorithm, algorithm),
            zorder=3,
        )

    if not positive_seen:
        positive_eps = df.loc[df["epsilon"] > 0, "epsilon"].to_numpy(dtype=float)
        dummy_x = float(np.min(positive_eps)) if len(positive_eps) else 1e-8
        ax.plot([dummy_x], [1.0], alpha=0.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=tick_size, length=6)
    ax.tick_params(axis="both", which="minor", length=3)
    y_major_ticks = [10.0**exp for exp in range(0, -8, -1)]
    y_labeled_ticks = {10.0**exp for exp in range(0, -8, -2)}
    ax.yaxis.set_major_locator(FixedLocator(y_major_ticks))
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _pos: LogFormatterMathtext(base=10.0)(value)
            if value in y_labeled_ticks
            else ""
        )
    )
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())
    if positive_eps_values:
        major_exps = list(range(-15, 0))
        major_ticks = [10.0**exp for exp in major_exps]
        labeled_ticks = {major_ticks[i] for i in range(0, len(major_ticks), 4)}
        ax.set_xlim(1e-15 / np.sqrt(10.0), 1e-1 * np.sqrt(10.0))
        ax.xaxis.set_major_locator(FixedLocator(major_ticks))
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _pos: LogFormatterMathtext(base=10.0)(value)
                if value in labeled_ticks
                else ""
            )
        )
        for label in ax.xaxis.get_ticklabels():
            label.set_rotation(0)
            label.set_ha("center")
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
        ax.xaxis.set_minor_formatter(NullFormatter())
    if not positive_seen:
        ax.set_ylim(1e-16, 1.0)
    ax.grid(False)

    hd_sub = df[
        (df["algorithm"] == "NHMIS-HDQAA") & (df["epsilon"] >= 1e-4) & (df["epsilon"] > 0)
    ].sort_values("epsilon")
    if not hd_sub.empty:
        hd_x = hd_sub["epsilon"].to_numpy(dtype=float)
        hd_median = hd_sub["median_p_raw"].to_numpy(dtype=float)
        hd_low_column = "min_p_raw" if "min_p_raw" in hd_sub.columns else "q1_p_raw"
        hd_high_column = "max_p_raw" if "max_p_raw" in hd_sub.columns else "q3_p_raw"
        hd_low = hd_sub[hd_low_column].to_numpy(dtype=float)
        hd_high = hd_sub[hd_high_column].to_numpy(dtype=float)
        finite = np.isfinite(hd_low) & np.isfinite(hd_high) & np.isfinite(hd_median)
        if np.any(finite):
            color = ALGORITHM_COLORS_PRL_MUTED["NHMIS-HDQAA"]
            inset = ax.inset_axes([0.10, 0.09, 0.42, 0.30])
            inset.fill_between(
                hd_x[finite],
                hd_low[finite],
                hd_high[finite],
                color=color,
                alpha=0.24,
                linewidth=0,
            )
            inset.plot(
                hd_x[finite],
                hd_median[finite],
                marker=marker,
                color=color,
                linestyle="-",
                linewidth=4.0,
                markersize=9.6,
            )
            y_min = float(np.nanmin(hd_low[finite]))
            y_max = float(np.nanmax(hd_high[finite]))
            y_pad = max((y_max - y_min) * 0.20, 1e-3)
            inset.set_xscale("log")
            inset.set_xlim(
                float(np.min(hd_x[finite])) / np.sqrt(10.0),
                float(np.max(hd_x[finite])) * np.sqrt(10.0),
            )
            inset.set_ylim(y_min - y_pad, min(1.005, y_max + y_pad))
            inset.xaxis.set_major_locator(FixedLocator([1e-4, 1e-3, 1e-2, 1e-1]))
            inset.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
            inset.xaxis.set_minor_formatter(NullFormatter())
            inset.tick_params(axis="both", which="major", labelsize=22, length=4.5, pad=1.5)
            inset.tick_params(axis="both", which="minor", length=2.0)
            for label in inset.xaxis.get_ticklabels():
                label.set_rotation(0)
                label.set_ha("center")
            inset.grid(False)

    fig.subplots_adjust(
        left=MAIN_AXES_RECT[0],
        bottom=MAIN_AXES_RECT[1],
        right=MAIN_AXES_RECT[2],
        top=MAIN_AXES_RECT[3],
    )
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.12)
    return fig


def run_experiment_three(
    eps_list: list[float],
    n_real: int,
    instance_name: str = "paper_m2",
    n_noise: int = 70,
    steps_per_segment: int = 8,
    seed: int = 1234,
    out_dir: str = "outputs/experiment_three_noisy",
    algorithms: list[str] | None = None,
    gamma: float = 10.0,
    p: float = 2.0,
    q: float = 4.0,
    Omega: float = 1.0,
    renormalize_each_step: bool = True,
    noise_model: str = "ginibre",
    integrator: str = "combined",
    norm_method: str = "power",
    norm_iters: int = 32,
    seed_mode: str = "per_algorithm",
    baseline_mode: str = "midpoint",
    validate_baseline: bool = True,
    max_workers: int | None = None,
    reserve_cores: int = 1,
    serial: bool = False,
    force: bool = False,
    fk_include_hx: bool = True,
) -> list[dict]:
    """Run/cache experiment-three noisy realizations."""
    if algorithms is None:
        algorithms = ALGORITHMS
    if seed_mode not in {"per_algorithm", "paired_noise"}:
        raise ValueError("seed_mode must be 'per_algorithm' or 'paired_noise'")
    os.makedirs(os.path.join(out_dir, "cache"), exist_ok=True)

    baseline_by_algorithm = {}
    for algorithm in algorithms:
        baseline = compute_baseline(
            algorithm,
            instance_name=instance_name,
            gamma=gamma,
            p=p,
            q=q,
            Omega=Omega,
            n_noise=n_noise,
            steps_per_segment=steps_per_segment,
            baseline_mode=baseline_mode,
            integrator=integrator,
            validate_tolerance=validate_baseline,
            out_dir=out_dir,
            force=force,
            fk_include_hx=fk_include_hx,
        )
        if baseline.get("status") != "success":
            raise RuntimeError(f"baseline failed for {algorithm}: {baseline.get('error_message')}")
        baseline_by_algorithm[algorithm] = baseline
    write_exp3_baselines(baseline_by_algorithm, out_dir, norm_iters)

    tasks = build_exp3_tasks(
        instance_name,
        algorithms,
        eps_list,
        n_real,
        seed,
        n_noise,
        steps_per_segment,
        baseline_by_algorithm,
        out_dir,
        gamma,
        p,
        q,
        Omega,
        renormalize_each_step,
        noise_model,
        integrator,
        norm_method,
        norm_iters,
        seed_mode,
        fk_include_hx,
        force,
    )
    rows: list[dict] = []

    if serial:
        for task in tasks:
            rows.append(_exp3_task(task))
            write_exp3_outputs(rows, out_dir)
    else:
        workers = max_workers if max_workers is not None else default_exp3_max_workers(reserve_cores)
        print(f"using ProcessPoolExecutor with max_workers = {workers}, reserve_cores = {reserve_cores}")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_exp3_task, task) for task in tasks]
            for future in as_completed(futures):
                rows.append(future.result())
                write_exp3_outputs(rows, out_dir)

    real_path, summary_path = write_exp3_outputs(rows, out_dir)
    fig = plot_experiment_three_from_summary(summary_path, out_dir)
    fig.clf()
    print(f"saved {real_path}")
    print(f"saved {summary_path}")
    return rows
