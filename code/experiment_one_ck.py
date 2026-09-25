"""Experiment one: finite-time noiseless CK-graph success probabilities.

This module implements the reduced simulations needed for comparing
NHMIS-FKQAA, NHMIS-HDQAA, and HMIS-HDQAA on CK graphs.  The non-Hermitian
MIS circuit is diagonal, so prefix norms are computed in log space using a
CK-specific compressed sum instead of enumerating all ``2**n`` bit strings.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from typing import Iterable

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.sparse import diags
from scipy.sparse.linalg import expm_multiply
from scipy.special import logsumexp


Gate = tuple
RESULT_COLUMNS = [
    "m",
    "n",
    "r",
    "L",
    "gamma",
    "Omega",
    "absf2",
    "absg2",
    "p_FK",
    "p_NHHD",
    "p_HHD",
    "ideal_NH_circuit",
    "fk_method",
    "fk_num_steps",
    "fk_norm_error",
]
HMIS_EXTENDED_COLUMNS = [
    "m",
    "n",
    "L",
    "gamma",
    "Omega",
    "absf2",
    "p_Grover",
    "p_HHD",
]
FK_CONVERGENCE_COLUMNS = [
    "m",
    "n",
    "L",
    "gamma",
    "fk_step_factor",
    "fk_num_steps",
    "p_FK",
    "p_NHHD",
    "p_HHD",
    "fk_norm_error",
]


def default_max_workers(reserve_cores: int = 1) -> int:
    """Return a conservative process count, leaving cores for the system."""
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - reserve_cores)


def _threadpool_limit_context():
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        return nullcontext()
    return threadpool_limits(limits=1)


def ck_graph_info(m: int) -> dict:
    """Return the CK graph size data and the unique MIS bit string."""
    if m < 1:
        raise ValueError("m must be a positive integer")

    n = 4 * m - 3
    return {
        "m": m,
        "n": n,
        "num_edges": 3 * (m * m - 1),
        "x_star": tuple([1] * m + [0] * (n - m)),
    }


def build_ck_gate_sequence(m: int, r: int) -> list[Gate]:
    """Build the prescribed CK non-Hermitian MIS gate sequence."""
    if m < 1:
        raise ValueError("m must be a positive integer")
    if r < 0:
        raise ValueError("r must be nonnegative")

    n = ck_graph_info(m)["n"]
    one_round: list[Gate] = []

    for i in range(m):
        for a in range(m - 1):
            for u in range(3):
                one_round.append(("B", i, m + 3 * a + u))

    for a in range(m - 1):
        v0 = m + 3 * a
        v1 = m + 3 * a + 1
        v2 = m + 3 * a + 2
        one_round.append(("B", v0, v1))
        one_round.append(("B", v0, v2))
        one_round.append(("B", v1, v2))

    for i in range(n):
        one_round.append(("A", i))

    return one_round * r


def _right_location(vertex: int, m: int) -> tuple[int, int]:
    tri, u = divmod(vertex - m, 3)
    if tri < 0 or tri >= m - 1 or u < 0 or u >= 3:
        raise ValueError(f"invalid right-side CK vertex {vertex}")
    return tri, u


def _internal_pair_id(u: int, v: int) -> int:
    pair = tuple(sorted((u, v)))
    if pair == (0, 1):
        return 0
    if pair == (0, 2):
        return 1
    if pair == (1, 2):
        return 2
    raise ValueError(f"invalid triangle edge endpoints {u}, {v}")


def _validate_gate(gate: Gate) -> None:
    if not gate:
        raise ValueError("empty gate")
    if gate[0] == "A" and len(gate) != 2:
        raise ValueError(f"invalid A gate {gate}")
    if gate[0] == "B" and len(gate) != 3:
        raise ValueError(f"invalid B gate {gate}")
    if gate[0] not in {"A", "B"}:
        raise ValueError(f"unknown gate kind {gate[0]}")


def _compute_logZ_from_counts(
    m: int,
    a_left: np.ndarray,
    a_right: np.ndarray,
    b_cross: np.ndarray,
    b_internal: np.ndarray,
    left_patterns: np.ndarray,
    tri_patterns: np.ndarray,
    log_p: float,
    log_q: float,
) -> float:
    n = 4 * m - 3
    two_log_p = 2.0 * log_p
    two_log_q = 2.0 * log_q
    left_terms = two_log_p * (left_patterns @ a_left)
    y0 = tri_patterns[:, 0]
    y1 = tri_patterns[:, 1]
    y2 = tri_patterns[:, 2]

    for tri in range(m - 1):
        right_terms = two_log_p * (tri_patterns @ a_right[tri])

        internal_counts = b_internal[tri]
        internal_satisfied = (
            internal_counts[0] * (1.0 - y0 * y1)
            + internal_counts[1] * (1.0 - y0 * y2)
            + internal_counts[2] * (1.0 - y1 * y2)
        )

        cross_counts = b_cross[:, tri, :]
        total_cross = float(np.sum(cross_counts))
        selected_cross = left_patterns @ cross_counts
        cross_satisfied = total_cross - selected_cross @ tri_patterns.T

        tri_terms = right_terms[None, :] + two_log_q * (
            internal_satisfied[None, :] + cross_satisfied
        )
        left_terms += logsumexp(tri_terms, axis=1)

    return float(logsumexp(left_terms) - n * math.log(2.0))


def compute_prefix_weights_ck(
    m: int, gates: list[Gate], p: float = 2.0, q: float = 4.0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute CK prefix log norms without enumerating all bit strings.

    Returns ``logZ[l] = log sum_x w_l(x)`` and
    ``logw_star[l] = log w_l(x_star)`` for all prefixes ``l``.
    """
    if m < 1:
        raise ValueError("m must be a positive integer")
    if p <= 0.0 or q <= 0.0:
        raise ValueError("p and q must be positive")

    L = len(gates)
    n = ck_graph_info(m)["n"]
    log_p = math.log(p)
    log_q = math.log(q)
    minus_n_log2 = -n * math.log(2.0)

    a_left = np.zeros(m, dtype=np.int64)
    a_right = np.zeros((m - 1, 3), dtype=np.int64)
    b_cross = np.zeros((m, m - 1, 3), dtype=np.int64)
    b_internal = np.zeros((m - 1, 3), dtype=np.int64)

    left_patterns = (
        (np.arange(1 << m, dtype=np.int64)[:, None] >> np.arange(m)) & 1
    ).astype(float)
    tri_patterns = (
        (np.arange(8, dtype=np.int64)[:, None] >> np.arange(3)) & 1
    ).astype(float)

    logZ = np.empty(L + 1, dtype=float)
    logw_star = np.empty(L + 1, dtype=float)
    logZ[0] = 0.0
    logw_star[0] = minus_n_log2

    total_left_A = 0
    total_B = 0

    for step, gate in enumerate(gates, start=1):
        _validate_gate(gate)
        kind = gate[0]

        if kind == "A":
            vertex = gate[1]
            if vertex < 0 or vertex >= n:
                raise ValueError(f"invalid A gate vertex {vertex}")
            if vertex < m:
                a_left[vertex] += 1
                total_left_A += 1
            else:
                tri, u = _right_location(vertex, m)
                a_right[tri, u] += 1
        else:
            i, j = gate[1], gate[2]
            if i < 0 or i >= n or j < 0 or j >= n:
                raise ValueError(f"invalid B gate endpoints {i}, {j}")
            total_B += 1

            i_left = i < m
            j_left = j < m
            if i_left and not j_left:
                tri, u = _right_location(j, m)
                b_cross[i, tri, u] += 1
            elif j_left and not i_left:
                tri, u = _right_location(i, m)
                b_cross[j, tri, u] += 1
            elif not i_left and not j_left:
                tri_i, u_i = _right_location(i, m)
                tri_j, u_j = _right_location(j, m)
                if tri_i != tri_j:
                    raise ValueError(f"right-side endpoints are in different triangles: {i}, {j}")
                pair_id = _internal_pair_id(u_i, u_j)
                b_internal[tri_i, pair_id] += 1
            else:
                raise ValueError(f"CK graph has no left-left edge: {i}, {j}")

        logZ[step] = _compute_logZ_from_counts(
            m,
            a_left,
            a_right,
            b_cross,
            b_internal,
            left_patterns,
            tri_patterns,
            log_p,
            log_q,
        )
        logw_star[step] = (
            minus_n_log2
            + 2.0 * log_p * total_left_A
            + 2.0 * log_q * total_B
        )

    return logZ, logw_star


def compute_prefix_weights_bruteforce(
    m: int, gates: list[Gate], p: float = 2.0, q: float = 4.0
) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force prefix weights for small CK graphs, for tests only."""
    if m < 1:
        raise ValueError("m must be a positive integer")
    if p <= 0.0 or q <= 0.0:
        raise ValueError("p and q must be positive")

    n = ck_graph_info(m)["n"]
    if n > 24:
        raise ValueError("bruteforce is intended only for small test instances")

    bitstrings = (
        (np.arange(1 << n, dtype=np.int64)[:, None] >> np.arange(n)) & 1
    ).astype(np.int8)
    x_star = np.asarray(ck_graph_info(m)["x_star"], dtype=np.int8)
    star_index = int(np.flatnonzero(np.all(bitstrings == x_star, axis=1))[0])

    log_p = math.log(p)
    log_q = math.log(q)
    log_weights = np.full(1 << n, -n * math.log(2.0), dtype=float)

    L = len(gates)
    logZ = np.empty(L + 1, dtype=float)
    logw_star = np.empty(L + 1, dtype=float)
    logZ[0] = float(logsumexp(log_weights))
    logw_star[0] = float(log_weights[star_index])

    for step, gate in enumerate(gates, start=1):
        _validate_gate(gate)
        if gate[0] == "A":
            i = gate[1]
            log_weights += 2.0 * log_p * bitstrings[:, i]
        else:
            i, j = gate[1], gate[2]
            satisfied = 1 - bitstrings[:, i] * bitstrings[:, j]
            log_weights += 2.0 * log_q * satisfied

        logZ[step] = float(logsumexp(log_weights))
        logw_star[step] = float(log_weights[star_index])

    return logZ, logw_star


def theta_schedule(s):
    """Smoothstep active-link angle theta(s) for s in [0, 1]."""
    s_arr = np.asarray(s)
    u = 10.0 * s_arr**3 - 15.0 * s_arr**4 + 6.0 * s_arr**5
    return 0.5 * np.pi * u


def solve_empty_hd_segment(gamma: float, Omega: float = 1.0) -> tuple[complex, complex]:
    """Solve one empty 2x2 HD active-link segment.

    Returns ``(f, g)``, where ``f`` is the transfer amplitude and ``g`` is the
    leakage amplitude.
    """
    if gamma < 0.0:
        raise ValueError("gamma must be nonnegative")
    if Omega <= 0.0:
        raise ValueError("Omega must be positive")

    tau = gamma / Omega
    if tau == 0.0:
        return 0.0 + 0.0j, 1.0 + 0.0j

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        s = t / tau
        theta = theta_schedule(s)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        h00 = sin_theta * sin_theta
        h01 = -sin_theta * cos_theta
        h11 = cos_theta * cos_theta
        a, b = y
        return -1j * Omega * np.array(
            [h00 * a + h01 * b, h01 * a + h11 * b], dtype=complex
        )

    sol = solve_ivp(
        rhs,
        (0.0, tau),
        np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        method="DOP853",
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(f"HD segment solve failed: {sol.message}")

    g = complex(sol.y[0, -1])
    f = complex(sol.y[1, -1])
    return f, g


def solve_empty_fk_clock_ivp(
    L: int, gamma: float, rtol: float = 1e-8, atol: float = 1e-10
) -> np.ndarray:
    """Reference DOP853 solve for the empty FK clock evolution."""
    if L < 0:
        raise ValueError("L must be nonnegative")
    if gamma < 0.0:
        raise ValueError("gamma must be nonnegative")

    dim = L + 1
    y0 = np.zeros(dim, dtype=complex)
    y0[0] = 1.0 + 0.0j
    if L == 0 or gamma == 0.0:
        return y0

    T = gamma * L
    init_diag = np.ones(dim, dtype=float)
    init_diag[0] = 0.0
    clock_diag = np.ones(dim, dtype=float)
    clock_diag[0] = 0.5
    clock_diag[-1] = 0.5

    def rhs(s: float, y: np.ndarray) -> np.ndarray:
        diag = (1.0 - s) * init_diag + s * clock_diag
        out = diag * y
        off = -0.5 * s
        out[:-1] += off * y[1:]
        out[1:] += off * y[:-1]
        return -1j * T * out

    sol = solve_ivp(
        rhs,
        (0.0, 1.0),
        y0,
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"FK clock solve failed: {sol.message}")
    return sol.y[:, -1]


def _fk_expm_num_steps(
    L: int, num_steps: int | None, step_factor: int, min_steps: int
) -> int:
    if num_steps is not None:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        return int(num_steps)
    if step_factor <= 0:
        raise ValueError("step_factor must be positive")
    if min_steps <= 0:
        raise ValueError("min_steps must be positive")
    return int(max(min_steps, step_factor * L))


def solve_empty_fk_clock_expm(
    L: int,
    gamma: float,
    num_steps: int | None = None,
    step_factor: int = 8,
    min_steps: int = 200,
    renormalize: bool = False,
) -> np.ndarray:
    """Sparse midpoint-exponential solve for the empty FK clock evolution."""
    if L < 0:
        raise ValueError("L must be nonnegative")
    if gamma < 0.0:
        raise ValueError("gamma must be nonnegative")

    dim = L + 1
    c = np.zeros(dim, dtype=complex)
    c[0] = 1.0 + 0.0j
    if L == 0 or gamma == 0.0:
        return c

    steps = _fk_expm_num_steps(L, num_steps, step_factor, min_steps)
    ds = 1.0 / steps
    T = gamma * L

    init_diag = np.ones(dim, dtype=float)
    init_diag[0] = 0.0
    clock_diag = np.ones(dim, dtype=float)
    clock_diag[0] = 0.5
    clock_diag[-1] = 0.5

    for step in range(steps):
        s_mid = (step + 0.5) * ds
        diag = (1.0 - s_mid) * init_diag + s_mid * clock_diag
        off_diag = np.full(L, -0.5 * s_mid, dtype=float)
        hamiltonian = diags(
            (off_diag, diag, off_diag),
            offsets=(-1, 0, 1),
            shape=(dim, dim),
            format="csr",
        )
        c = expm_multiply((-1j * T * ds) * hamiltonian, c)
        if renormalize:
            norm = np.linalg.norm(c)
            if norm != 0.0:
                c = c / norm

    return c


def solve_empty_fk_clock(
    L: int,
    gamma: float,
    method: str = "expm",
    num_steps: int | None = None,
    step_factor: int = 8,
    min_steps: int = 200,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> np.ndarray:
    """Solve the empty FK clock evolution in dimension ``L + 1``."""
    if method == "expm":
        return solve_empty_fk_clock_expm(
            L,
            gamma,
            num_steps=num_steps,
            step_factor=step_factor,
            min_steps=min_steps,
        )
    if method == "ivp":
        return solve_empty_fk_clock_ivp(L, gamma, rtol=rtol, atol=atol)
    raise ValueError("method must be 'expm' or 'ivp'")


def _log_positive(value: float) -> float:
    return math.log(value) if value > 0.0 else -np.inf


def _log_power(log_value: float, power: int) -> float:
    if power == 0:
        return 0.0
    if np.isneginf(log_value):
        return -np.inf
    return power * log_value


def _log_powers(log_value: float, powers: np.ndarray) -> np.ndarray:
    out = np.empty_like(powers, dtype=float)
    zero_mask = powers == 0
    out[zero_mask] = 0.0
    if np.isneginf(log_value):
        out[~zero_mask] = -np.inf
    else:
        out[~zero_mask] = powers[~zero_mask] * log_value
    return out


def _prob_from_log_ratio(log_num: float, log_den: float) -> float:
    if np.isneginf(log_num):
        return 0.0
    if np.isneginf(log_den):
        return 0.0
    value = float(np.exp(log_num - log_den))
    return float(np.clip(value, 0.0, 1.0))


def success_nhmis_fk(c: np.ndarray, logZ: np.ndarray, logw_star: np.ndarray) -> float:
    """Finite-time NHMIS-FKQAA success probability."""
    L = len(logZ) - 1
    if len(c) != L + 1 or len(logw_star) != L + 1:
        raise ValueError("c, logZ, and logw_star must have compatible lengths")

    abs2 = np.abs(c) ** 2
    log_abs2 = np.full_like(abs2, -np.inf, dtype=float)
    mask = abs2 > 0.0
    log_abs2[mask] = np.log(abs2[mask])

    log_num = log_abs2[L] + logw_star[L]
    log_den = float(logsumexp(log_abs2 + logZ))
    return _prob_from_log_ratio(float(log_num), log_den)


def success_nhmis_hd(
    f: complex, g: complex, logZ: np.ndarray, logw_star: np.ndarray
) -> float:
    """Finite-time NHMIS-HDQAA success probability."""
    L = len(logZ) - 1
    if len(logw_star) != L + 1:
        raise ValueError("logZ and logw_star must have compatible lengths")

    absf2 = abs(f) ** 2
    absg2 = abs(g) ** 2
    log_absf2 = _log_positive(absf2)
    log_absg2 = _log_positive(absg2)

    log_num = _log_power(log_absf2, L) + logw_star[L]
    log_final = _log_power(log_absf2, L) + logZ[L]

    powers = np.arange(L, dtype=float)
    leak_terms = log_absg2 + _log_powers(log_absf2, powers) + logZ[:L]
    log_den = float(logsumexp(np.concatenate(([log_final], leak_terms))))
    return _prob_from_log_ratio(float(log_num), log_den)


def success_hmis_hd(f: complex, n: int, L: int) -> float:
    """Finite-time HMIS-HDQAA success probability from HD transfer and Grover."""
    if n < 1:
        raise ValueError("n must be positive")
    if L < 0:
        raise ValueError("L must be nonnegative")

    alpha = np.arcsin(2.0 ** (-0.5 * n))
    p_grover = np.sin((2 * L + 1) * alpha) ** 2
    absf2 = abs(f) ** 2
    transfer = 0.0 if absf2 == 0.0 and L > 0 else math.exp(_log_power(_log_positive(absf2), L))
    return float(np.clip(transfer * p_grover, 0.0, 1.0))


def closed_form_ideal_success_ck(
    m: int, r: int, p: float = 2.0, q: float = 4.0
) -> float:
    """Closed-form ideal non-Hermitian MIS-circuit success on CK graphs."""
    if m < 1:
        raise ValueError("m must be a positive integer")
    if r < 0:
        raise ValueError("r must be nonnegative")
    if p <= 0.0 or q <= 0.0:
        raise ValueError("p and q must be positive")

    log_p = math.log(p)
    log_q = math.log(q)
    log_num = 2.0 * r * m * log_p
    ell_terms = []

    for ell in range(m + 1):
        inner_terms = []
        for k in range(4):
            violations = ell * k + math.comb(k, 2)
            inner_terms.append(
                math.log(math.comb(3, k))
                + 2.0 * r * k * log_p
                - 2.0 * r * violations * log_q
            )
        inner = float(logsumexp(inner_terms))
        ell_terms.append(
            math.log(math.comb(m, ell))
            + 2.0 * r * ell * log_p
            + (m - 1) * inner
        )

    log_den = float(logsumexp(ell_terms))
    return _prob_from_log_ratio(log_num, log_den)


def prepare_ck_instance(m: int, p: float = 2.0, q: float = 4.0) -> dict:
    """Precompute CK data and gamma-independent prefix weights."""
    info = ck_graph_info(m)
    n = info["n"]
    r = n
    gates = build_ck_gate_sequence(m, r)
    L = len(gates)
    logZ, logw_star = compute_prefix_weights_ck(m, gates, p, q)
    ideal = _prob_from_log_ratio(float(logw_star[L]), float(logZ[L]))
    return {
        "m": m,
        "n": n,
        "r": r,
        "L": L,
        "gates": gates,
        "logZ": logZ,
        "logw_star": logw_star,
        "ideal_NH_circuit": ideal,
    }


def run_prepared_instance(
    instance: dict,
    gamma: float,
    Omega: float = 1.0,
    fk_method: str = "expm",
    fk_num_steps: int | None = None,
    fk_step_factor: int = 8,
    fk_min_steps: int = 200,
) -> dict:
    """Run finite-time algorithms using a precomputed CK instance."""
    m = int(instance["m"])
    n = int(instance["n"])
    r = int(instance["r"])
    L = int(instance["L"])
    logZ = instance["logZ"]
    logw_star = instance["logw_star"]

    f, g = solve_empty_hd_segment(gamma, Omega)
    c = solve_empty_fk_clock(
        L,
        gamma,
        method=fk_method,
        num_steps=fk_num_steps,
        step_factor=fk_step_factor,
        min_steps=fk_min_steps,
    )

    p_fk = success_nhmis_fk(c, logZ, logw_star)
    p_nhhd = success_nhmis_hd(f, g, logZ, logw_star)
    p_hhd = success_hmis_hd(f, n, L)
    actual_fk_num_steps = (
        _fk_expm_num_steps(L, fk_num_steps, fk_step_factor, fk_min_steps)
        if fk_method == "expm"
        else np.nan
    )
    fk_norm_error = float(abs(np.sum(np.abs(c) ** 2) - 1.0))

    return {
        "m": m,
        "n": n,
        "r": r,
        "L": L,
        "gamma": gamma,
        "Omega": Omega,
        "absf2": abs(f) ** 2,
        "absg2": abs(g) ** 2,
        "p_FK": p_fk,
        "p_NHHD": p_nhhd,
        "p_HHD": p_hhd,
        "ideal_NH_circuit": instance["ideal_NH_circuit"],
        "fk_method": fk_method,
        "fk_num_steps": actual_fk_num_steps,
        "fk_norm_error": fk_norm_error,
    }


def _run_all_gammas_for_m_worker(args: tuple) -> list[dict]:
    """Worker: prepare one CK instance and run all requested gammas."""
    (
        m,
        gamma_list,
        p,
        q,
        Omega,
        fk_method,
        fk_num_steps,
        fk_step_factor,
        fk_min_steps,
    ) = args
    with _threadpool_limit_context():
        instance = prepare_ck_instance(m, p=p, q=q)
        return [
            run_prepared_instance(
                instance,
                float(gamma),
                Omega=Omega,
                fk_method=fk_method,
                fk_num_steps=fk_num_steps,
                fk_step_factor=fk_step_factor,
                fk_min_steps=fk_min_steps,
            )
            for gamma in gamma_list
        ]


def run_gammas_parallel(
    m_list: Iterable[int],
    gamma_list: Iterable[float],
    p: float = 2.0,
    q: float = 4.0,
    Omega: float = 1.0,
    fk_method: str = "expm",
    fk_num_steps: int | None = None,
    fk_step_factor: int = 8,
    fk_min_steps: int = 200,
    max_workers: int | None = None,
    reserve_cores: int = 1,
) -> dict[float, list[dict]]:
    """Run multiple gammas in parallel by CK graph size ``m``."""
    m_values = [int(m) for m in m_list]
    gamma_values = [float(gamma) for gamma in gamma_list]
    workers = max_workers if max_workers is not None else default_max_workers(reserve_cores)

    output = {gamma: [] for gamma in gamma_values}
    tasks = [
        (
            m,
            gamma_values,
            p,
            q,
            Omega,
            fk_method,
            fk_num_steps,
            fk_step_factor,
            fk_min_steps,
        )
        for m in m_values
    ]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for rows in executor.map(_run_all_gammas_for_m_worker, tasks):
            for row in rows:
                output[float(row["gamma"])].append(row)

    for gamma in gamma_values:
        output[gamma].sort(key=lambda row: row["m"])
    return output


def run_one_instance(
    m: int,
    gamma: float,
    p: float = 2.0,
    q: float = 4.0,
    Omega: float = 1.0,
    fk_method: str = "expm",
    fk_num_steps: int | None = None,
    fk_step_factor: int = 8,
    fk_min_steps: int = 200,
) -> dict:
    """Run all three finite-time algorithms for one CK graph size."""
    instance = prepare_ck_instance(m, p=p, q=q)
    return run_prepared_instance(
        instance,
        gamma,
        Omega=Omega,
        fk_method=fk_method,
        fk_num_steps=fk_num_steps,
        fk_step_factor=fk_step_factor,
        fk_min_steps=fk_min_steps,
    )


def run_scaling(
    m_list: Iterable[int],
    gamma: float,
    p: float = 2.0,
    q: float = 4.0,
    Omega: float = 1.0,
    fk_method: str = "expm",
    fk_num_steps: int | None = None,
    fk_step_factor: int = 8,
    fk_min_steps: int = 200,
) -> list[dict]:
    """Run one fixed-gamma CK-size scaling."""
    instances = [prepare_ck_instance(m, p=p, q=q) for m in m_list]
    return [
        run_prepared_instance(
            instance,
            gamma,
            Omega=Omega,
            fk_method=fk_method,
            fk_num_steps=fk_num_steps,
            fk_step_factor=fk_step_factor,
            fk_min_steps=fk_min_steps,
        )
        for instance in instances
    ]


def sweep_gamma(
    m_list: Iterable[int],
    gamma_list: Iterable[float],
    p: float = 2.0,
    q: float = 4.0,
    Omega: float = 1.0,
    fk_method: str = "expm",
    fk_num_steps: int | None = None,
    fk_step_factor: int = 8,
    fk_min_steps: int = 200,
) -> dict[float, list[dict]]:
    """Run a gamma sweep, returning a mapping from gamma to result rows."""
    instances = [prepare_ck_instance(m, p=p, q=q) for m in m_list]
    return {
        float(gamma): [
            run_prepared_instance(
                instance,
                float(gamma),
                Omega=Omega,
                fk_method=fk_method,
                fk_num_steps=fk_num_steps,
                fk_step_factor=fk_step_factor,
                fk_min_steps=fk_min_steps,
            )
            for instance in instances
        ]
        for gamma in gamma_list
    }


def _convergence_for_m_worker(args: tuple) -> list[dict]:
    """Worker: prepare one CK instance and test several FK step factors."""
    (
        m,
        gamma,
        step_factors,
        p,
        q,
        Omega,
        fk_min_steps,
    ) = args
    with _threadpool_limit_context():
        instance = prepare_ck_instance(m, p=p, q=q)
        rows = []
        for step_factor in step_factors:
            result = run_prepared_instance(
                instance,
                gamma,
                Omega=Omega,
                fk_method="expm",
                fk_step_factor=int(step_factor),
                fk_min_steps=fk_min_steps,
            )
            rows.append(
                {
                    "m": result["m"],
                    "n": result["n"],
                    "L": result["L"],
                    "gamma": result["gamma"],
                    "fk_step_factor": int(step_factor),
                    "fk_num_steps": result["fk_num_steps"],
                    "p_FK": result["p_FK"],
                    "p_NHHD": result["p_NHHD"],
                    "p_HHD": result["p_HHD"],
                    "fk_norm_error": result["fk_norm_error"],
                }
            )
        return rows


def convergence_fk_step_factor(
    m_list: Iterable[int],
    gamma: float,
    step_factors: Iterable[int],
    p: float = 2.0,
    q: float = 4.0,
    Omega: float = 1.0,
    fk_min_steps: int = 200,
    max_workers: int | None = None,
    reserve_cores: int = 1,
) -> list[dict]:
    """Check finite-time FK success convergence across expm step factors."""
    m_values = [int(m) for m in m_list]
    factor_values = [int(step_factor) for step_factor in step_factors]
    workers = max_workers if max_workers is not None else default_max_workers(reserve_cores)
    tasks = [
        (m, float(gamma), factor_values, p, q, Omega, fk_min_steps)
        for m in m_values
    ]

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for worker_rows in executor.map(_convergence_for_m_worker, tasks):
            rows.extend(worker_rows)

    rows.sort(key=lambda row: (row["m"], row["fk_step_factor"]))
    return rows


def save_results_csv(results: list[dict], filename: str) -> None:
    """Save experiment rows to CSV with a stable column order."""
    directory = os.path.dirname(filename)
    if directory:
        os.makedirs(directory, exist_ok=True)
    pd.DataFrame(results, columns=RESULT_COLUMNS).to_csv(filename, index=False)


def _pdf_filename_from_png(filename: str | None, pdf_filename: str | None) -> str | None:
    if pdf_filename is not None:
        return pdf_filename
    if filename is None:
        return None
    root, _ext = os.path.splitext(filename)
    return f"{root}.pdf"


def _save_figure_outputs(fig, filename: str | None, pdf_filename: str | None) -> None:
    if filename is not None:
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(filename, dpi=300)

    pdf_path = _pdf_filename_from_png(filename, pdf_filename)
    if pdf_path is not None:
        directory = os.path.dirname(pdf_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(pdf_path)


def _positive_for_log_axis(values: np.ndarray) -> np.ndarray:
    return np.where(values > 0.0, values, np.nan)


def plot_experiment_one(results: list[dict], filename: str | None = None):
    """Plot the three finite-time success-probability curves."""
    import matplotlib.pyplot as plt

    df = pd.DataFrame(results).sort_values("n")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    curves = [
        ("p_FK", "NHMIS-FKQAA"),
        ("p_NHHD", "NHMIS-HDQAA"),
        ("p_HHD", "HMIS-HDQAA"),
    ]

    for column, label in curves:
        y = df[column].to_numpy(dtype=float)
        y = np.where(y > 0.0, y, np.nan)
        ax.plot(df["n"], y, marker="o", linewidth=1.8, label=label)

    ax.set_xlabel("CK graph size n")
    ax.set_ylabel("success probability")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax.legend()
    fig.tight_layout()

    if filename is not None:
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fig.savefig(filename, dpi=300)

    return fig


def plot_experiment_one_dual_axis(
    results: list[dict], filename: str | None = None, pdf_filename: str | None = None
):
    """Plot experiment-one results with NHMIS and HMIS on separate log axes."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    df = pd.DataFrame(results).sort_values("n")
    x = df["n"].to_numpy(dtype=int)

    fig, ax_left = plt.subplots(figsize=(6.4, 4.2))
    ax_right = ax_left.twinx()

    line_fk = ax_left.plot(
        x,
        _positive_for_log_axis(df["p_FK"].to_numpy(dtype=float)),
        marker="o",
        linewidth=1.8,
        label="NHMIS-FKQAA",
    )[0]
    line_nhhd = ax_left.plot(
        x,
        _positive_for_log_axis(df["p_NHHD"].to_numpy(dtype=float)),
        marker="s",
        linewidth=1.8,
        label="NHMIS-HDQAA",
    )[0]
    line_hhd = ax_right.plot(
        x,
        _positive_for_log_axis(df["p_HHD"].to_numpy(dtype=float)),
        marker="^",
        linewidth=1.8,
        linestyle="--",
        color="tab:green",
        label="HMIS-HDQAA",
    )[0]

    ax_left.set_xlabel(r"CK graph size $n$")
    ax_left.set_ylabel("Success probability, NHMIS")
    ax_right.set_ylabel("Success probability, HMIS-HDQAA")
    ax_left.set_yscale("log")
    ax_right.set_yscale("log")
    ax_left.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_left.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax_left.legend(
        [line_fk, line_nhhd, line_hhd],
        ["NHMIS-FKQAA", "NHMIS-HDQAA", "HMIS-HDQAA"],
        loc="best",
    )
    fig.tight_layout()

    _save_figure_outputs(fig, filename, pdf_filename)
    return fig


def _pure_grover_success(n: int, L: int) -> float:
    alpha = np.arcsin(2.0 ** (-0.5 * n))
    return float(np.sin((2 * L + 1) * alpha) ** 2)


def compute_hmis_hd_extended(
    m_list: Iterable[int], gamma: float, Omega: float = 1.0
) -> list[dict]:
    """Compute HMIS-HDQAA and pure Grover probabilities for larger CK sizes."""
    f, _g = solve_empty_hd_segment(gamma, Omega)
    absf2 = abs(f) ** 2
    rows = []
    for m in m_list:
        info = ck_graph_info(int(m))
        n = info["n"]
        num_edges = info["num_edges"]
        L = n * (n + num_edges)
        p_grover = _pure_grover_success(n, L)
        rows.append(
            {
                "m": int(m),
                "n": n,
                "L": L,
                "gamma": gamma,
                "Omega": Omega,
                "absf2": absf2,
                "p_Grover": p_grover,
                "p_HHD": success_hmis_hd(f, n, L),
            }
        )
    return rows


def plot_hmis_hd_extended_dual_axis(
    rows: list[dict], filename: str | None = None, pdf_filename: str | None = None
):
    """Plot finite-time HMIS-HDQAA and pure Grover curves on separate log axes."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    df = pd.DataFrame(rows).sort_values("n")
    x = df["n"].to_numpy(dtype=int)

    fig, ax_left = plt.subplots(figsize=(6.4, 4.2))
    ax_right = ax_left.twinx()

    line_hhd = ax_left.plot(
        x,
        _positive_for_log_axis(df["p_HHD"].to_numpy(dtype=float)),
        marker="o",
        linewidth=1.8,
        label="HMIS-HDQAA",
    )[0]
    line_grover = ax_right.plot(
        x,
        _positive_for_log_axis(df["p_Grover"].to_numpy(dtype=float)),
        marker="^",
        linewidth=1.8,
        linestyle="--",
        color="tab:orange",
        label="Pure Grover",
    )[0]

    ax_left.set_xlabel(r"CK graph size $n$")
    ax_left.set_ylabel("HMIS-HDQAA success probability")
    ax_right.set_ylabel("Pure Grover success probability")
    ax_left.set_yscale("log")
    ax_right.set_yscale("log")
    ax_left.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_left.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax_left.legend(
        [line_hhd, line_grover],
        ["HMIS-HDQAA", "Pure Grover"],
        loc="best",
    )
    fig.tight_layout()

    _save_figure_outputs(fig, filename, pdf_filename)
    return fig
