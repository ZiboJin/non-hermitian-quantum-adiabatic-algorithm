"""Full work-space x clock-space validation for experiment one at m=2.

This script intentionally uses direct small full-space simulations only for
the m=2 validation case.  The production scaling code remains reduced.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.special import logsumexp

try:
    from .experiment_one_ck import (
        build_ck_gate_sequence,
        ck_graph_info,
        closed_form_ideal_success_ck,
        compute_prefix_weights_bruteforce,
        compute_prefix_weights_ck,
        prepare_ck_instance,
        run_one_instance,
        solve_empty_fk_clock_ivp,
        solve_empty_hd_segment,
        success_hmis_hd,
        success_nhmis_fk,
        theta_schedule,
    )
except ImportError:  # pragma: no cover - used when run as a script path.
    from experiment_one_ck import (
        build_ck_gate_sequence,
        ck_graph_info,
        closed_form_ideal_success_ck,
        compute_prefix_weights_bruteforce,
        compute_prefix_weights_ck,
        prepare_ck_instance,
        run_one_instance,
        solve_empty_fk_clock_ivp,
        solve_empty_hd_segment,
        success_hmis_hd,
        success_nhmis_fk,
        theta_schedule,
    )


VALIDATION_COLUMNS = [
    "check_name",
    "expected_or_reduced_value",
    "full_or_computed_value",
    "abs_error",
    "rel_error",
    "tolerance",
    "status",
]


def m2_dimensions() -> tuple[int, int, int, int]:
    """Return n, L, work_dim, full_dim for the m=2 validation instance."""
    m = 2
    info = ck_graph_info(m)
    n = info["n"]
    r = n
    L = len(build_ck_gate_sequence(m, r))
    work_dim = 1 << n
    return n, L, work_dim, work_dim * (L + 1)


def x_star_index_for_m(m: int) -> int:
    """Return the integer bit-string index of the CK target state."""
    x_star = ck_graph_info(m)["x_star"]
    return int(sum(bit << i for i, bit in enumerate(x_star)))


def bit_table(n: int) -> np.ndarray:
    """Return bit table with convention bit i = (x >> i) & 1."""
    return ((np.arange(1 << n, dtype=np.int64)[:, None] >> np.arange(n)) & 1)


def gate_eigenvalues(gate: tuple, bits: np.ndarray, p: float, q: float) -> np.ndarray:
    """Eigenvalues v_l(x) for one diagonal MIS gate."""
    if gate[0] == "A":
        return np.where(bits[:, gate[1]] == 1, p, 1.0).astype(float)
    i, j = gate[1], gate[2]
    violated = bits[:, i] * bits[:, j]
    return np.where(violated == 1, 1.0, q).astype(float)


def all_gate_eigenvalues(
    gates: list[tuple], n: int, p: float, q: float
) -> np.ndarray:
    """Stack all gate eigenvalue vectors, shape (L, 2**n)."""
    bits = bit_table(n)
    return np.vstack([gate_eigenvalues(gate, bits, p, q) for gate in gates])


def build_grover_unitary(n: int, x_star_index: int) -> np.ndarray:
    """Build U_G = D O for the unique marked state."""
    work_dim = 1 << n
    plus = np.full(work_dim, 1.0 / math.sqrt(work_dim), dtype=complex)
    diffusion = 2.0 * np.outer(plus, plus.conj()) - np.eye(work_dim, dtype=complex)
    oracle = np.eye(work_dim, dtype=complex)
    oracle[x_star_index, x_star_index] = -1.0
    return diffusion @ oracle


def pure_grover_success_direct(n: int, L: int, x_star_index: int) -> float:
    """Directly apply U_G^L to |+> and return marked-state probability."""
    work_dim = 1 << n
    state = np.full(work_dim, 1.0 / math.sqrt(work_dim), dtype=complex)
    unitary = build_grover_unitary(n, x_star_index)
    for _ in range(L):
        state = unitary @ state
    return float(abs(state[x_star_index]) ** 2)


def pure_grover_success_formula(n: int, L: int) -> float:
    """Closed-form unique-solution Grover success probability."""
    alpha = np.arcsin(2.0 ** (-0.5 * n))
    return float(np.sin((2 * L + 1) * alpha) ** 2)


def simulate_full_nhmis_fk(
    gamma: float = 10.0,
    p: float = 2.0,
    q: float = 4.0,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> dict:
    """Direct full-space NHMIS-FKQAA solve for m=2."""
    m = 2
    info = ck_graph_info(m)
    n = info["n"]
    r = n
    gates = build_ck_gate_sequence(m, r)
    L = len(gates)
    work_dim = 1 << n
    clock_dim = L + 1
    x_star_index = x_star_index_for_m(m)
    v = all_gate_eigenvalues(gates, n, p, q)
    inv_v = 1.0 / v
    T = gamma * L
    flip_indices = [np.arange(work_dim) ^ (1 << bit) for bit in range(n)]

    init_diag = np.ones(clock_dim, dtype=float)
    init_diag[0] = 0.0
    y0 = np.zeros((work_dim, clock_dim), dtype=complex)
    y0[:, 0] = 1.0 / math.sqrt(work_dim)

    def rhs(s: float, y: np.ndarray) -> np.ndarray:
        psi = y.reshape(work_dim, clock_dim)
        prop = np.zeros_like(psi)
        for idx in range(L):
            left = idx
            right = idx + 1
            prop[:, left] += 0.5 * psi[:, left]
            prop[:, right] += 0.5 * psi[:, right]
            prop[:, right] += -0.5 * v[idx] * psi[:, left]
            prop[:, left] += -0.5 * inv_v[idx] * psi[:, right]
        out = (1.0 - s) * (psi * init_diag[None, :]) + s * prop
        hx_clock0 = 0.5 * n * psi[:, 0]
        for indices in flip_indices:
            hx_clock0 = hx_clock0 - 0.5 * psi[indices, 0]
        out[:, 0] += hx_clock0
        return (-1j * T * out).reshape(-1)

    sol = solve_ivp(
        rhs,
        (0.0, 1.0),
        y0.reshape(-1),
        method="DOP853",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"full FK solve failed: {sol.message}")
    psi_final = sol.y[:, -1].reshape(work_dim, clock_dim)
    norm = float(np.sum(np.abs(psi_final) ** 2))
    return {
        "p_FK_full": float(abs(psi_final[x_star_index, L]) ** 2 / norm),
        "full_norm": norm,
    }


def simulate_full_nhmis_hd(
    gamma: float = 10.0,
    p: float = 2.0,
    q: float = 4.0,
    Omega: float = 1.0,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> dict:
    """Direct full-space sequential NHMIS-HDQAA solve for m=2."""
    m = 2
    info = ck_graph_info(m)
    n = info["n"]
    r = n
    gates = build_ck_gate_sequence(m, r)
    L = len(gates)
    work_dim = 1 << n
    clock_dim = L + 1
    x_star_index = x_star_index_for_m(m)
    v_all = all_gate_eigenvalues(gates, n, p, q)
    tau = gamma / Omega
    psi = np.zeros((work_dim, clock_dim), dtype=complex)
    psi[:, 0] = 1.0 / math.sqrt(work_dim)

    for idx in range(L):
        v = v_all[idx]
        y0 = np.concatenate([psi[:, idx], psi[:, idx + 1]])

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            s = t / tau
            theta = theta_schedule(s)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            a = y[:work_dim]
            b = y[work_dim:]
            da = -1j * Omega * (
                sin_theta * sin_theta * a - sin_theta * cos_theta * b / v
            )
            db = -1j * Omega * (
                -sin_theta * cos_theta * v * a + cos_theta * cos_theta * b
            )
            return np.concatenate([da, db])

        sol = solve_ivp(
            rhs,
            (0.0, tau),
            y0,
            method="DOP853",
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"full NHHD segment {idx + 1} failed: {sol.message}")
        final_pair = sol.y[:, -1]
        psi[:, idx] = final_pair[:work_dim]
        psi[:, idx + 1] = final_pair[work_dim:]

    norm = float(np.sum(np.abs(psi) ** 2))
    return {
        "p_NHHD_full": float(abs(psi[x_star_index, L]) ** 2 / norm),
        "full_norm": norm,
    }


def simulate_full_hmis_hd(
    gamma: float = 10.0,
    Omega: float = 1.0,
    rtol: float = 1e-9,
    atol: float = 1e-11,
) -> dict:
    """Direct full-space sequential HMIS-HDQAA solve for m=2."""
    m = 2
    info = ck_graph_info(m)
    n = info["n"]
    r = n
    L = len(build_ck_gate_sequence(m, r))
    work_dim = 1 << n
    clock_dim = L + 1
    x_star_index = x_star_index_for_m(m)
    unitary = build_grover_unitary(n, x_star_index)
    unitary_dag = unitary.conj().T
    tau = gamma / Omega
    psi = np.zeros((work_dim, clock_dim), dtype=complex)
    psi[:, 0] = 1.0 / math.sqrt(work_dim)

    for idx in range(L):
        y0 = np.concatenate([psi[:, idx], psi[:, idx + 1]])

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            s = t / tau
            theta = theta_schedule(s)
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            a = y[:work_dim]
            b = y[work_dim:]
            da = -1j * Omega * (
                sin_theta * sin_theta * a - sin_theta * cos_theta * (unitary_dag @ b)
            )
            db = -1j * Omega * (
                -sin_theta * cos_theta * (unitary @ a) + cos_theta * cos_theta * b
            )
            return np.concatenate([da, db])

        sol = solve_ivp(
            rhs,
            (0.0, tau),
            y0,
            method="DOP853",
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"full HHD segment {idx + 1} failed: {sol.message}")
        final_pair = sol.y[:, -1]
        psi[:, idx] = final_pair[:work_dim]
        psi[:, idx + 1] = final_pair[work_dim:]

    norm = float(np.sum(np.abs(psi) ** 2))
    return {
        "p_HHD_full": float(abs(psi[x_star_index, L]) ** 2),
        "full_norm": norm,
    }


def _rel_error(expected: float, actual: float) -> float:
    return abs(expected - actual) / max(abs(expected), 1e-300)


def make_check(
    check_name: str,
    expected,
    actual,
    tolerance: float,
    abs_error: float | None = None,
    rel_error: float | None = None,
) -> dict:
    """Build one validation-report row."""
    if abs_error is None:
        try:
            abs_error = abs(float(expected) - float(actual))
        except Exception:
            abs_error = 0.0 if expected == actual else np.inf
    if rel_error is None:
        try:
            rel_error = _rel_error(float(expected), float(actual))
        except Exception:
            rel_error = 0.0 if expected == actual else np.inf
    status = "PASS" if abs_error <= tolerance else "FAIL"
    return {
        "check_name": check_name,
        "expected_or_reduced_value": expected,
        "full_or_computed_value": actual,
        "abs_error": abs_error,
        "rel_error": rel_error,
        "tolerance": tolerance,
        "status": status,
    }


def run_validation(gamma: float, p: float, q: float, Omega: float) -> tuple[list[dict], dict]:
    """Run all m=2 reduced, exact-value, and full-space checks."""
    m = 2
    info = ck_graph_info(m)
    n = info["n"]
    r = n
    gates = build_ck_gate_sequence(m, r)
    L = len(gates)
    x_star_index = x_star_index_for_m(m)
    work_dim = 1 << n
    full_dim = work_dim * (L + 1)
    logZ, logw_star = compute_prefix_weights_ck(m, gates, p=p, q=q)
    logZ_brute, logw_brute = compute_prefix_weights_bruteforce(m, gates, p=p, q=q)
    instance = prepare_ck_instance(m, p=p, q=q)
    reduced = run_one_instance(
        m=m,
        gamma=gamma,
        p=p,
        q=q,
        Omega=Omega,
        fk_method="expm",
        fk_step_factor=16,
    )
    c_ivp = solve_empty_fk_clock_ivp(L=L, gamma=gamma)
    p_fk_reduced_ivp = success_nhmis_fk(c_ivp, logZ, logw_star)
    f, g = solve_empty_hd_segment(gamma=gamma, Omega=Omega)
    p_grover_formula = pure_grover_success_formula(n, L)
    p_grover_direct = pure_grover_success_direct(n, L, x_star_index)
    ideal_prefix = float(np.exp(logw_star[L] - logZ[L]))
    ideal_closed = closed_form_ideal_success_ck(m=m, r=r, p=p, q=q)

    full_fk = simulate_full_nhmis_fk(gamma=gamma, p=p, q=q)
    full_nhhd = simulate_full_nhmis_hd(gamma=gamma, p=p, q=q, Omega=Omega)
    full_hhd = simulate_full_hmis_hd(gamma=gamma, Omega=Omega)

    checks = [
        make_check("CK graph count", "m=2,n=5,E=9,L=70", f"m={m},n={n},E={info['num_edges']},L={L}", 0.0),
        make_check("full-space dimension", 2272, full_dim, 0.0),
        make_check("gate sequence length", 70, len(gates), 0.0),
        make_check("x_star index", 3, x_star_index, 0.0),
        make_check("initial prefix logZ[0]", 0.0, logZ[0], 1e-14),
        make_check("initial prefix logw_star[0]", -5 * math.log(2.0), logw_star[0], 1e-14),
        make_check(
            "optimized prefix vs brute force logZ max error",
            0.0,
            float(np.max(np.abs(logZ - logZ_brute))),
            1e-10,
        ),
        make_check(
            "optimized prefix vs brute force logw max error",
            0.0,
            float(np.max(np.abs(logw_star - logw_brute))),
            1e-10,
        ),
        make_check("known final logZ[L]", 135.16858063256245, logZ[L], 1e-9),
        make_check("known final logw_star[L]", 135.16370020918922, logw_star[L], 1e-9),
        make_check("known ideal_NH_circuit", 0.9951314665424426, ideal_prefix, 1e-10),
        make_check("closed-form ideal circuit", ideal_closed, ideal_prefix, 1e-10),
        make_check("HD two-level norm", 1.0, abs(f) ** 2 + abs(g) ** 2, 1e-8),
        make_check("HD |f|^2 reference", 0.9999450780736, abs(f) ** 2, 1e-6),
        make_check("HD |g|^2 reference", 0.0000549219263, abs(g) ** 2, 1e-6),
        make_check("pure Grover direct vs formula", p_grover_formula, p_grover_direct, 1e-12),
        make_check("full-space NHMIS-FKQAA vs reduced ivp", p_fk_reduced_ivp, full_fk["p_FK_full"], 5e-6),
        make_check("full-space NHMIS-HDQAA vs reduced", reduced["p_NHHD"], full_nhhd["p_NHHD_full"], 1e-6),
        make_check("full-space HMIS-HDQAA vs reduced", reduced["p_HHD"], full_hhd["p_HHD_full"], 1e-6),
        make_check("FK full norm finite positive", 1.0, float(np.isfinite(full_fk["full_norm"]) and full_fk["full_norm"] > 0), 0.0),
        make_check("NHHD full norm finite positive", 1.0, float(np.isfinite(full_nhhd["full_norm"]) and full_nhhd["full_norm"] > 0), 0.0),
        make_check("HHD full norm close to one", 1.0, full_hhd["full_norm"], 1e-8),
    ]

    probabilities = [
        p_fk_reduced_ivp,
        reduced["p_FK"],
        full_fk["p_FK_full"],
        reduced["p_NHHD"],
        full_nhhd["p_NHHD_full"],
        reduced["p_HHD"],
        full_hhd["p_HHD_full"],
        ideal_prefix,
        p_grover_direct,
    ]
    range_ok = all(0.0 <= value <= 1.0 for value in probabilities)
    checks.append(make_check("probability range", 1.0, float(range_ok), 0.0))

    values = {
        "m": m,
        "n": n,
        "num_edges": info["num_edges"],
        "L": L,
        "work_dim": work_dim,
        "full_dim": full_dim,
        "x_star_index": x_star_index,
        "ideal_NH_circuit": ideal_prefix,
        "p_FK_reduced_ivp": p_fk_reduced_ivp,
        "p_FK_reduced_expm16": reduced["p_FK"],
        "p_FK_full": full_fk["p_FK_full"],
        "p_NHHD_reduced": reduced["p_NHHD"],
        "p_NHHD_full": full_nhhd["p_NHHD_full"],
        "p_HHD_reduced": reduced["p_HHD"],
        "p_HHD_full": full_hhd["p_HHD_full"],
        "p_Grover_formula": p_grover_formula,
        "p_Grover_direct": p_grover_direct,
        "logZ_L": logZ[L],
        "logw_star_L": logw_star[L],
        "absf2": abs(f) ** 2,
        "absg2": abs(g) ** 2,
        "fk_full_norm": full_fk["full_norm"],
        "nhhd_full_norm": full_nhhd["full_norm"],
        "hhd_full_norm": full_hhd["full_norm"],
        "ideal_from_instance": instance["ideal_NH_circuit"],
    }
    return checks, values


def write_markdown_report(checks: list[dict], values: dict, filename: str) -> None:
    """Write a readable black-box validation report."""
    overall = "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
    lines = [
        f"# Full-Space Validation Report, m=2",
        "",
        f"Overall validation status: {overall}",
        "",
        "## Instance",
        "",
        f"- m = {values['m']}",
        f"- n = {values['n']}",
        f"- |E| = {values['num_edges']}",
        f"- L = {values['L']}",
        f"- work dimension = {values['work_dim']}",
        f"- full dimension = {values['full_dim']}",
        f"- x_star index = {values['x_star_index']}",
        "",
        "## Key Probabilities",
        "",
        f"- p_FK_full = {values['p_FK_full']:.15g}",
        f"- p_FK_reduced_ivp = {values['p_FK_reduced_ivp']:.15g}",
        f"- p_FK_reduced_expm16 = {values['p_FK_reduced_expm16']:.15g}",
        f"- p_NHHD_full = {values['p_NHHD_full']:.15g}",
        f"- p_NHHD_reduced = {values['p_NHHD_reduced']:.15g}",
        f"- p_HHD_full = {values['p_HHD_full']:.15g}",
        f"- p_HHD_reduced = {values['p_HHD_reduced']:.15g}",
        "",
        "## Known-Value Checks",
        "",
        f"- logZ[L] = {values['logZ_L']:.15g}",
        f"- logw_star[L] = {values['logw_star_L']:.15g}",
        f"- ideal_NH_circuit = {values['ideal_NH_circuit']:.15g}",
        f"- pure Grover formula = {values['p_Grover_formula']:.15g}",
        f"- pure Grover direct simulation = {values['p_Grover_direct']:.15g}",
        f"- |f|^2 = {values['absf2']:.15g}",
        f"- |g|^2 = {values['absg2']:.15g}",
        "",
        "## Full-Space Norms",
        "",
        f"- FK final norm = {values['fk_full_norm']:.15g}",
        f"- NHHD final norm = {values['nhhd_full_norm']:.15g}",
        f"- HHD final norm = {values['hhd_full_norm']:.15g}",
        "",
        "## Check Table",
        "",
        "| check | expected/reduced | full/computed | abs error | tolerance | status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in checks:
        lines.append(
            "| {check_name} | {expected_or_reduced_value} | {full_or_computed_value} | "
            "{abs_error:.3e} | {tolerance:.3e} | {status} |".format(**row)
        )
    Path(filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate m=2 using full-space simulations.")
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--q", type=float, default=4.0)
    parser.add_argument("--Omega", type=float, default=1.0)
    parser.add_argument("--out-dir", type=str, default="outputs/validation_m2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    checks, values = run_validation(args.gamma, args.p, args.q, args.Omega)
    csv_path = os.path.join(args.out_dir, "full_space_validation_m2.csv")
    md_path = os.path.join(args.out_dir, "full_space_validation_m2.md")
    pd.DataFrame(checks, columns=VALIDATION_COLUMNS).to_csv(csv_path, index=False)
    write_markdown_report(checks, values, md_path)
    overall = "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL"
    print(f"Overall validation status: {overall}")
    print(f"saved {csv_path}")
    print(f"saved {md_path}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
