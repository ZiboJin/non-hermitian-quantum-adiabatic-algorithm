"""Resolvent pseudospectrum version of paper experiment two.

The experiment uses the smallest singular value definition

    f_H(z) = sigma_min(z I - H)

instead of the legacy random dense perturbation protocol.  For FK, the
diagonal MIS gates split the Hamiltonian into clock-chain sectors H_x, so
large-n calculations never build the full work-space x clock-space matrix.
"""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import time
import warnings
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import eig_banded
from scipy.ndimage import label as cc_label
from scipy.sparse import csr_matrix, diags, eye, issparse
from scipy.sparse.linalg import eigsh, svds
from scipy.special import logsumexp

try:
    import mpmath as mp
except ImportError:  # pragma: no cover - checked explicitly by HP entry point.
    mp = None

plt.rcParams["axes.unicode_minus"] = False

try:
    from .experiment_one_ck import build_ck_gate_sequence, ck_graph_info, default_max_workers
except ImportError:  # pragma: no cover - script-path execution
    from experiment_one_ck import build_ck_gate_sequence, ck_graph_info, default_max_workers


GATE_ORDER = "legacy_project_order: B_cross_then_B_internal_then_A"
ALGORITHMS = ["NHMIS-FKQAA", "NHMIS-HDQAA", "HMIS-HDQAA"]
TINY_FLOOR = 1e-300
LN10 = math.log(10.0)


@dataclass(frozen=True)
class GridWindow:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    nx: int
    ny: int

    def expanded(self, factor: float = 2.0) -> "GridWindow":
        cx = 0.5 * (self.x_min + self.x_max)
        cy = 0.5 * (self.y_min + self.y_max)
        hx = 0.5 * (self.x_max - self.x_min) * factor
        hy = 0.5 * (self.y_max - self.y_min) * factor
        return GridWindow(cx - hx, cx + hx, cy - hy, cy + hy, self.nx, self.ny)


class UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)
        self.contains_ground = np.zeros(size, dtype=bool)
        self.contains_excited = np.zeros(size, dtype=bool)
        self.touches_boundary = np.zeros(size, dtype=bool)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[x] != x:
            parent = int(self.parent[x])
            self.parent[x] = root
            x = parent
        return root

    def union(self, a: int, b: int) -> int:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        self.contains_ground[ra] = self.contains_ground[ra] or self.contains_ground[rb]
        self.contains_excited[ra] = self.contains_excited[ra] or self.contains_excited[rb]
        self.touches_boundary[ra] = self.touches_boundary[ra] or self.touches_boundary[rb]
        return ra


def ensure_dirs(out_dir: str | Path) -> dict[str, Path]:
    base = Path(out_dir)
    dirs = {
        "base": base,
        "figures": base / "figures",
        "data": base / "data",
        "logs": base / "logs",
        "cache": base / "cache",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def configure_logger(out_dir: str | Path, append: bool = False) -> logging.Logger:
    dirs = ensure_dirs(out_dir)
    logger = logging.getLogger("experiment_two_resolvent")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(
        dirs["logs"] / "experiment2_resolvent.log",
        mode="a" if append else "w",
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    return logger


def build_ck_graph_from_n(n: int) -> dict:
    """Return explicit CK graph data for n = 4m - 3."""
    if n <= 0 or n % 4 != 1:
        raise ValueError("CK family requires n > 0 and n % 4 == 1")
    m = (n + 3) // 4
    left_vertices = list(range(m))
    triangle_list: list[tuple[int, int, int]] = []
    right_vertices: list[int] = []
    edges: list[tuple[int, int]] = []

    for a in range(m - 1):
        tri = (m + 3 * a, m + 3 * a + 1, m + 3 * a + 2)
        triangle_list.append(tri)
        right_vertices.extend(tri)

    for i in left_vertices:
        for j in right_vertices:
            edges.append((i, j))

    for tri in triangle_list:
        v0, v1, v2 = tri
        edges.extend([(v0, v1), (v0, v2), (v1, v2)])

    x_mis = tuple([1] * m + [0] * (n - m))
    if len(edges) != 3 * (m * m - 1):
        raise RuntimeError("internal CK edge-count construction error")
    return {
        "m": m,
        "n": n,
        "left_vertices": left_vertices,
        "right_vertices": right_vertices,
        "triangle_list": triangle_list,
        "edges": edges,
        "x_mis": x_mis,
    }


def get_legacy_ck_gates(n: int, r: int | None = None) -> list[tuple]:
    graph = build_ck_graph_from_n(n)
    return build_ck_gate_sequence(graph["m"], n if r is None else int(r))


def bit_tuple(index: int, n: int) -> tuple[int, ...]:
    return tuple(int((index >> i) & 1) for i in range(n))


def bitstring_label(x: Iterable[int]) -> str:
    return "".join(str(int(b)) for b in x)


def is_independent_set(x: Iterable[int], edges: Iterable[tuple[int, int]]) -> bool:
    bits = tuple(int(v) for v in x)
    return all(not (bits[i] == 1 and bits[j] == 1) for i, j in edges)


def gate_multiplier_for_config(gate: tuple, x: Iterable[int], p: float = 2.0, q: float = 4.0) -> float:
    bits = tuple(int(v) for v in x)
    if gate[0] == "A":
        return float(p if bits[int(gate[1])] == 1 else 1.0)
    if gate[0] == "B":
        i, j = int(gate[1]), int(gate[2])
        return float(1.0 if bits[i] == 1 and bits[j] == 1 else q)
    raise ValueError(f"unknown gate kind {gate[0]!r}")


def multipliers_for_config(
    x: Iterable[int], gates: Iterable[tuple], p: float = 2.0, q: float = 4.0
) -> np.ndarray:
    return np.array([gate_multiplier_for_config(gate, x, p=p, q=q) for gate in gates], dtype=float)


def log_W_path_for_config(
    x: Iterable[int], gates: Iterable[tuple], p: float = 2.0, q: float = 4.0
) -> np.ndarray:
    """Return amplitude-level log W_l, length L + 1."""
    increments = []
    log_p = math.log(float(p))
    log_q = math.log(float(q))
    bits = tuple(int(v) for v in x)
    for gate in gates:
        if gate[0] == "A":
            increments.append(log_p if bits[int(gate[1])] == 1 else 0.0)
        elif gate[0] == "B":
            i, j = int(gate[1]), int(gate[2])
            increments.append(0.0 if bits[i] == 1 and bits[j] == 1 else log_q)
        else:
            raise ValueError(f"unknown gate kind {gate[0]!r}")
    out = np.empty(len(increments) + 1, dtype=float)
    out[0] = 0.0
    out[1:] = np.cumsum(increments)
    return out


def final_log_gain_formula(
    x: Iterable[int], edges: Iterable[tuple[int, int]], r: int, p: float = 2.0, q: float = 4.0
) -> float:
    bits = tuple(int(v) for v in x)
    satisfied = sum(1 for i, j in edges if not (bits[i] == 1 and bits[j] == 1))
    return float(r * (sum(bits) * math.log(p) + satisfied * math.log(q)))


def build_fk_sector_matrix_from_multipliers(v_list: Iterable[float], sparse: bool = True):
    """Build the FK clock-sector matrix H_x from positive gate multipliers."""
    v = np.asarray(list(v_list), dtype=float)
    if v.ndim != 1 or np.any(v <= 0.0):
        raise ValueError("v_list must be a one-dimensional positive sequence")
    L = int(v.size)
    N = L + 1
    diag = np.zeros(N, dtype=float)
    if L:
        diag[0] += 0.5
        diag[-1] += 0.5
        if N > 2:
            diag[1:-1] += 1.0
    lower = -0.5 * v
    upper = -0.5 / v
    if sparse:
        return diags([lower, diag, upper], offsets=[-1, 0, 1], shape=(N, N), format="csr", dtype=float)
    mat = np.diag(diag).astype(float)
    idx = np.arange(L)
    mat[idx + 1, idx] = lower
    mat[idx, idx + 1] = upper
    return mat


def build_h_clock(L: int, sparse: bool = True):
    if L < 0:
        raise ValueError("L must be nonnegative")
    N = L + 1
    diag = np.zeros(N, dtype=float)
    if L:
        diag[0] = 0.5
        diag[-1] = 0.5
        if N > 2:
            diag[1:-1] = 1.0
    off = np.full(L, -0.5, dtype=float)
    if sparse:
        return diags([off, diag, off], [-1, 0, 1], shape=(N, N), format="csr")
    mat = np.diag(diag)
    idx = np.arange(L)
    mat[idx + 1, idx] = off
    mat[idx, idx + 1] = off
    return mat


def clock_eigenvalues(L: int) -> np.ndarray:
    if L < 0:
        raise ValueError("L must be nonnegative")
    k = np.arange(L + 1, dtype=float)
    return 1.0 - np.cos(k * np.pi / (L + 1))


def clock_eigenvectors_chunk(L: int, k_start: int, k_stop: int) -> np.ndarray:
    """Return analytic clock eigenvectors as rows for k_start <= k < k_stop."""
    N = L + 1
    if not (0 <= k_start <= k_stop <= N):
        raise ValueError("invalid k chunk bounds")
    ks = np.arange(k_start, k_stop, dtype=float)
    sites = np.arange(N, dtype=float)
    out = np.empty((len(ks), N), dtype=float)
    for row, k in enumerate(ks.astype(int)):
        if k == 0:
            out[row, :] = 1.0 / math.sqrt(N)
        else:
            out[row, :] = math.sqrt(2.0 / N) * np.cos(k * np.pi * (sites + 0.5) / N)
    return out


def hd_block(sigma: float, theta: float = np.pi / 4, omega: float = 1.0) -> np.ndarray:
    sigma = float(sigma)
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    st = math.sin(float(theta))
    ct = math.cos(float(theta))
    return float(omega) * np.array(
        [[st * st, -st * ct / sigma], [-st * ct * sigma, ct * ct]],
        dtype=float,
    )


def sigma_min_dense(H, z: complex) -> float:
    mat = H.toarray() if issparse(H) else np.asarray(H)
    A = complex(z) * np.eye(mat.shape[0], dtype=complex) - mat
    return float(np.linalg.svd(A, compute_uv=False)[-1])


def sigma_min_2x2(H: np.ndarray, z: complex) -> float:
    mat = np.asarray(H, dtype=complex)
    if mat.shape != (2, 2):
        raise ValueError("sigma_min_2x2 requires a 2 x 2 matrix")
    A = complex(z) * np.eye(2, dtype=complex) - mat
    T = float(np.real(np.sum(np.abs(A) ** 2)))
    D = float(abs(np.linalg.det(A)) ** 2)
    disc = max(T * T - 4.0 * D, 0.0)
    s_max_sq = 0.5 * (T + math.sqrt(disc))
    s_min_sq = D / s_max_sq if s_max_sq > 0.0 else 0.0
    return math.sqrt(max(float(s_min_sq), 0.0))


def sigma_min_sparse(H, z: complex, method: str = "eigsh", return_info: bool = False):
    """Smallest singular value of zI-H for sparse or dense H."""
    if not issparse(H):
        value = sigma_min_dense(H, z)
        info = {"method": "dense", "warning": "", "reliable": bool(value >= 1e-15)}
        return (value, info) if return_info else value
    N = H.shape[0]
    A = complex(z) * eye(N, format="csr", dtype=complex) - H.astype(complex)
    warning_text = ""
    try:
        if method == "svds":
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                vals = svds(A, k=1, which="SM", return_singular_vectors=False, tol=1e-10, maxiter=30000)
            if caught:
                warning_text = "; ".join(str(w.message) for w in caught)
            value = float(np.min(vals))
            info = {"method": "svds", "warning": warning_text, "reliable": bool(value >= 1e-15)}
            return (value, info) if return_info else value
    except Exception as exc:
        warning_text = f"svds failed: {exc!r}"
    try:
        B = A.conj().T @ A
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            vals = eigsh(B, k=1, which="SA", return_eigenvectors=False, tol=1e-10, maxiter=30000)
        if caught:
            warning_text = "; ".join(filter(None, [warning_text] + [str(w.message) for w in caught]))
        lam = max(float(np.real(vals[0])), 0.0)
        value = math.sqrt(lam)
        info = {"method": "eigsh", "warning": warning_text, "reliable": bool(value >= 1e-15)}
        return (value, info) if return_info else value
    except Exception as exc:
        if H.shape[0] <= 900:
            value = sigma_min_dense(H, z)
            warning_text = "; ".join(filter(None, [warning_text, f"eigsh failed: {exc!r}; dense fallback"]))
            info = {"method": "dense_fallback", "warning": warning_text, "reliable": bool(value >= 1e-15)}
            return (value, info) if return_info else value
        raise


def _fk_tridiagonal_parts_from_multipliers(v_list: Iterable[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v = np.asarray(list(v_list), dtype=float)
    L = len(v)
    N = L + 1
    hdiag = np.zeros(N, dtype=float)
    if L:
        hdiag[0] = 0.5
        hdiag[-1] = 0.5
        if N > 2:
            hdiag[1:-1] = 1.0
    h_lower = -0.5 * v
    h_upper = -0.5 / v
    return hdiag, h_lower, h_upper


def sigma_min_fk_banded_from_multipliers(v_list: Iterable[float], z: complex) -> float:
    """Smallest singular value for a FK sector using the banded A^* A matrix."""
    hdiag, h_lower, h_upper = _fk_tridiagonal_parts_from_multipliers(v_list)
    N = hdiag.size
    diag_a = complex(z) - hdiag.astype(complex)
    lower_a = -h_lower.astype(complex)
    upper_a = -h_upper.astype(complex)

    main = np.abs(diag_a) ** 2
    if N > 1:
        main[:-1] += np.abs(lower_a) ** 2
        main[1:] += np.abs(upper_a) ** 2
        first = np.conj(diag_a[:-1]) * upper_a + np.conj(lower_a) * diag_a[1:]
    else:
        first = np.empty(0, dtype=complex)
    if N > 2:
        second = np.conj(lower_a[:-1]) * upper_a[1:]
    else:
        second = np.empty(0, dtype=complex)

    ab = np.zeros((3, N), dtype=complex)
    ab[2, :] = main
    if N > 1:
        ab[1, 1:] = first
    if N > 2:
        ab[0, 2:] = second
    try:
        lam = eig_banded(
            ab,
            lower=False,
            eigvals_only=True,
            select="i",
            select_range=(0, 0),
            check_finite=False,
        )[0]
        return math.sqrt(max(float(np.real(lam)), 0.0))
    except Exception:
        H = build_fk_sector_matrix_from_multipliers(v_list, sparse=False)
        return sigma_min_dense(H, z)


def log_chi_for_k(logW: np.ndarray, k: int) -> float:
    vec = clock_eigenvectors_chunk(len(logW) - 1, k, k + 1)[0]
    u2 = vec * vec
    log_u2 = np.full_like(u2, -np.inf, dtype=float)
    positive = u2 > 0.0
    log_u2[positive] = np.log(u2[positive])
    log_plus_sq = logsumexp(2.0 * logW + log_u2)
    log_minus_sq = logsumexp(-2.0 * logW + log_u2)
    return 0.5 * log_plus_sq + 0.5 * log_minus_sq


def eigenvalue_condition_estimate_for_sector(
    logW: np.ndarray,
    chunk_size: int = 128,
    return_per_k: bool = False,
) -> dict:
    """Return the log-domain eigenvalue-condition estimate for one FK sector."""
    logW = np.asarray(logW, dtype=float)
    L = len(logW) - 1
    lambdas = clock_eigenvalues(L)
    log_chi_0 = log_chi_for_k(logW, 0)
    best_log_eps = math.inf
    best_k = 1
    best_log_chi = math.nan
    per_k_rows: list[dict] = []

    for k_start in range(1, L + 1, int(chunk_size)):
        k_stop = min(L + 1, k_start + int(chunk_size))
        vecs = clock_eigenvectors_chunk(L, k_start, k_stop)
        u2 = vecs * vecs
        log_u2 = np.full_like(u2, -np.inf, dtype=float)
        positive = u2 > 0.0
        log_u2[positive] = np.log(u2[positive])
        log_plus_sq = logsumexp(2.0 * logW[None, :] + log_u2, axis=1)
        log_minus_sq = logsumexp(-2.0 * logW[None, :] + log_u2, axis=1)
        log_chi = 0.5 * log_plus_sq + 0.5 * log_minus_sq
        ks = np.arange(k_start, k_stop)
        log_eps = np.log(lambdas[ks]) - np.logaddexp(log_chi_0, log_chi)
        idx = int(np.argmin(log_eps))
        if float(log_eps[idx]) < best_log_eps:
            best_log_eps = float(log_eps[idx])
            best_k = int(ks[idx])
            best_log_chi = float(log_chi[idx])
        if return_per_k:
            for kk, lc, le in zip(ks, log_chi, log_eps):
                per_k_rows.append(
                    {
                        "k": int(kk),
                        "lambda_k": float(lambdas[int(kk)]),
                        "log10_chi_k": float(lc / LN10),
                        "log10_epsilon_est_k": float(le / LN10),
                    }
                )

    result = {
        "log_epsilon_est": float(best_log_eps),
        "log10_epsilon_est": float(best_log_eps / LN10),
        "k_star_est": int(best_k),
        "log_chi_0": float(log_chi_0),
        "log10_chi_0": float(log_chi_0 / LN10),
        "log_chi_kstar": float(best_log_chi),
        "log10_chi_kstar": float(best_log_chi / LN10),
    }
    if return_per_k:
        result["per_k"] = per_k_rows
    return result


def candidate_excited_modes(estimate: dict, L: int, max_low_k: int = 20, within_decades: float = 3.0) -> list[int]:
    candidates = set(range(1, min(max_low_k, L) + 1))
    candidates.add(int(estimate["k_star_est"]))
    for row in estimate.get("per_k", []):
        if row["log10_epsilon_est_k"] <= estimate["log10_epsilon_est"] + within_decades:
            candidates.add(int(row["k"]))
    return sorted(k for k in candidates if 1 <= k <= L)


def _grid_arrays(window: GridWindow) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(window.x_min, window.x_max, int(window.nx))
    ys = np.linspace(window.y_min, window.y_max, int(window.ny))
    Z = xs[None, :] + 1j * ys[:, None]
    return xs, ys, Z


def nearest_grid_flat_index(xs: np.ndarray, ys: np.ndarray, z: complex) -> int:
    ix = int(np.argmin(np.abs(xs - float(np.real(z)))))
    iy = int(np.argmin(np.abs(ys - float(np.imag(z)))))
    return iy * len(xs) + ix


def compute_sigma_grid(
    sigma_min_solver: Callable[[complex], float],
    window: GridWindow,
    logger: logging.Logger | None = None,
) -> np.ndarray:
    xs, ys, Z = _grid_arrays(window)
    F = np.empty((len(ys), len(xs)), dtype=float)
    start = time.perf_counter()
    total = F.size
    last_log = start
    for flat, z in enumerate(Z.ravel()):
        F.ravel()[flat] = sigma_min_solver(complex(z))
        now = time.perf_counter()
        if logger is not None and now - last_log > 20.0:
            logger.info("sigma grid progress %d/%d in %.1fs", flat + 1, total, now - start)
            last_log = now
    return F


def pseudospectral_closing_threshold_grid(
    H,
    eigenvalues: Iterable[complex],
    ground_indices: Iterable[int],
    excited_indices: Iterable[int],
    grid_params: GridWindow,
    sigma_min_solver: Callable[[complex], float] | None = None,
    logger: logging.Logger | None = None,
) -> dict:
    """Union-find grid approximation to the ground-to-excited closing threshold."""
    eigenvalues = list(eigenvalues)
    solver = sigma_min_solver
    if solver is None:
        solver = lambda z: sigma_min_dense(H, z)
    xs, ys, _ = _grid_arrays(grid_params)
    F = compute_sigma_grid(solver, grid_params, logger=logger)
    for idx in list(ground_indices) + list(excited_indices):
        F.ravel()[nearest_grid_flat_index(xs, ys, eigenvalues[int(idx)])] = 0.0

    order = np.argsort(F.ravel())
    active = np.zeros(F.size, dtype=bool)
    uf = UnionFind(F.size)
    ground_seeds = {nearest_grid_flat_index(xs, ys, eigenvalues[int(idx)]) for idx in ground_indices}
    excited_seeds = {nearest_grid_flat_index(xs, ys, eigenvalues[int(idx)]) for idx in excited_indices}
    ny, nx = F.shape
    neighbor_offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    for flat in order:
        flat = int(flat)
        iy, ix = divmod(flat, nx)
        active[flat] = True
        uf.contains_ground[flat] = flat in ground_seeds
        uf.contains_excited[flat] = flat in excited_seeds
        uf.touches_boundary[flat] = ix == 0 or iy == 0 or ix == nx - 1 or iy == ny - 1
        root = flat
        for dy, dx in neighbor_offsets:
            jy = iy + dy
            jx = ix + dx
            if 0 <= jy < ny and 0 <= jx < nx:
                other = jy * nx + jx
                if active[other]:
                    root = uf.union(root, other)
        root = uf.find(root)
        if uf.contains_ground[root] and uf.contains_excited[root]:
            value = float(F.ravel()[flat])
            return {
                "epsilon_c": value,
                "log10_epsilon_c": float(math.log10(max(value, TINY_FLOOR))),
                "touches_boundary": bool(uf.touches_boundary[root]),
                "reliable": bool(value >= 1e-15 and not uf.touches_boundary[root]),
                "grid_Nx": int(nx),
                "grid_Ny": int(ny),
                "window": grid_params.__dict__,
            }

    return {
        "epsilon_c": math.nan,
        "log10_epsilon_c": math.nan,
        "touches_boundary": True,
        "reliable": False,
        "grid_Nx": int(nx),
        "grid_Ny": int(ny),
        "window": grid_params.__dict__,
    }


def radial_pseudospectrum_boundary(
    center: complex,
    epsilon: float,
    sigma_min_solver: Callable[[complex], float],
    radius_hint: float,
    n_angles: int = 181,
) -> np.ndarray:
    """Resolve a local epsilon-boundary by radial bisection from an eigenvalue."""
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=True)
    pts = np.empty((len(angles), 2), dtype=float)
    for idx, phi in enumerate(angles):
        direction = complex(math.cos(float(phi)), math.sin(float(phi)))
        lo = 0.0
        hi = max(float(radius_hint), 10.0 * epsilon, 1e-16)
        for _ in range(80):
            if sigma_min_solver(center + hi * direction) > epsilon:
                break
            hi *= 2.0
        for _ in range(70):
            mid = 0.5 * (lo + hi)
            if sigma_min_solver(center + mid * direction) <= epsilon:
                lo = mid
            else:
                hi = mid
        z = center + 0.5 * (lo + hi) * direction
        pts[idx] = [float(np.real(z)), float(np.imag(z))]
    return pts


def contour_components_for_epsilon(
    eigenvalues: tuple[complex, complex],
    epsilon: float,
    window: GridWindow,
    sigma_min_solver: Callable[[complex], float],
    logger: logging.Logger | None = None,
) -> dict:
    """Compute grid component metadata and contour vertices near E0 and E1."""
    xs, ys, _ = _grid_arrays(window)
    start = time.perf_counter()
    F = compute_sigma_grid(sigma_min_solver, window, logger=logger)
    seed0 = nearest_grid_flat_index(xs, ys, eigenvalues[0])
    seed1 = nearest_grid_flat_index(xs, ys, eigenvalues[1])
    F.ravel()[seed0] = 0.0
    F.ravel()[seed1] = 0.0
    mask = F <= float(epsilon)
    structure = np.ones((3, 3), dtype=int)
    labels, _ = cc_label(mask, structure=structure)
    label0 = int(labels.ravel()[seed0])
    label1 = int(labels.ravel()[seed1])
    connected = label0 != 0 and label0 == label1
    selected_labels = {label0, label1} - {0}
    selected = np.isin(labels, list(selected_labels)) if selected_labels else np.zeros_like(mask)
    touches = bool(
        np.any(selected[0, :])
        or np.any(selected[-1, :])
        or np.any(selected[:, 0])
        or np.any(selected[:, -1])
    )
    logF = np.log10(np.maximum(F, TINY_FLOOR))
    contour_vertices: list[np.ndarray] = []
    fig, ax = plt.subplots()
    try:
        cs = ax.contour(xs, ys, logF, levels=[math.log10(epsilon)])
        for seg in cs.allsegs[0]:
            if len(seg) >= 3:
                contour_vertices.append(np.asarray(seg, dtype=float))
    finally:
        plt.close(fig)

    if not contour_vertices:
        span = max(window.x_max - window.x_min, window.y_max - window.y_min)
        hint = max(0.05 * span, 100.0 * epsilon)
        contour_vertices = [
            radial_pseudospectrum_boundary(eigenvalues[0], epsilon, sigma_min_solver, hint),
            radial_pseudospectrum_boundary(eigenvalues[1], epsilon, sigma_min_solver, hint),
        ]

    return {
        "contours": contour_vertices,
        "whether_E0_E1_connected": bool(connected),
        "contour_touches_boundary": bool(touches),
        "reliable": bool(np.nanmin(F) <= epsilon),
        "runtime_seconds": float(time.perf_counter() - start),
    }


def selected_sector_for_n(n: int, selection_by_n: dict[int, tuple[int, ...]] | None = None) -> tuple[tuple[int, ...], str]:
    if selection_by_n and n in selection_by_n:
        return tuple(selection_by_n[n]), "validated_min_estimate_for_this_n"
    graph = build_ck_graph_from_n(n)
    return tuple(graph["x_mis"]), "mis_max_gain_sector"


def enumerate_sector_validation(
    n: int,
    p: float = 2.0,
    q: float = 4.0,
    chunk_size: int = 128,
    out_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    graph = build_ck_graph_from_n(n)
    gates = get_legacy_ck_gates(n, r=n)
    rows = []
    for idx in range(1 << n):
        x = bit_tuple(idx, n)
        logW = log_W_path_for_config(x, gates, p=p, q=q)
        est = eigenvalue_condition_estimate_for_sector(logW, chunk_size=chunk_size)
        rows.append(
            {
                "n": n,
                "m": graph["m"],
                "sector_index": idx,
                "bitstring": bitstring_label(x),
                "is_mis_sector": x == tuple(graph["x_mis"]),
                "is_independent_set": is_independent_set(x, graph["edges"]),
                "independent_set_size": int(sum(x)) if is_independent_set(x, graph["edges"]) else np.nan,
                "logW_L": float(logW[-1]),
                "log10_eps_est": float(est["log10_epsilon_est"]),
                "k_star_est": int(est["k_star_est"]),
                "log10_chi_0": float(est["log10_chi_0"]),
                "log10_chi_kstar": float(est["log10_chi_kstar"]),
            }
        )
    df = pd.DataFrame(rows)
    df["rank_log_gain_desc"] = df["logW_L"].rank(method="min", ascending=False).astype(int)
    df["rank_epsilon_est_asc"] = df["log10_eps_est"].rank(method="min", ascending=True).astype(int)
    if out_csv is not None:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    mis = df[df["is_mis_sector"]].iloc[0]
    selected = df.sort_values(["log10_eps_est", "rank_log_gain_desc"], ascending=[True, True]).iloc[0]
    summary = {
        "n": int(n),
        "rank_mis_by_log_gain": int(mis["rank_log_gain_desc"]),
        "rank_mis_by_epsilon_est": int(mis["rank_epsilon_est_asc"]),
        "min_log10_epsilon_est": float(selected["log10_eps_est"]),
        "mis_log10_epsilon_est": float(mis["log10_eps_est"]),
        "selected_sector_bitstring": str(selected["bitstring"]),
        "selected_sector_index": int(selected["sector_index"]),
        "selected_sector_logW_L": float(selected["logW_L"]),
        "selected_sector_rule": "validated_min_estimate_for_this_n",
        "mis_bitstring": bitstring_label(graph["x_mis"]),
    }
    return df, summary


def _fk_data_for_sector(n: int, x: tuple[int, ...], p: float, q: float) -> dict:
    graph = build_ck_graph_from_n(n)
    gates = get_legacy_ck_gates(n, r=n)
    v = multipliers_for_config(x, gates, p=p, q=q)
    logW = log_W_path_for_config(x, gates, p=p, q=q)
    L = len(gates)
    lambdas = clock_eigenvalues(L)
    return {"graph": graph, "gates": gates, "v": v, "logW": logW, "L": L, "lambdas": lambdas}


def compute_left_contours(
    n_left: list[int],
    epsilon_vis: float,
    grid_N: int,
    p: float,
    q: float,
    theta: float,
    omega: float,
    selection_by_n: dict[int, tuple[int, ...]],
    logger: logging.Logger,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    records: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    contour_index = 0

    for n in n_left:
        x, rule = selected_sector_for_n(n, selection_by_n)
        fk = _fk_data_for_sector(n, x, p, q)
        Delta = float(fk["lambdas"][1])
        window = GridWindow(-0.25 * Delta, 1.25 * Delta, -0.75 * Delta, 0.75 * Delta, grid_N, grid_N)
        solver = lambda z, v=fk["v"]: sigma_min_fk_banded_from_multipliers(v, z)
        result = contour_components_for_epsilon((0.0 + 0.0j, Delta + 0.0j), epsilon_vis, window, solver, logger)
        contours = result["contours"]
        contour_source = "grid_log_sigma_contour"
        if result["contour_touches_boundary"] or result["whether_E0_E1_connected"] or len(contours) > 12:
            contours = []
            contour_source = "exceeds_window_no_fake_contour"
        for contour in contours:
            key = f"contour_{contour_index}"
            arrays[key] = contour
            records.append(
                {
                    "contour_key": key,
                    "n": n,
                    "m": fk["graph"]["m"],
                    "L": fk["L"],
                    "algorithm": "NHMIS-FKQAA",
                    "E0": 0.0,
                    "E1": Delta,
                    "epsilon_vis": epsilon_vis,
                    "whether_E0_E1_connected": result["whether_E0_E1_connected"],
                    "contour_touches_boundary": result["contour_touches_boundary"],
                    "sigma_solver": "fk_banded_AstarA",
                    "reliable": result["reliable"],
                    "runtime_seconds": result["runtime_seconds"],
                    "sector_used": bitstring_label(x),
                    "sector_selection_rule": rule,
                    "contour_source": contour_source,
                    "contour_center": np.nan,
                }
            )
            contour_index += 1

    for algorithm, sigma in [("NHMIS-HDQAA", q), ("HMIS-HDQAA", 1.0)]:
        H = hd_block(sigma, theta=theta, omega=omega)
        solver = lambda z, HH=H: sigma_min_2x2(HH, z)
        window = GridWindow(-0.08, 1.08, -0.12, 0.12, grid_N, grid_N)
        for n in n_left:
            start = time.perf_counter()
            local_radius = 1e-12 if algorithm == "NHMIS-HDQAA" else max(100.0 * epsilon_vis, 1e-13)
            contours = [
                radial_pseudospectrum_boundary(0.0 + 0.0j, epsilon_vis, solver, local_radius),
                radial_pseudospectrum_boundary(omega + 0.0j, epsilon_vis, solver, local_radius),
            ]
            for center, contour in zip([0.0, float(omega)], contours):
                key = f"contour_{contour_index}"
                arrays[key] = contour
                records.append(
                    {
                        "contour_key": key,
                        "n": n,
                        "m": build_ck_graph_from_n(n)["m"],
                        "L": len(get_legacy_ck_gates(n, r=n)),
                        "algorithm": algorithm,
                        "E0": 0.0,
                        "E1": float(omega),
                        "epsilon_vis": epsilon_vis,
                        "whether_E0_E1_connected": False,
                        "contour_touches_boundary": False,
                        "sigma_solver": "analytic_2x2_radial",
                        "reliable": True,
                        "runtime_seconds": float(time.perf_counter() - start),
                        "sector_used": "local_hd_block",
                        "sector_selection_rule": "worst_local_sigma_q" if algorithm == "NHMIS-HDQAA" else "unitary_sigma_1",
                        "contour_source": "radial_resolvent_boundary",
                        "contour_center": center,
                    }
                )
                contour_index += 1

    return records, arrays


def compute_threshold_for_target_mode(
    v: np.ndarray,
    lambda_k: float,
    grid_N: int,
    logger: logging.Logger,
) -> dict:
    solver = lambda z: sigma_min_fk_banded_from_multipliers(v, z)
    window = GridWindow(-0.2 * lambda_k, 1.2 * lambda_k, -0.8 * lambda_k, 0.8 * lambda_k, grid_N, grid_N)
    eigenvalues = [0.0 + 0.0j, complex(lambda_k, 0.0)]
    return pseudospectral_closing_threshold_grid(
        None,
        eigenvalues,
        ground_indices=[0],
        excited_indices=[1],
        grid_params=window,
        sigma_min_solver=solver,
        logger=logger,
    )


def compute_right_thresholds(
    n_right: list[int],
    eps_floor: float,
    grid_N: int,
    p: float,
    q: float,
    selection_by_n: dict[int, tuple[int, ...]],
    exact_grid_n_max: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    conv_rows: list[dict] = []
    for n in n_right:
        start_total = time.perf_counter()
        x, rule = selected_sector_for_n(n, selection_by_n)
        fk = _fk_data_for_sector(n, x, p, q)
        est_start = time.perf_counter()
        est = eigenvalue_condition_estimate_for_sector(fk["logW"], chunk_size=128, return_per_k=(n <= exact_grid_n_max))
        est_runtime = time.perf_counter() - est_start
        L = fk["L"]
        Delta = float(fk["lambdas"][1])
        log10_kappa = float((np.max(fk["logW"]) - np.min(fk["logW"])) / LN10)
        grid_runtime = 0.0
        eps_c_grid = math.nan
        log10_eps_c_grid = math.nan
        reliable_grid = False
        is_upper_bound = True
        grid_status = "skipped_estimate_only"

        if n <= exact_grid_n_max:
            candidates = candidate_excited_modes(est, L, max_low_k=8 if grid_N <= 160 else 20)
            logger.info("n=%d grid candidates=%s", n, candidates[:20])
            best = None
            for k in candidates:
                t0 = time.perf_counter()
                result = compute_threshold_for_target_mode(fk["v"], float(fk["lambdas"][k]), grid_N, logger)
                elapsed = time.perf_counter() - t0
                grid_runtime += elapsed
                conv_rows.append(
                    {
                        "n": n,
                        "m": fk["graph"]["m"],
                        "L": L,
                        "k": k,
                        "grid_N": grid_N,
                        "log10_eps_c_grid": result["log10_epsilon_c"],
                        "eps_c_grid": result["epsilon_c"],
                        "touches_boundary": result["touches_boundary"],
                        "reliable": result["reliable"],
                        "runtime_seconds": elapsed,
                        "status": "quick_or_main_grid",
                    }
                )
                if best is None or result["epsilon_c"] < best["epsilon_c"]:
                    best = {"k": k, **result}
            if best is not None:
                eps_c_grid = float(best["epsilon_c"])
                log10_eps_c_grid = float(best["log10_epsilon_c"])
                reliable_grid = bool(best["reliable"] and eps_c_grid >= eps_floor)
                is_upper_bound = bool(eps_c_grid < eps_floor or not reliable_grid)
                grid_status = f"grid_best_k_{best['k']}"

        if not np.isfinite(eps_c_grid) or eps_c_grid < eps_floor:
            plot_log10 = math.log10(eps_floor)
            is_upper_bound = True
        else:
            plot_log10 = log10_eps_c_grid

        rows.append(
            {
                "n": n,
                "m": fk["graph"]["m"],
                "L": L,
                "Delta_FK": Delta,
                "eps_floor": eps_floor,
                "log10_eps_floor": math.log10(eps_floor),
                "eps_c_grid": eps_c_grid,
                "log10_eps_c_grid": log10_eps_c_grid,
                "plot_log10_eps_c": plot_log10,
                "is_upper_bound": is_upper_bound,
                "reliable_grid": reliable_grid,
                "grid_status": grid_status,
                "log10_eps_est": float(est["log10_epsilon_est"]),
                "k_star_est": int(est["k_star_est"]),
                "log10_chi_0": float(est["log10_chi_0"]),
                "log10_chi_kstar": float(est["log10_chi_kstar"]),
                "log10_kappa_S_selected": log10_kappa,
                "sector_used": bitstring_label(x),
                "sector_selection_rule": rule,
                "grid_runtime_seconds": float(grid_runtime),
                "estimate_runtime_seconds": float(est_runtime),
                "total_runtime_seconds": float(time.perf_counter() - start_total),
            }
        )
        logger.info(
            "n=%d m=%d L=%d Delta=%.3e sector=%s log10_est=%.3f k*=%d grid=%s runtime=%.1fs",
            n,
            fk["graph"]["m"],
            L,
            Delta,
            bitstring_label(x),
            est["log10_epsilon_est"],
            est["k_star_est"],
            grid_status,
            time.perf_counter() - start_total,
        )

    return pd.DataFrame(rows), pd.DataFrame(conv_rows)


def _plot_contours_on_axis(
    ax,
    records: list[dict],
    arrays: dict[str, np.ndarray],
    algorithm: str,
    colors: dict[int, tuple],
    marker_mode: str,
) -> None:
    subset = [r for r in records if r["algorithm"] == algorithm]
    for row in subset:
        arr = arrays[row["contour_key"]]
        n = int(row["n"])
        ax.plot(arr[:, 0], arr[:, 1], color=colors[n], linewidth=1.2, alpha=0.95)
    seen = set()
    for row in subset:
        n = int(row["n"])
        key = (algorithm, n)
        if key in seen:
            continue
        seen.add(key)
        if marker_mode == "scheme":
            markers = {
                "NHMIS-FKQAA": ("s", "^"),
                "NHMIS-HDQAA": ("o", "p"),
                "HMIS-HDQAA": ("D", "*"),
            }[algorithm]
        else:
            markers = ("s", "^")
        ax.scatter([float(row["E0"])], [0.0], marker=markers[0], s=28, color=colors[n], edgecolor="black", linewidth=0.4)
        ax.scatter([float(row["E1"])], [0.0], marker=markers[1], s=34, color=colors[n], edgecolor="black", linewidth=0.4)
        if bool(row["whether_E0_E1_connected"]):
            ax.text(float(row["E1"]) * 0.55, 0.0, "closed", color=colors[n], fontsize=8)


def plot_fig2_left(
    records: list[dict],
    arrays: dict[str, np.ndarray],
    figures_dir: str | Path,
    marker_mode: str = "level",
) -> tuple[Path, Path]:
    n_values = sorted({int(r["n"]) for r in records})
    cmap = plt.get_cmap("tab10")
    colors = {n: cmap(i % 10) for i, n in enumerate(n_values)}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), constrained_layout=True)
    titles = ["Panel A: NHMIS-FKQAA", "Panel B: NHMIS-HDQAA", "Panel C: HMIS-HDQAA"]
    for ax, algorithm, title in zip(axes, ALGORITHMS, titles):
        _plot_contours_on_axis(ax, records, arrays, algorithm, colors, marker_mode)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Re z")
        ax.set_ylabel("Im z")
        ax.axhline(0.0, color="0.85", linewidth=0.7, zorder=0)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.65)
        subset = [r for r in records if r["algorithm"] == algorithm]
        if subset:
            all_pts = np.vstack([arrays[r["contour_key"]] for r in subset])
            xmin, xmax = float(np.min(all_pts[:, 0])), float(np.max(all_pts[:, 0]))
            ymin, ymax = float(np.min(all_pts[:, 1])), float(np.max(all_pts[:, 1]))
            padx = max(0.05 * (xmax - xmin), 1e-16)
            pady = max(0.08 * (ymax - ymin), 1e-16)
            ax.set_xlim(xmin - padx, xmax + padx)
            ax.set_ylim(ymin - pady, ymax + pady)
        if algorithm == "NHMIS-FKQAA" and subset:
            inset = ax.inset_axes([0.58, 0.55, 0.37, 0.37])
            _plot_contours_on_axis(inset, records, arrays, algorithm, colors, marker_mode)
            max_e1 = max(float(r["E1"]) for r in subset)
            inset.set_xlim(-0.15 * max_e1, 1.15 * max_e1)
            inset.set_ylim(-0.55 * max_e1, 0.55 * max_e1)
            inset.set_xticks([])
            inset.set_yticks([])
            inset.set_title("origin", fontsize=7)
        if algorithm in {"NHMIS-HDQAA", "HMIS-HDQAA"} and subset:
            for box, center, title_label in [([0.08, 0.55, 0.34, 0.36], 0.0, "E0"), ([0.58, 0.55, 0.34, 0.36], 1.0, "E1")]:
                inset = ax.inset_axes(box)
                pts = []
                for row in subset:
                    if abs(float(row.get("contour_center", np.nan)) - center) > 1e-12:
                        continue
                    arr = arrays[row["contour_key"]]
                    pts.append(arr)
                    n = int(row["n"])
                    inset.plot(arr[:, 0], arr[:, 1], color=colors[n], linewidth=1.0)
                if pts:
                    joined = np.vstack(pts)
                    dx = max(float(np.max(np.abs(joined[:, 0] - center))), 1e-16)
                    dy = max(float(np.max(np.abs(joined[:, 1]))), 1e-16)
                    inset.set_xlim(center - 1.35 * dx, center + 1.35 * dx)
                    inset.set_ylim(-1.35 * dy, 1.35 * dy)
                inset.set_xticks([])
                inset.set_yticks([])
                inset.set_title(title_label, fontsize=7)
    handles = [
        plt.Line2D([0], [0], color=colors[n], lw=1.6, label=f"n={n}") for n in n_values
    ]
    axes[0].legend(handles=handles, loc="best", fontsize=8)
    png = Path(figures_dir) / f"fig2_left_pseudospectrum_absolute_{marker_mode}_markers.png"
    pdf = Path(figures_dir) / f"fig2_left_pseudospectrum_absolute_{marker_mode}_markers.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def plot_fig2_right(threshold_df: pd.DataFrame, figures_dir: str | Path) -> tuple[Path, Path]:
    df = threshold_df.sort_values("n")
    fig, ax = plt.subplots(figsize=(6.2, 4.1), constrained_layout=True)
    reliable = df[(df["reliable_grid"]) & (~df["is_upper_bound"])]
    upper = df[df["is_upper_bound"]]
    if not reliable.empty:
        ax.scatter(reliable["n"], reliable["log10_eps_c_grid"], s=38, color="#1f77b4", label="grid threshold", zorder=3)
    if not upper.empty:
        ax.scatter(upper["n"], upper["log10_eps_floor"], marker="v", s=52, color="#d62728", label="upper bound due to numerical floor", zorder=3)
        for _, row in upper.iterrows():
            ax.annotate(
                "",
                xy=(row["n"], row["log10_eps_floor"] - 0.6),
                xytext=(row["n"], row["log10_eps_floor"] + 0.15),
                arrowprops={"arrowstyle": "->", "color": "#d62728", "lw": 1.0},
            )
    ax.plot(df["n"], df["log10_eps_est"], linestyle="--", color="#2ca02c", linewidth=1.6, label="eigenvalue-condition estimate")
    ax.axhline(float(df["log10_eps_floor"].iloc[0]), color="0.55", linestyle=":", linewidth=1.1, label="eps_floor")
    ax.set_xlabel("n")
    ax.set_ylabel("log10 epsilon")
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(loc="best", fontsize=8)
    png = Path(figures_dir) / "fig2_right_fk_threshold.png"
    pdf = Path(figures_dir) / "fig2_right_fk_threshold.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def save_left_outputs(records: list[dict], arrays: dict[str, np.ndarray], data_dir: str | Path) -> tuple[Path, Path]:
    npz_path = Path(data_dir) / "fig2_left_contours.npz"
    csv_path = Path(data_dir) / "fig2_left_contours_metadata.csv"
    np.savez_compressed(npz_path, **arrays)
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return npz_path, csv_path


# ---------------------------------------------------------------------------
# History-accumulation high-precision experiment-two implementation.
# ---------------------------------------------------------------------------


def default_history_hp_workers(reserve_cores: int = 2, max_workers: int | None = None) -> int:
    if max_workers is not None:
        return max(1, int(max_workers))
    cpu = os.cpu_count() or 1
    return max(1, int(cpu) - int(reserve_cores))


def ensure_mpmath_available() -> None:
    if mp is None:
        raise RuntimeError("mpmath is required for high-precision history experiment two")


def configure_history_hp_logger(out_dir: str | Path, append: bool = False) -> logging.Logger:
    dirs = ensure_dirs(out_dir)
    logger = logging.getLogger("experiment_two_history_hp")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(
        dirs["logs"] / "experiment2_history_hp.log",
        mode="a" if append else "w",
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    return logger


def enumerate_bitstring_sectors(n: int = 5) -> list[tuple[int, ...]]:
    return [bit_tuple(idx, n) for idx in range(1 << n)]


def history_sector_data_for_r(
    r: int,
    n: int = 5,
    p: float = 2.0,
    q: float = 4.0,
) -> list[dict]:
    gates = get_legacy_ck_gates(n, r=r)
    sectors = []
    for idx, x in enumerate(enumerate_bitstring_sectors(n)):
        sectors.append(
            {
                "sector_index": idx,
                "bitstring": bitstring_label(x),
                "x": tuple(x),
                "v": multipliers_for_config(x, gates, p=p, q=q),
                "logW": log_W_path_for_config(x, gates, p=p, q=q),
            }
        )
    return sectors


def tridiagonal_parts_for_fk_sector(v_list: Iterable[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return H_x tridiagonal parts lower, diag, upper."""
    hdiag, h_lower, h_upper = _fk_tridiagonal_parts_from_multipliers(v_list)
    return h_lower, hdiag, h_upper


def _mp_from_float(value: float):
    return mp.mpf(str(float(value)))


def _mpc_from_complex(z: complex):
    if mp is not None and isinstance(z, (mp.mpc, mp.mpf)):
        return mp.mpc(z)
    return mp.mpc(str(float(np.real(z))), str(float(np.imag(z))))


def _fk_A_tridiagonal_parts_mp(v_list: Iterable[float], z: complex, mp_dps: int):
    ensure_mpmath_available()
    mp.mp.dps = int(mp_dps)
    v_mp = [_mp_from_float(v) for v in v_list]
    L = len(v_mp)
    N = L + 1
    z_mp = _mpc_from_complex(z)
    diag_h = [mp.mpf("0") for _ in range(N)]
    if L:
        diag_h[0] = mp.mpf("0.5")
        diag_h[-1] = mp.mpf("0.5")
        for i in range(1, N - 1):
            diag_h[i] = mp.mpf("1.0")
    lower = [mp.mpf("0.5") * v for v in v_mp]
    upper = [mp.mpf("0.5") / v for v in v_mp]
    diag = [z_mp - d for d in diag_h]
    return lower, diag, upper


def thomas_solve_tridiagonal_mp(lower, diag, upper, rhs):
    """Solve a tridiagonal system with mpmath arithmetic."""
    n = len(diag)
    if len(rhs) != n or len(lower) != n - 1 or len(upper) != n - 1:
        raise ValueError("invalid tridiagonal dimensions")
    if n == 0:
        return []
    c = [mp.mpc(0) for _ in range(max(n - 1, 0))]
    d = [mp.mpc(0) for _ in range(n)]
    denom = mp.mpc(diag[0])
    if abs(denom) == 0:
        raise ZeroDivisionError("singular tridiagonal pivot")
    if n > 1:
        c[0] = upper[0] / denom
    d[0] = rhs[0] / denom
    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c[i - 1]
        if abs(denom) == 0:
            raise ZeroDivisionError("singular tridiagonal pivot")
        if i < n - 1:
            c[i] = upper[i] / denom
        d[i] = (rhs[i] - lower[i - 1] * d[i - 1]) / denom
    x = [mp.mpc(0) for _ in range(n)]
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]
    return x


def _norm2_mp(vec) -> mp.mpf:
    return mp.sqrt(mp.fsum([abs(v) ** 2 for v in vec]))


def _inner_mp(a, b):
    return mp.fsum([mp.conj(x) * y for x, y in zip(a, b)])


def log10_sigma_min_fk_sector_hp(
    v_list: Iterable[float],
    z: complex,
    mp_dps: int = 90,
    n_iter: int = 60,
    tol_log10: float = 1e-8,
) -> float:
    """High-precision log10 sigma_min(zI-H_x) using tridiagonal solves."""
    ensure_mpmath_available()
    mp.mp.dps = int(mp_dps)
    v_tuple = tuple(float(v) for v in v_list)
    L = len(v_tuple)
    N = L + 1
    try:
        lower, diag, upper = _fk_A_tridiagonal_parts_mp(v_tuple, z, mp_dps)
        lower_dag = [mp.conj(u) for u in upper]
        diag_dag = [mp.conj(d) for d in diag]
        upper_dag = [mp.conj(l) for l in lower]
        y = [mp.mpc(1) / mp.sqrt(N) for _ in range(N)]
        prev_log_sigma = None
        best_log_sigma = None
        for _ in range(int(n_iter)):
            u = thomas_solve_tridiagonal_mp(lower_dag, diag_dag, upper_dag, y)
            w = thomas_solve_tridiagonal_mp(lower, diag, upper, u)
            rayleigh = mp.re(_inner_mp(y, w) / _inner_mp(y, y))
            if rayleigh <= 0:
                return math.nan
            log_sigma = float(-mp.mpf("0.5") * mp.log10(rayleigh))
            best_log_sigma = log_sigma
            norm = _norm2_mp(w)
            if norm == 0:
                return math.inf
            y = [val / norm for val in w]
            if prev_log_sigma is not None and abs(log_sigma - prev_log_sigma) < float(tol_log10):
                break
            prev_log_sigma = log_sigma
        return float(best_log_sigma)
    except ZeroDivisionError:
        return -math.inf


def log10_sigma_min_fk_sector_mpmath_svd(
    v_list: Iterable[float],
    z: complex,
    mp_dps: int = 90,
) -> float:
    """Package reference using mpmath dense SVD."""
    ensure_mpmath_available()
    mp.mp.dps = int(mp_dps)
    v_tuple = tuple(float(v) for v in v_list)
    L = len(v_tuple)
    N = L + 1
    H = mp.matrix(N)
    for i in range(N):
        H[i, i] = mp.mpf("0")
    if L:
        H[0, 0] += mp.mpf("0.5")
        H[N - 1, N - 1] += mp.mpf("0.5")
        for i in range(1, N - 1):
            H[i, i] += mp.mpf("1.0")
    for l, v in enumerate(v_tuple, start=1):
        v_mp = _mp_from_float(v)
        H[l, l - 1] += -mp.mpf("0.5") * v_mp
        H[l - 1, l] += -mp.mpf("0.5") / v_mp
    z_mp = _mpc_from_complex(z)
    A = mp.eye(N) * z_mp - H
    svals = mp.svd(A, compute_uv=False)
    smin = min(svals)
    if smin == 0:
        return -math.inf
    return float(mp.log10(smin))


def log10_sigma_min_fk_global_hp(
    sector_data_list: list[dict],
    z: complex,
    mp_dps: int = 90,
    n_iter: int = 60,
    tol_log10: float = 1e-8,
) -> float:
    vals = [
        log10_sigma_min_fk_sector_hp(sector["v"], z, mp_dps=mp_dps, n_iter=n_iter, tol_log10=tol_log10)
        for sector in sector_data_list
    ]
    return float(np.nanmin(vals))


def log10_sigma_min_fk_global_hp_hybrid(
    sector_data_list: list[dict],
    z: complex,
    mp_dps: int = 90,
    n_iter: int = 60,
    tol_log10: float = 1e-8,
    double_safe_log10: float = -8.0,
) -> float:
    """Use double precision only to reject points far above the HP target region."""
    double_vals = []
    for sector in sector_data_list:
        try:
            value = sigma_min_fk_banded_from_multipliers(sector["v"], z)
            double_vals.append(math.log10(max(value, TINY_FLOOR)))
        except Exception:
            double_vals.append(-math.inf)
    double_min = float(np.nanmin(double_vals))
    if double_min > float(double_safe_log10):
        return double_min
    return log10_sigma_min_fk_global_hp(
        sector_data_list, z, mp_dps=mp_dps, n_iter=n_iter, tol_log10=tol_log10
    )


def mpmath_benchmark_required() -> None:
    ensure_mpmath_available()
    if not hasattr(mp, "svd"):
        raise RuntimeError("mpmath.svd is required for high-precision benchmark")


def run_hp_solver_benchmark(
    out_dir: str | Path,
    mp_dps: int = 90,
    p: float = 2.0,
    q: float = 4.0,
    force: bool = False,
) -> pd.DataFrame:
    mpmath_benchmark_required()
    dirs = ensure_dirs(out_dir)
    path = dirs["data"] / "hp_solver_benchmark.csv"
    if path.exists() and not force:
        df = pd.read_csv(path)
        if not df.empty and (df["status"] == "PASS").all():
            return df

    rows: list[dict] = []

    def add_case(test_name: str, v: np.ndarray, z: complex, sector: str) -> None:
        fast = log10_sigma_min_fk_sector_hp(v, z, mp_dps=mp_dps, n_iter=80, tol_log10=1e-10)
        ref = log10_sigma_min_fk_sector_mpmath_svd(v, z, mp_dps=mp_dps)
        if np.isneginf(fast) and np.isneginf(ref):
            err = 0.0
            status = "PASS"
        elif np.isfinite(fast) and np.isfinite(ref):
            err = abs(float(fast) - float(ref))
            status = "PASS" if err < 1e-6 else "FAIL"
        else:
            err = math.inf
            status = "PASS" if (fast < -50 and ref < -50) else "FAIL"
        if np.isfinite(ref) and ref > -12:
            H = build_fk_sector_matrix_from_multipliers(v, sparse=False)
            dense = sigma_min_dense(H, z)
            dense_log = math.log10(max(dense, TINY_FLOOR))
            dense_err = abs(dense_log - ref)
            if dense_err > 1e-8:
                status = "FAIL"
        rows.append(
            {
                "test_name": test_name,
                "L": len(v),
                "sector": sector,
                "z_real": float(np.real(z)),
                "z_imag": float(np.imag(z)),
                "log10_fast": fast,
                "log10_mpmath_svd": ref,
                "abs_log10_error": err,
                "status": status,
            }
        )

    for L, v in [(4, np.array([2.0, 1.0, 4.0, 2.0])), (6, np.array([1.4, 2.2, 1.1, 3.0, 2.5, 1.7]))]:
        for z in [0.23 + 0.17j, 0.61 - 0.09j, 1.2 + 0.4j]:
            add_case("small_random_like_matrix", v, z, "synthetic")

    n = 5
    r = 1
    gates = get_legacy_ck_gates(n, r=r)
    graph = build_ck_graph_from_n(n)
    sector_bits = [tuple(graph["x_mis"]), (0, 0, 0, 0, 0), (1, 0, 1, 0, 0)]
    Delta = float(clock_eigenvalues(len(gates))[1])
    z_points = [
        0.5 * Delta,
        0.25 * Delta + 0.1j * Delta,
        0.75 * Delta + 0.1j * Delta,
        0.0 + 1e-8j * Delta,
        Delta + 1e-8j * Delta,
    ]
    for x in sector_bits:
        v = multipliers_for_config(x, gates, p=p, q=q)
        for z in z_points:
            add_case("n5_r1_sector_reference", v, z, bitstring_label(x))

    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    if not (df["status"] == "PASS").all():
        raise RuntimeError(f"high-precision solver benchmark failed; see {path}")
    return df


def global_condition_estimate_for_r(
    r: int,
    n: int = 5,
    p: float = 2.0,
    q: float = 4.0,
    chunk_size: int = 128,
) -> tuple[dict, pd.DataFrame]:
    sectors = history_sector_data_for_r(r, n=n, p=p, q=q)
    rows: list[dict] = []
    L = len(sectors[0]["v"])
    for sector in sectors:
        est = eigenvalue_condition_estimate_for_sector(sector["logW"], chunk_size=chunk_size, return_per_k=True)
        log10_kappa = float((np.max(sector["logW"]) - np.min(sector["logW"])) / LN10)
        for row in est["per_k"]:
            rows.append(
                {
                    "r": r,
                    "n": n,
                    "m": 2,
                    "L": L,
                    "sector_index": sector["sector_index"],
                    "sector_bitstring": sector["bitstring"],
                    "k": int(row["k"]),
                    "lambda_k": float(row["lambda_k"]),
                    "log10_chi_0": float(est["log10_chi_0"]),
                    "log10_chi_k": float(row["log10_chi_k"]),
                    "log10_epsilon_est": float(row["log10_epsilon_est_k"]),
                    "log10_kappa_S_sector": log10_kappa,
                }
            )
    df = pd.DataFrame(rows)
    best = df.sort_values("log10_epsilon_est").iloc[0]
    summary = {
        "r": int(r),
        "n": int(n),
        "m": 2,
        "L": int(L),
        "Delta_FK": float(clock_eigenvalues(L)[1]),
        "log10_epsilon_est_global": float(best["log10_epsilon_est"]),
        "k_star_est": int(best["k"]),
        "sector_est_min": int(best["sector_index"]),
        "sector_est_min_bitstring": str(best["sector_bitstring"]),
        "log10_chi_0": float(best["log10_chi_0"]),
        "log10_chi_kstar": float(best["log10_chi_k"]),
        "log10_kappa_S_est_sector": float(best["log10_kappa_S_sector"]),
    }
    return summary, df


def candidate_modes_from_global_estimates(est_df: pd.DataFrame, best_summary: dict, within_decades: float = 2.0) -> list[int]:
    L = int(best_summary["L"])
    candidates = {1}
    if L >= 2:
        candidates.add(2)
    candidates.add(int(best_summary["k_star_est"]))
    cutoff = float(best_summary["log10_epsilon_est_global"]) + float(within_decades)
    for k in sorted(est_df.loc[est_df["log10_epsilon_est"] <= cutoff, "k"].unique()):
        if 1 <= int(k) <= L:
            candidates.add(int(k))
    return sorted(candidates)


def _history_cache_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:20]


def _log_grid_worker(args: tuple) -> tuple[int, np.ndarray]:
    (
        row_start,
        ys_chunk,
        xs,
        sector_v_lists,
        mp_dps,
        n_iter,
        tol_log10,
    ) = args
    sector_data = [{"v": np.asarray(v, dtype=float)} for v in sector_v_lists]
    out = np.empty((len(ys_chunk), len(xs)), dtype=float)
    for iy, y in enumerate(ys_chunk):
        for ix, x in enumerate(xs):
            z = complex(float(x), float(y))
            out[iy, ix] = log10_sigma_min_fk_global_hp_hybrid(
                sector_data, z, mp_dps=mp_dps, n_iter=n_iter, tol_log10=tol_log10
            )
    return int(row_start), out


def compute_log_grid_global_hp(
    sector_data_list: list[dict],
    window: GridWindow,
    mp_dps: int,
    cache_dir: str | Path,
    cache_payload: dict,
    force: bool = False,
    max_workers: int = 1,
    n_iter: int = 60,
    tol_log10: float = 1e-8,
    hp_target_log10: float | None = None,
    p: float = 2.0,
    q: float = 4.0,
    n: int = 5,
    m: int = 2,
    calculation_kind: str = "grid",
    logger: logging.Logger | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    xs, ys, _ = _grid_arrays(window)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(cache_payload)
    payload.update(
        {
            "window": window.__dict__,
            "mp_dps": int(mp_dps),
            "n_iter": int(n_iter),
            "tol_log10": float(tol_log10),
            "hp_target_log10": None if hp_target_log10 is None else float(hp_target_log10),
            "p": float(p),
            "q": float(q),
            "gate_order": GATE_ORDER,
            "n": int(n),
            "m": int(m),
            "calculation_kind": calculation_kind,
        }
    )
    cache_path = cache_dir / f"loggrid_{_history_cache_key(payload)}.npz"
    if cache_path.exists() and not force:
        loaded = np.load(cache_path, allow_pickle=True)
        logF_loaded = np.nan_to_num(loaded["logF"], nan=math.inf, posinf=math.inf, neginf=-math.inf)
        return loaded["xs"], loaded["ys"], logF_loaded, {"cache_path": str(cache_path), "from_cache": True}

    start = time.perf_counter()
    workers = max(1, int(max_workers))
    row_chunk = max(1, int(math.ceil(len(ys) / (workers * 4))))
    chunks = [(start_idx, ys[start_idx : start_idx + row_chunk]) for start_idx in range(0, len(ys), row_chunk)]
    logF = np.empty((len(ys), len(xs)), dtype=float)
    sector_v_lists = [np.asarray(sector["v"], dtype=float) for sector in sector_data_list]
    if workers == 1:
        for idx, chunk in chunks:
            _, rows = _log_grid_worker((idx, chunk, xs, sector_v_lists, mp_dps, n_iter, tol_log10))
            logF[idx : idx + len(chunk), :] = rows
            if logger is not None:
                logger.info("grid rows %d/%d done", min(idx + len(chunk), len(ys)), len(ys))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_log_grid_worker, (idx, chunk, xs, sector_v_lists, mp_dps, n_iter, tol_log10))
                for idx, chunk in chunks
            ]
            done = 0
            for future in as_completed(futures):
                idx, rows = future.result()
                logF[idx : idx + rows.shape[0], :] = rows
                done += rows.shape[0]
                if logger is not None:
                    logger.info("grid rows %d/%d done", done, len(ys))
    logF = np.nan_to_num(logF, nan=math.inf, posinf=math.inf, neginf=-math.inf)
    np.savez_compressed(cache_path, xs=xs, ys=ys, logF=logF)
    return xs, ys, logF, {
        "cache_path": str(cache_path),
        "from_cache": False,
        "runtime_seconds": float(time.perf_counter() - start),
    }


def nearest_grid_flat_index_from_arrays(xs: np.ndarray, ys: np.ndarray, z: complex) -> int:
    return nearest_grid_flat_index(xs, ys, z)


def pseudospectral_closing_threshold_grid_log(
    logF: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    eigenvalues: Iterable[complex],
    ground_indices: Iterable[int],
    excited_indices: Iterable[int],
) -> dict:
    eigenvalues = list(eigenvalues)
    work = np.array(logF, dtype=float, copy=True)
    ground_seeds = {nearest_grid_flat_index_from_arrays(xs, ys, eigenvalues[int(idx)]) for idx in ground_indices}
    excited_seeds = {nearest_grid_flat_index_from_arrays(xs, ys, eigenvalues[int(idx)]) for idx in excited_indices}
    for seed in ground_seeds | excited_seeds:
        work.ravel()[seed] = -math.inf
    order = np.argsort(work.ravel())
    active = np.zeros(work.size, dtype=bool)
    uf = UnionFind(work.size)
    ny, nx = work.shape
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for flat in order:
        flat = int(flat)
        iy, ix = divmod(flat, nx)
        active[flat] = True
        uf.contains_ground[flat] = flat in ground_seeds
        uf.contains_excited[flat] = flat in excited_seeds
        uf.touches_boundary[flat] = ix == 0 or iy == 0 or ix == nx - 1 or iy == ny - 1
        root = flat
        for dy, dx in offsets:
            jy = iy + dy
            jx = ix + dx
            if 0 <= jy < ny and 0 <= jx < nx:
                other = jy * nx + jx
                if active[other]:
                    root = uf.union(root, other)
        root = uf.find(root)
        if uf.contains_ground[root] and uf.contains_excited[root]:
            return {
                "log10_epsilon_c": float(work.ravel()[flat]),
                "touches_boundary": bool(uf.touches_boundary[root]),
                "reliable": bool(np.isfinite(work.ravel()[flat])),
            }
    return {"log10_epsilon_c": math.nan, "touches_boundary": True, "reliable": False}


def component_contours_from_log_grid(
    logF: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    eigenvalues: tuple[complex, complex],
    log10_epsilon: float,
) -> dict:
    work = np.array(logF, dtype=float, copy=True)
    seed_value = float(log10_epsilon) - 20.0
    seed0 = nearest_grid_flat_index_from_arrays(xs, ys, eigenvalues[0])
    seed1 = nearest_grid_flat_index_from_arrays(xs, ys, eigenvalues[1])
    physical_mask = work <= float(log10_epsilon)
    seed0_was_physical = bool(physical_mask.ravel()[seed0])
    seed1_was_physical = bool(physical_mask.ravel()[seed1])
    work.ravel()[seed0] = seed_value
    work.ravel()[seed1] = seed_value
    mask = work <= float(log10_epsilon)
    labels, _ = cc_label(mask, structure=np.ones((3, 3), dtype=int))
    label0 = int(labels.ravel()[seed0])
    label1 = int(labels.ravel()[seed1])
    connected = label0 != 0 and label0 == label1
    selected_labels = {label0, label1} - {0}
    selected = np.isin(labels, list(selected_labels)) if selected_labels else np.zeros_like(mask)
    touches = bool(
        np.any(selected[0, :])
        or np.any(selected[-1, :])
        or np.any(selected[:, 0])
        or np.any(selected[:, -1])
    )
    contours: list[np.ndarray] = []
    component_physical_points = int(np.count_nonzero(selected & physical_mask))
    underresolved_seed_component = bool(selected_labels and component_physical_points < 4)
    if not touches and selected_labels and not underresolved_seed_component:
        finite_work = np.nan_to_num(work, nan=math.inf, posinf=math.inf, neginf=seed_value)
        masked = np.where(selected, finite_work, float(log10_epsilon) + 1.0)
        fig, ax = plt.subplots()
        try:
            cs = ax.contour(xs, ys, masked, levels=[float(log10_epsilon)])
            for seg in cs.allsegs[0]:
                if len(seg) >= 3:
                    contours.append(np.asarray(seg, dtype=float))
        finally:
            plt.close(fig)
    status = "ok"
    if touches:
        status = "exceeds_window"
    elif underresolved_seed_component:
        status = "seed_only_no_physical_component"
    elif not contours:
        status = "no_grid_contour"
    return {
        "contours": contours,
        "connected": bool(connected),
        "touches_boundary": bool(touches),
        "contour_status": status,
        "physical_points_total": int(np.count_nonzero(physical_mask)),
        "component_physical_points": component_physical_points,
        "seed0_was_physical": seed0_was_physical,
        "seed1_was_physical": seed1_was_physical,
    }


def single_component_contours_from_log_grid(
    logF: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    center: complex,
    log10_epsilon: float,
    force_seed: bool = True,
) -> dict:
    """Return contours for the component containing one eigenvalue seed."""
    work = np.array(logF, dtype=float, copy=True)
    seed_value = float(log10_epsilon) - 20.0
    seed = nearest_grid_flat_index_from_arrays(xs, ys, center)
    physical_mask = work <= float(log10_epsilon)
    seed_was_physical = bool(physical_mask.ravel()[seed])
    if force_seed:
        work.ravel()[seed] = seed_value
    mask = work <= float(log10_epsilon)
    labels, _ = cc_label(mask, structure=np.ones((3, 3), dtype=int))
    label_seed = int(labels.ravel()[seed])
    if label_seed == 0:
        return {
            "contours": [],
            "touches_boundary": False,
            "contour_status": "seed_not_in_component",
            "physical_points_total": int(np.count_nonzero(physical_mask)),
            "component_physical_points": 0,
            "seed_was_physical": seed_was_physical,
        }
    selected = labels == label_seed
    component_physical_points = int(np.count_nonzero(selected & physical_mask))
    if component_physical_points < 4:
        return {
            "contours": [],
            "touches_boundary": False,
            "contour_status": "seed_only_no_physical_component",
            "physical_points_total": int(np.count_nonzero(physical_mask)),
            "component_physical_points": component_physical_points,
            "seed_was_physical": seed_was_physical,
        }
    touches = bool(
        np.any(selected[0, :])
        or np.any(selected[-1, :])
        or np.any(selected[:, 0])
        or np.any(selected[:, -1])
    )
    contours: list[np.ndarray] = []
    if not touches:
        finite_work = np.nan_to_num(work, nan=math.inf, posinf=math.inf, neginf=seed_value)
        masked = np.where(selected, finite_work, float(log10_epsilon) + 1.0)
        fig, ax = plt.subplots()
        try:
            cs = ax.contour(xs, ys, masked, levels=[float(log10_epsilon)])
            for seg in cs.allsegs[0]:
                if len(seg) >= 3:
                    contours.append(np.asarray(seg, dtype=float))
        finally:
            plt.close(fig)
    status = "ok"
    if touches:
        status = "exceeds_window"
    elif not contours:
        status = "no_grid_contour"
    return {
        "contours": contours,
        "touches_boundary": bool(touches),
        "contour_status": status,
        "physical_points_total": int(np.count_nonzero(physical_mask)),
        "component_physical_points": component_physical_points,
        "seed_was_physical": seed_was_physical,
    }


def _radial_log_boundary_angle_worker(args: tuple) -> tuple[int, tuple[float, float]]:
    (
        idx,
        phi,
        sector_v_lists,
        center,
        log10_epsilon,
        radius_hint,
        mp_dps,
        n_iter,
    ) = args
    sector_data = [{"v": np.asarray(v, dtype=float)} for v in sector_v_lists]
    direction = complex(math.cos(float(phi)), math.sin(float(phi)))
    lo = 0.0
    hi = max(float(radius_hint), 1e-18)
    for _ in range(80):
        val = log10_sigma_min_fk_global_hp(sector_data, center + hi * direction, mp_dps=mp_dps, n_iter=n_iter)
        if val > log10_epsilon:
            break
        hi *= 2.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        val = log10_sigma_min_fk_global_hp(sector_data, center + mid * direction, mp_dps=mp_dps, n_iter=n_iter)
        if val <= log10_epsilon:
            lo = mid
        else:
            hi = mid
    z = center + 0.5 * (lo + hi) * direction
    return int(idx), (float(np.real(z)), float(np.imag(z)))


def radial_log_boundary_global_hp(
    sector_data_list: list[dict],
    center: complex,
    log10_epsilon: float,
    radius_hint: float,
    mp_dps: int,
    n_angles: int = 65,
    n_iter: int = 60,
    max_workers: int = 1,
) -> np.ndarray:
    pts = np.empty((int(n_angles), 2), dtype=float)
    phis = list(np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=True))
    sector_v_lists = [np.asarray(sector["v"], dtype=float) for sector in sector_data_list]
    tasks = [
        (idx, phi, sector_v_lists, center, float(log10_epsilon), float(radius_hint), int(mp_dps), int(n_iter))
        for idx, phi in enumerate(phis)
    ]
    workers = max(1, min(int(max_workers), len(tasks)))
    if workers == 1:
        for task in tasks:
            idx, point = _radial_log_boundary_angle_worker(task)
            pts[idx] = point
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for idx, point in executor.map(_radial_log_boundary_angle_worker, tasks):
                pts[idx] = point
    return pts


def local_condition_scales_for_r(sector_data_list: list[dict], L: int) -> dict[int, float]:
    out: dict[int, float] = {}
    for k in [0, 1]:
        log_chis = [log_chi_for_k(sector["logW"], k) for sector in sector_data_list]
        out[k] = 10.0 ** (max(log_chis) / LN10)
    return out


def _hd_radial_contour(center: complex, epsilon: float, H: np.ndarray, n_angles: int = 161) -> np.ndarray:
    solver = lambda z: sigma_min_2x2(H, z)
    radius = max(1e-40, 100.0 * epsilon)
    return radial_pseudospectrum_boundary(center, epsilon, solver, radius, n_angles=n_angles)


def _append_fk_contour_records(
    records: list[dict],
    arrays: dict[str, np.ndarray],
    contours: list[np.ndarray],
    contour_idx: int,
    *,
    r: int,
    L: int,
    E1: float,
    epsilon_vis: float,
    logeps: float,
    center_type: str,
    center_value: float,
    contour_status: str,
    contour_source: str,
    contour_touches_boundary: bool,
    connected: bool,
    grid_N: int,
    local_window_radius: float,
    window_expansions: int,
    mp_dps: int,
    hp_n_iter: int,
    from_cache: bool,
    runtime_seconds: float,
    final_failed: bool,
) -> int:
    if contours:
        for contour in contours:
            key = f"contour_{contour_idx}"
            arrays[key] = contour
            records.append(
                {
                    "contour_key": key,
                    "algorithm": "NHMIS-FKQAA-full-direct-sum",
                    "n": 5,
                    "m": 2,
                    "r": r,
                    "L": L,
                    "E0": 0.0,
                    "E1": E1,
                    "center_type": center_type,
                    "center_value": center_value,
                    "epsilon_vis": epsilon_vis,
                    "log10_epsilon_vis": logeps,
                    "E0_E1_connected_at_epsilon_vis": bool(connected),
                    "contour_touches_boundary": bool(contour_touches_boundary),
                    "contour_status": contour_status,
                    "contour_source": contour_source,
                    "grid_N": int(grid_N),
                    "local_window_radius": float(local_window_radius),
                    "window_expansions": int(window_expansions),
                    "mp_dps": int(mp_dps),
                    "hp_n_iter": int(hp_n_iter),
                    "from_cache": bool(from_cache),
                    "runtime_seconds": float(runtime_seconds),
                    "final_failed": bool(final_failed),
                }
            )
            contour_idx += 1
        return contour_idx

    records.append(
        {
            "contour_key": "",
            "algorithm": "NHMIS-FKQAA-full-direct-sum",
            "n": 5,
            "m": 2,
            "r": r,
            "L": L,
            "E0": 0.0,
            "E1": E1,
            "center_type": center_type,
            "center_value": center_value,
            "epsilon_vis": epsilon_vis,
            "log10_epsilon_vis": logeps,
            "E0_E1_connected_at_epsilon_vis": bool(connected),
            "contour_touches_boundary": bool(contour_touches_boundary),
            "contour_status": contour_status,
            "contour_source": contour_source,
            "grid_N": int(grid_N),
            "local_window_radius": float(local_window_radius),
            "window_expansions": int(window_expansions),
            "mp_dps": int(mp_dps),
            "hp_n_iter": int(hp_n_iter),
            "from_cache": bool(from_cache),
            "runtime_seconds": float(runtime_seconds),
            "final_failed": bool(final_failed),
        }
    )
    return contour_idx


def compute_local_fk_component_contour_hp(
    sectors: list[dict],
    r: int,
    L: int,
    center_type: str,
    center: complex,
    epsilon_vis: float,
    grid_N: int,
    mp_dps: int,
    hp_n_iter: int,
    out_dir: str | Path,
    p: float,
    q: float,
    force: bool,
    max_workers: int,
    hp_target_log10: float,
    logger: logging.Logger,
) -> dict:
    dirs = ensure_dirs(out_dir)
    logeps = math.log10(float(epsilon_vis))
    k = 0 if center_type == "E0" else 1
    log_chi = max(log_chi_for_k(sector["logW"], k) for sector in sectors)
    log_radius = logeps + log_chi / LN10 + math.log10(50.0)
    radius = max(10.0 ** max(log_radius, -300.0), 1e-40)
    total_runtime = 0.0
    last_info = {"from_cache": False}
    status = "failed"
    touches = False
    contours: list[np.ndarray] = []
    expansions = 0
    used_radius = radius
    for expansions in range(7):
        used_radius = radius
        window = GridWindow(
            float(np.real(center)) - radius,
            float(np.real(center)) + radius,
            float(np.imag(center)) - radius,
            float(np.imag(center)) + radius,
            grid_N,
            grid_N,
        )
        t0 = time.perf_counter()
        xs, ys, logF, info = compute_log_grid_global_hp(
            sectors,
            window,
            mp_dps=mp_dps,
            cache_dir=dirs["cache"],
            cache_payload={
                "kind": "left_local",
                "r": r,
                "center_type": center_type,
                "center_real": float(np.real(center)),
                "center_imag": float(np.imag(center)),
                "grid_N": grid_N,
                "hp_target_log10": float(hp_target_log10),
                "p": float(p),
                "q": float(q),
            },
            force=force,
            max_workers=max_workers,
            n_iter=hp_n_iter,
            hp_target_log10=hp_target_log10,
            p=p,
            q=q,
            calculation_kind="left_local",
            logger=logger,
        )
        total_runtime += time.perf_counter() - t0
        last_info = info
        component = single_component_contours_from_log_grid(logF, xs, ys, center, logeps)
        contours = component["contours"]
        status = component["contour_status"]
        touches = bool(component["touches_boundary"])
        if contours and not touches:
            break
        radius *= 10.0

    source = "full_direct_sum_global_hp_local_grid"
    final_failed = status != "ok" or not contours
    return {
        "contours": contours,
        "contour_status": status,
        "contour_source": source,
        "contour_touches_boundary": touches,
        "window_expansions": int(expansions),
        "local_window_radius": float(used_radius),
        "from_cache": bool(last_info.get("from_cache", False)),
        "runtime_seconds": float(total_runtime),
        "final_failed": bool(final_failed),
    }


def compute_connected_fk_component_contour_hp(
    sectors: list[dict],
    r: int,
    L: int,
    Delta: float,
    epsilon_vis: float,
    grid_N: int,
    mp_dps: int,
    hp_n_iter: int,
    out_dir: str | Path,
    p: float,
    q: float,
    force: bool,
    max_workers: int,
    hp_target_log10: float,
    logger: logging.Logger,
) -> dict:
    dirs = ensure_dirs(out_dir)
    logeps = math.log10(float(epsilon_vis))
    window = GridWindow(-0.3 * Delta, 1.3 * Delta, -1.0 * Delta, 1.0 * Delta, grid_N, grid_N)
    total_runtime = 0.0
    contours: list[np.ndarray] = []
    status = "failed"
    touches = False
    connected = False
    last_info = {"from_cache": False}
    expansions = 0
    used_window = window
    for expansions in range(7):
        used_window = window
        t0 = time.perf_counter()
        xs, ys, logF, info = compute_log_grid_global_hp(
            sectors,
            window,
            mp_dps=mp_dps,
            cache_dir=dirs["cache"],
            cache_payload={
                "kind": "left_connected",
                "r": r,
                "grid_N": grid_N,
                "hp_target_log10": float(hp_target_log10),
                "p": float(p),
                "q": float(q),
            },
            force=force,
            max_workers=max_workers,
            n_iter=hp_n_iter,
            hp_target_log10=hp_target_log10,
            p=p,
            q=q,
            calculation_kind="left_connected",
            logger=logger,
        )
        total_runtime += time.perf_counter() - t0
        last_info = info
        component = component_contours_from_log_grid(logF, xs, ys, (0.0 + 0j, Delta + 0j), logeps)
        contours = component["contours"]
        status = component["contour_status"]
        touches = bool(component["touches_boundary"])
        connected = bool(component["connected"])
        if contours and not touches:
            break
        window = window.expanded(2.0)
    return {
        "contours": contours,
        "contour_status": status if contours else ("exceeds_window" if touches else status),
        "contour_source": "full_direct_sum_global_hp_connected_grid",
        "contour_touches_boundary": touches,
        "E0_E1_connected_at_epsilon_vis": connected,
        "window_expansions": int(expansions),
        "local_window_radius": float(max(used_window.x_max - used_window.x_min, used_window.y_max - used_window.y_min) / 2.0),
        "from_cache": bool(last_info.get("from_cache", False)),
        "runtime_seconds": float(total_runtime),
        "final_failed": bool((not contours) or touches),
    }


def compute_history_left_outputs(
    r_list: list[int],
    epsilon_vis: float,
    grid_N_left: int,
    mp_dps: int,
    out_dir: str | Path,
    p: float,
    q: float,
    theta: float,
    omega: float,
    force: bool,
    max_workers: int,
    logger: logging.Logger,
    hp_n_iter: int = 60,
    radial_refinement: bool = True,
    hp_target_log10: float = -34.0,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    dirs = ensure_dirs(out_dir)
    logeps = math.log10(float(epsilon_vis))
    records: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    contour_idx = 0
    for r in r_list:
        sectors = history_sector_data_for_r(r, n=5, p=p, q=q)
        L = len(sectors[0]["v"])
        Delta = float(clock_eigenvalues(L)[1])
        summary, _ = global_condition_estimate_for_r(r, n=5, p=p, q=q)
        threshold_est = float(summary["log10_epsilon_est_global"])
        if logeps < threshold_est:
            for center_type, center_value, center in [("E0", 0.0, 0.0 + 0j), ("E1", Delta, Delta + 0j)]:
                result = compute_local_fk_component_contour_hp(
                    sectors,
                    r,
                    L,
                    center_type,
                    center,
                    epsilon_vis,
                    grid_N_left,
                    mp_dps,
                    hp_n_iter,
                    out_dir,
                    p,
                    q,
                    force,
                    max_workers,
                    hp_target_log10,
                    logger,
                )
                if radial_refinement and result["final_failed"]:
                    radius_hint = max(result["local_window_radius"], 1e-40)
                    contour = radial_log_boundary_global_hp(
                        sectors,
                        center,
                        logeps,
                        radius_hint=radius_hint,
                        mp_dps=mp_dps,
                        n_angles=65,
                        n_iter=hp_n_iter,
                        max_workers=max_workers,
                    )
                    result = {
                        **result,
                        "contours": [contour],
                        "contour_status": "ok_radial_fallback",
                        "contour_source": "radial_fallback_not_primary",
                        "contour_touches_boundary": False,
                        "final_failed": False,
                    }
                contour_idx = _append_fk_contour_records(
                    records,
                    arrays,
                    result["contours"],
                    contour_idx,
                    r=r,
                    L=L,
                    E1=Delta,
                    epsilon_vis=epsilon_vis,
                    logeps=logeps,
                    center_type=center_type,
                    center_value=center_value,
                    contour_status=result["contour_status"],
                    contour_source=result["contour_source"],
                    contour_touches_boundary=result["contour_touches_boundary"],
                    connected=False,
                    grid_N=grid_N_left,
                    local_window_radius=result["local_window_radius"],
                    window_expansions=result["window_expansions"],
                    mp_dps=mp_dps,
                    hp_n_iter=hp_n_iter,
                    from_cache=result["from_cache"],
                    runtime_seconds=result["runtime_seconds"],
                    final_failed=result["final_failed"],
                )
        else:
            result = compute_connected_fk_component_contour_hp(
                sectors,
                r,
                L,
                Delta,
                epsilon_vis,
                grid_N_left,
                mp_dps,
                hp_n_iter,
                out_dir,
                p,
                q,
                force,
                max_workers,
                hp_target_log10,
                logger,
            )
            if radial_refinement and result["final_failed"] and not result["contour_touches_boundary"]:
                contour = radial_log_boundary_global_hp(
                    sectors,
                    0.5 * Delta + 0j,
                    logeps,
                    radius_hint=max(Delta, 1e-40),
                    mp_dps=mp_dps,
                    n_angles=65,
                    n_iter=hp_n_iter,
                    max_workers=max_workers,
                )
                result = {
                    **result,
                    "contours": [contour],
                    "contour_status": "ok_radial_fallback",
                    "contour_source": "radial_fallback_not_primary",
                    "contour_touches_boundary": False,
                    "final_failed": False,
                }
            contour_idx = _append_fk_contour_records(
                records,
                arrays,
                result["contours"],
                contour_idx,
                r=r,
                L=L,
                E1=Delta,
                epsilon_vis=epsilon_vis,
                logeps=logeps,
                center_type="connected_E0_E1",
                center_value=0.5 * Delta,
                contour_status=result["contour_status"],
                contour_source=result["contour_source"],
                contour_touches_boundary=result["contour_touches_boundary"],
                connected=result["E0_E1_connected_at_epsilon_vis"],
                grid_N=grid_N_left,
                local_window_radius=result["local_window_radius"],
                window_expansions=result["window_expansions"],
                mp_dps=mp_dps,
                hp_n_iter=hp_n_iter,
                from_cache=result["from_cache"],
                runtime_seconds=result["runtime_seconds"],
                final_failed=result["final_failed"],
            )

    for algorithm, sigma in [("NHMIS-HDQAA", q), ("HMIS-HDQAA", 1.0)]:
        H = hd_block(sigma, theta=theta, omega=omega)
        for center in [0.0, float(omega)]:
            contour = _hd_radial_contour(center + 0j, epsilon_vis, H)
            key = f"contour_{contour_idx}"
            arrays[key] = contour
            center_type = "E0" if abs(center) < 1e-15 else "E1"
            records.append(
                {
                    "contour_key": key,
                    "algorithm": algorithm,
                    "n": 5,
                    "m": 2,
                    "r": "independent_of_r",
                    "L": "",
                    "E0": 0.0,
                    "E1": float(omega),
                    "center_type": center_type,
                    "center_value": float(center),
                    "epsilon_vis": epsilon_vis,
                    "log10_epsilon_vis": logeps,
                    "E0_E1_connected_at_epsilon_vis": False,
                    "contour_touches_boundary": False,
                    "contour_status": "analytic_2x2_radial",
                    "contour_source": "hd_local_block",
                    "grid_N": 0,
                    "local_window_radius": float(epsilon_vis),
                    "window_expansions": 0,
                    "mp_dps": int(mp_dps),
                    "hp_n_iter": int(hp_n_iter),
                    "from_cache": False,
                    "runtime_seconds": 0.0,
                    "final_failed": False,
                    "contour_center": center,
                }
            )
            contour_idx += 1
    return records, arrays


def compute_history_right_thresholds(
    r_list: list[int],
    grid_N_right: int,
    epsilon_vis: float,
    hp_target_log10: float,
    mp_dps: int,
    out_dir: str | Path,
    p: float,
    q: float,
    force: bool,
    max_workers: int,
    logger: logging.Logger,
    hp_n_iter: int = 60,
    max_grid_candidates: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dirs = ensure_dirs(out_dir)
    rows: list[dict] = []
    convergence_rows: list[dict] = []
    sector_estimate_tables: list[pd.DataFrame] = []
    for r in r_list:
        t_est = time.perf_counter()
        summary, est_df = global_condition_estimate_for_r(r, n=5, p=p, q=q)
        sector_estimate_tables.append(est_df)
        estimate_runtime = time.perf_counter() - t_est
        candidates = candidate_modes_from_global_estimates(est_df, summary, within_decades=2.0)
        if max_grid_candidates is not None:
            keep = {1}
            if int(summary["L"]) >= 2:
                keep.add(2)
            keep.add(int(summary["k_star_est"]))
            for k in candidates:
                keep.add(int(k))
                if len(keep) >= int(max_grid_candidates):
                    break
            candidates = sorted(k for k in keep if k in candidates)[: int(max_grid_candidates)]
        sectors = history_sector_data_for_r(r, n=5, p=p, q=q)
        L = int(summary["L"])
        lambdas = clock_eigenvalues(L)
        best_log = math.inf
        best_k = None
        touches_any = False
        expansions_total = 0
        grid_runtime = 0.0
        status = "success"
        for k in candidates:
            lam = float(lambdas[int(k)])
            window = GridWindow(-0.2 * lam, 1.2 * lam, -0.8 * lam, 0.8 * lam, grid_N_right, grid_N_right)
            result = None
            last_info = {"from_cache": False}
            k_runtime = 0.0
            for expansions in range(7):
                t0 = time.perf_counter()
                xs, ys, logF, cache_info = compute_log_grid_global_hp(
                    sectors,
                    window,
                    mp_dps=mp_dps,
                    cache_dir=dirs["cache"],
                    cache_payload={
                        "kind": "right",
                        "r": r,
                        "k": int(k),
                        "grid_N": grid_N_right,
                        "hp_target_log10": float(hp_target_log10),
                        "p": float(p),
                        "q": float(q),
                    },
                    force=force,
                    max_workers=max_workers,
                    n_iter=hp_n_iter,
                    hp_target_log10=hp_target_log10,
                    p=p,
                    q=q,
                    calculation_kind="right_threshold",
                    logger=logger,
                )
                elapsed = time.perf_counter() - t0
                k_runtime += elapsed
                grid_runtime += elapsed
                last_info = cache_info
                result = pseudospectral_closing_threshold_grid_log(
                    logF,
                    xs,
                    ys,
                    eigenvalues=[0.0 + 0j, complex(lam, 0.0)],
                    ground_indices=[0],
                    excited_indices=[1],
                )
                if not result["touches_boundary"]:
                    break
                window = window.expanded(2.0)
            expansions_total += expansions
            touches_any = touches_any or bool(result["touches_boundary"])
            convergence_rows.append(
                {
                    "r": r,
                    "L": L,
                    "k": int(k),
                    "grid_N": int(grid_N_right),
                    "log10_epsilon_c_grid_hp": float(result["log10_epsilon_c"]),
                    "log10_epsilon_est_global": float(summary["log10_epsilon_est_global"]),
                    "abs_difference_from_previous_grid": np.nan,
                    "touches_boundary": bool(result["touches_boundary"]),
                    "window_expansions": int(expansions),
                    "runtime_seconds": float(k_runtime),
                    "from_cache": bool(last_info.get("from_cache", False)),
                    "status": "main_grid",
                    "converged": "",
                    "provisional": "",
                }
            )
            if np.isfinite(result["log10_epsilon_c"]) and float(result["log10_epsilon_c"]) < best_log:
                best_log = float(result["log10_epsilon_c"])
                best_k = int(k)
        if best_k is None:
            status = "failed_no_finite_grid_threshold"
            best_log = math.nan
        rows.append(
            {
                "r": int(r),
                "L": L,
                "n": 5,
                "m": 2,
                "Delta_FK": float(summary["Delta_FK"]),
                "log10_epsilon_vis": math.log10(float(epsilon_vis)),
                "hp_target_log10": float(hp_target_log10),
                "mp_dps": int(mp_dps),
                "grid_N_right": int(grid_N_right),
                "log10_epsilon_c_grid_hp": float(best_log),
                "k_grid_best": best_k if best_k is not None else "",
                "candidate_k_list": " ".join(str(k) for k in candidates),
                "log10_epsilon_est_global": float(summary["log10_epsilon_est_global"]),
                "k_star_est": int(summary["k_star_est"]),
                "sector_est_min": int(summary["sector_est_min"]),
                "sector_est_min_bitstring": str(summary["sector_est_min_bitstring"]),
                "log10_chi_0": float(summary["log10_chi_0"]),
                "log10_chi_kstar": float(summary["log10_chi_kstar"]),
                "log10_kappa_S_est_sector": float(summary["log10_kappa_S_est_sector"]),
                "grid_runtime_seconds": float(grid_runtime),
                "estimate_runtime_seconds": float(estimate_runtime),
                "reliable_hp_grid": bool(np.isfinite(best_log) and best_log >= hp_target_log10 and not touches_any),
                "touches_boundary": bool(touches_any),
                "window_expansions": int(expansions_total),
                "status": status,
            }
        )
        logger.info(
            "history HP r=%d L=%d log10_grid=%s log10_est=%.3f candidates=%s",
            r,
            L,
            f"{best_log:.6g}" if np.isfinite(best_log) else "nan",
            summary["log10_epsilon_est_global"],
            candidates,
        )
    sector_df = pd.concat(sector_estimate_tables, ignore_index=True) if sector_estimate_tables else pd.DataFrame()
    return pd.DataFrame(rows), pd.DataFrame(convergence_rows), sector_df


def run_history_runtime_benchmark(
    out_dir: str | Path,
    machine_label: str = "local",
    worker_list: list[int] | None = None,
    reserve_cores: int = 2,
    r: int = 5,
    k: int = 1,
    grid_N: int = 21,
    mp_dps: int = 70,
    hp_n_iter: int = 24,
    force: bool = False,
    p: float = 2.0,
    q: float = 4.0,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    dirs = ensure_dirs(out_dir)
    cpu = os.cpu_count() or 1
    if worker_list is None:
        base = [16, 32, 48, max(1, cpu - reserve_cores)]
        worker_list = sorted({min(max(1, cpu - reserve_cores), w) for w in base if w >= 1})
    sectors = history_sector_data_for_r(r, n=5, p=p, q=q)
    L = len(sectors[0]["v"])
    lam = float(clock_eigenvalues(L)[int(k)])
    window = GridWindow(-0.2 * lam, 1.2 * lam, -0.8 * lam, 0.8 * lam, int(grid_N), int(grid_N))
    rows = []
    for workers in worker_list:
        t0 = time.perf_counter()
        xs, ys, _logF, info = compute_log_grid_global_hp(
            sectors,
            window,
            mp_dps=mp_dps,
            cache_dir=dirs["cache"],
            cache_payload={
                "kind": "runtime_benchmark",
                "machine_label": machine_label,
                "r": int(r),
                "k": int(k),
                "grid_N": int(grid_N),
                "workers": int(workers),
                "p": float(p),
                "q": float(q),
            },
            force=force,
            max_workers=int(workers),
            n_iter=int(hp_n_iter),
            hp_target_log10=-34.0,
            p=p,
            q=q,
            calculation_kind="runtime_benchmark",
            logger=logger,
        )
        runtime = time.perf_counter() - t0
        points = int(len(xs) * len(ys))
        pps = points / max(runtime, 1e-12)
        final_points = 5 * 2 * (61**2)
        iter_factor = 40.0 / max(float(hp_n_iter), 1.0)
        estimated_final_hours = final_points * iter_factor / max(pps, 1e-12) / 3600.0
        rows.append(
            {
                "machine_label": machine_label,
                "cpu_count": cpu,
                "max_workers": int(workers),
                "grid_N": int(grid_N),
                "r": int(r),
                "k": int(k),
                "mp_dps": int(mp_dps),
                "hp_n_iter": int(hp_n_iter),
                "runtime_seconds": float(runtime),
                "points_per_second": float(pps),
                "estimated_final_hours": float(estimated_final_hours),
                "from_cache": bool(info.get("from_cache", False)),
            }
        )
    df = pd.DataFrame(rows)
    path = dirs["data"] / "runtime_benchmark.csv"
    df.to_csv(path, index=False)
    return df


def run_history_threshold_convergence(
    r_list: list[int],
    grid_list: list[int],
    out_dir: str | Path,
    epsilon_vis: float,
    hp_target_log10: float,
    mp_dps: int,
    hp_n_iter: int,
    force: bool,
    max_workers: int,
    p: float,
    q: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    dirs = ensure_dirs(out_dir)
    rows: list[dict] = []
    for r in r_list:
        summary, est_df = global_condition_estimate_for_r(r, n=5, p=p, q=q)
        k_star = int(summary["k_star_est"])
        candidates = {1, k_star}
        if k_star == 1 and int(summary["L"]) >= 2:
            candidates.add(2)
        candidates = sorted(candidates)
        sectors = history_sector_data_for_r(r, n=5, p=p, q=q)
        L = int(summary["L"])
        lambdas = clock_eigenvalues(L)
        for k in candidates:
            previous = None
            for grid_N in grid_list:
                lam = float(lambdas[int(k)])
                window = GridWindow(-0.2 * lam, 1.2 * lam, -0.8 * lam, 0.8 * lam, int(grid_N), int(grid_N))
                result = None
                last_info = {"from_cache": False}
                k_runtime = 0.0
                for expansions in range(7):
                    t0 = time.perf_counter()
                    xs, ys, logF, info = compute_log_grid_global_hp(
                        sectors,
                        window,
                        mp_dps=mp_dps,
                        cache_dir=dirs["cache"],
                        cache_payload={
                            "kind": "right_convergence",
                            "r": int(r),
                            "k": int(k),
                            "grid_N": int(grid_N),
                            "hp_target_log10": float(hp_target_log10),
                            "p": float(p),
                            "q": float(q),
                        },
                        force=force,
                        max_workers=max_workers,
                        n_iter=hp_n_iter,
                        hp_target_log10=hp_target_log10,
                        p=p,
                        q=q,
                        calculation_kind="right_convergence",
                        logger=logger,
                    )
                    elapsed = time.perf_counter() - t0
                    k_runtime += elapsed
                    last_info = info
                    result = pseudospectral_closing_threshold_grid_log(
                        logF,
                        xs,
                        ys,
                        eigenvalues=[0.0 + 0j, complex(lam, 0.0)],
                        ground_indices=[0],
                        excited_indices=[1],
                    )
                    if not result["touches_boundary"]:
                        break
                    window = window.expanded(2.0)
                value = float(result["log10_epsilon_c"])
                diff = abs(value - previous) if previous is not None and np.isfinite(value) and np.isfinite(previous) else np.nan
                previous = value
                rows.append(
                    {
                        "r": int(r),
                        "L": L,
                        "k": int(k),
                        "grid_N": int(grid_N),
                        "log10_epsilon_c_grid_hp": value,
                        "log10_epsilon_est_global": float(summary["log10_epsilon_est_global"]),
                        "abs_difference_from_previous_grid": diff,
                        "touches_boundary": bool(result["touches_boundary"]),
                        "window_expansions": int(expansions),
                        "runtime_seconds": float(k_runtime),
                        "from_cache": bool(last_info.get("from_cache", False)),
                        "status": "convergence_grid",
                        "converged": bool(np.isfinite(diff) and diff < 0.3 and int(grid_N) >= 61),
                        "provisional": bool(int(grid_N) < 61),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(dirs["data"] / "fig2_right_history_threshold_convergence_hp.csv", index=False)
    return df


def plot_history_left(records: list[dict], arrays: dict[str, np.ndarray], figures_dir: str | Path) -> tuple[Path, Path, Path, Path]:
    r_values = sorted({int(r["r"]) for r in records if str(r["r"]).isdigit()})
    cmap = plt.get_cmap("viridis")
    colors = {r: cmap(i / max(len(r_values) - 1, 1)) for i, r in enumerate(r_values)}
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 3.9), constrained_layout=True)
    panels = [
        ("NHMIS-FKQAA-full-direct-sum", "Panel A: FK full-sector union"),
        ("NHMIS-HDQAA", "Panel B: NHMIS-HD local block"),
        ("HMIS-HDQAA", "Panel C: HMIS-HD local block"),
    ]
    for ax, (algorithm, title) in zip(axes, panels):
        subset = [row for row in records if row["algorithm"] == algorithm]
        for row in subset:
            key = row.get("contour_key", "")
            if not key or key not in arrays:
                continue
            if algorithm in {"NHMIS-HDQAA", "HMIS-HDQAA"}:
                continue
            arr = arrays[key]
            if str(row["r"]).isdigit():
                color = colors[int(row["r"])]
                label = f"r={row['r']}"
            else:
                color = "#ff7f0e"
                label = "independent of r"
            ax.plot(arr[:, 0], arr[:, 1], color=color, lw=1.2, label=label)
        seen = set()
        for row in subset:
            if str(row["r"]).isdigit():
                r = int(row["r"])
                if r in seen:
                    continue
                seen.add(r)
                color = colors[r]
                ax.scatter([0.0], [0.0], marker="s", s=30, color=color, edgecolor="black", linewidth=0.4)
                ax.scatter([float(row["E1"])], [0.0], marker="^", s=34, color=color, edgecolor="black", linewidth=0.4)
                if bool(row["E0_E1_connected_at_epsilon_vis"]):
                    ax.text(float(row["E1"]) * 0.45, 0.0, "closed", color=color, fontsize=8)
                if bool(row["contour_touches_boundary"]):
                    ax.text(float(row["E1"]) * 0.2, 0.0, "exceeds window", color=color, fontsize=8)
        if algorithm in {"NHMIS-HDQAA", "HMIS-HDQAA"}:
            ax.scatter([0.0], [0.0], marker="s", s=34, color="#ff7f0e", edgecolor="black", linewidth=0.4)
            ax.scatter([1.0], [0.0], marker="^", s=38, color="#ff7f0e", edgecolor="black", linewidth=0.4)
            ax.text(0.05, 0.05, "local contours are shown in insets", transform=ax.transAxes, fontsize=8)
            ax.set_xlim(-0.08, 1.08)
            ax.set_ylim(-0.12, 0.12)
            for box, center, title_label in [([0.08, 0.53, 0.34, 0.38], 0.0, "E0"), ([0.58, 0.53, 0.34, 0.38], 1.0, "E1")]:
                inset = ax.inset_axes(box)
                for row in subset:
                    key = row.get("contour_key", "")
                    if not key or key not in arrays:
                        continue
                    if abs(float(row.get("contour_center", np.nan)) - center) > 1e-12:
                        continue
                    arr = arrays[key]
                    inset.plot((arr[:, 0] - center) / 1e-32, arr[:, 1] / 1e-32, color="#ff7f0e", lw=1.0)
                inset.set_title(f"{title_label}: local / 1e-32", fontsize=7)
                inset.set_xlabel("Re", fontsize=6)
                inset.set_ylabel("Im", fontsize=6)
                inset.tick_params(labelsize=6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Re z")
        ax.set_ylabel("Im z")
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
        handles, labels = ax.get_legend_handles_labels()
        unique = {}
        for h, label in zip(handles, labels):
            unique.setdefault(label, h)
        if unique:
            ax.legend(unique.values(), unique.keys(), fontsize=7, loc="best")
    png = Path(figures_dir) / "fig2_left_history_pseudospectrum_hp.png"
    pdf = Path(figures_dir) / "fig2_left_history_pseudospectrum_hp.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.6), constrained_layout=True)
    subset = [row for row in records if row["algorithm"] == "NHMIS-FKQAA-full-direct-sum"]
    for row in subset:
        key = row.get("contour_key", "")
        if not key or key not in arrays:
            continue
        r = int(row["r"])
        arr = arrays[key]
        ax.plot(arr[:, 0], arr[:, 1], color=colors[r], lw=1.3, label=f"r={r}")
        ax.scatter([0.0], [0.0], marker="s", s=25, color=colors[r], edgecolor="black", linewidth=0.4)
        ax.scatter([float(row["E1"])], [0.0], marker="^", s=30, color=colors[r], edgecolor="black", linewidth=0.4)
    ax.set_title("FK full-sector union history zoom")
    ax.set_xlabel("Re z")
    ax.set_ylabel("Im z")
    ax.grid(True, linestyle=":", linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for h, label in zip(handles, labels):
        unique.setdefault(label, h)
    if unique:
        ax.legend(unique.values(), unique.keys(), fontsize=8)
    zoom_png = Path(figures_dir) / "fig2_left_fk_history_zoom_hp.png"
    zoom_pdf = Path(figures_dir) / "fig2_left_fk_history_zoom_hp.pdf"
    fig.savefig(zoom_png, dpi=220)
    fig.savefig(zoom_pdf)
    plt.close(fig)
    return png, pdf, zoom_png, zoom_pdf


def plot_history_right(threshold_df: pd.DataFrame, figures_dir: str | Path) -> tuple[Path, Path]:
    df = threshold_df.sort_values("r")
    fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
    ax.scatter(df["r"], df["log10_epsilon_c_grid_hp"], color="#1f77b4", s=42, label="high-precision grid threshold", zorder=3)
    ax.plot(df["r"], df["log10_epsilon_est_global"], color="#2ca02c", linestyle="--", lw=1.6, label="condition-number estimate")
    eps_vis = float(df["log10_epsilon_vis"].iloc[0])
    ax.axhline(eps_vis, color="#d62728", linestyle=":", lw=1.2, label=f"epsilon_vis = 1e{int(eps_vis)}")
    ax.set_xlabel("r  (L = 14 r)")
    ax.set_ylabel("log10 epsilon")
    ymin = math.floor(float(np.nanmin([df["log10_epsilon_c_grid_hp"].min(), df["log10_epsilon_est_global"].min()])) / 5.0) * 5
    ymax = math.ceil(float(np.nanmax([df["log10_epsilon_c_grid_hp"].max(), df["log10_epsilon_est_global"].max(), eps_vis])) / 5.0) * 5
    ticks = np.arange(ymin, ymax + 1, 5)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{int(t)}" for t in ticks])
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend(loc="best", fontsize=8)
    png = Path(figures_dir) / "fig2_right_fk_history_threshold_hp.png"
    pdf = Path(figures_dir) / "fig2_right_fk_history_threshold_hp.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def run_experiment_two_history_hp(
    r_list: list[int] | None = None,
    epsilon_vis: float = 1e-32,
    hp_target_log10: float = -35.0,
    mp_dps: int = 90,
    grid_N_left: int = 101,
    grid_N_right: int = 101,
    grid_list: list[int] | None = None,
    quick: bool = False,
    convergence: bool = False,
    final_overnight: bool = False,
    final_heavy: bool = False,
    runtime_benchmark: bool = False,
    machine_label: str = "local",
    runtime_worker_list: list[int] | None = None,
    runtime_grid_N: int = 21,
    force: bool = False,
    out_dir: str | Path = Path("outputs") / "experiment_two_resolvent_hp",
    reserve_cores: int = 2,
    max_workers: int | None = None,
    skip_left: bool = False,
    skip_right: bool = False,
    plot_only: bool = False,
    benchmark_only: bool = False,
    hp_n_iter: int = 60,
    radial_refinement: bool = True,
    max_grid_candidates: int | None = None,
    p: float = 2.0,
    q: float = 4.0,
    theta: float = np.pi / 4,
    omega: float = 1.0,
) -> dict:
    ensure_mpmath_available()
    requested_r_list = None if r_list is None else list(r_list)
    run_mode = "quick" if quick else "default"
    if final_overnight:
        run_mode = "final-overnight"
        r_list = [1, 2, 3, 4, 5]
        mp_dps = 70
        hp_n_iter = 40
        hp_target_log10 = -34.0
        grid_N_right = 61
        grid_N_left = 41
        epsilon_vis = 1e-32
        radial_refinement = True
        max_grid_candidates = None
    if final_heavy:
        run_mode = "final-heavy"
        r_list = [1, 2, 3, 4, 5]
        mp_dps = 80
        hp_n_iter = 60
        hp_target_log10 = -35.0
        grid_N_right = 81
        grid_N_left = 61
        epsilon_vis = 1e-32
        radial_refinement = True
        max_grid_candidates = None
    if quick:
        r_list = [1, 5] if r_list is None else r_list
        grid_N_left = min(int(grid_N_left), 51)
        grid_N_right = min(int(grid_N_right), 51)
        mp_dps = min(int(mp_dps), 70)
    else:
        r_list = [1, 2, 3, 4, 5] if r_list is None else r_list
    if runtime_benchmark and not final_overnight and not final_heavy:
        if int(mp_dps) == 90:
            mp_dps = 70
        if int(hp_n_iter) == 60:
            hp_n_iter = 24
    workers = default_history_hp_workers(reserve_cores=reserve_cores, max_workers=max_workers)
    dirs = ensure_dirs(out_dir)
    logger = configure_history_hp_logger(out_dir, append=plot_only)
    logger.info(
        "history HP start quick=%s r_list=%s grid_left=%d grid_right=%d mp_dps=%d workers=%d reserve_cores=%d",
        quick,
        r_list,
        grid_N_left,
        grid_N_right,
        mp_dps,
        workers,
        reserve_cores,
    )

    if plot_only:
        left_csv = dirs["data"] / "fig2_left_history_contours_metadata.csv"
        left_npz = dirs["data"] / "fig2_left_history_contours_hp.npz"
        right_csv = dirs["data"] / "fig2_right_history_thresholds_hp.csv"
        result = {}
        if left_csv.exists() and left_npz.exists():
            arrays = dict(np.load(left_npz))
            records = pd.read_csv(left_csv).to_dict("records")
            result["left_figures"] = plot_history_left(records, arrays, dirs["figures"])
        if right_csv.exists():
            result["right_figures"] = plot_history_right(pd.read_csv(right_csv), dirs["figures"])
        return result

    benchmark_df = run_hp_solver_benchmark(out_dir, mp_dps=mp_dps, p=p, q=q, force=force)
    if benchmark_only:
        return {"benchmark_rows": len(benchmark_df), "out_dir": str(out_dir)}

    if runtime_benchmark:
        runtime_df = run_history_runtime_benchmark(
            out_dir=out_dir,
            machine_label=machine_label,
            worker_list=runtime_worker_list,
            reserve_cores=reserve_cores,
            grid_N=runtime_grid_N,
            mp_dps=mp_dps,
            hp_n_iter=hp_n_iter,
            force=force,
            p=p,
            q=q,
            logger=logger,
        )
        return {
            "runtime_benchmark": str(dirs["data"] / "runtime_benchmark.csv"),
            "rows": runtime_df.to_dict("records"),
        }

    if convergence:
        conv_r_list = [1, 3, 5] if requested_r_list is None else r_list
        conv_grid_list = [21, 41, 61] if grid_list is None else grid_list
        conv_df = run_history_threshold_convergence(
            r_list=conv_r_list,
            grid_list=conv_grid_list,
            out_dir=out_dir,
            epsilon_vis=epsilon_vis,
            hp_target_log10=hp_target_log10,
            mp_dps=mp_dps,
            hp_n_iter=hp_n_iter,
            force=force,
            max_workers=workers,
            p=p,
            q=q,
            logger=logger,
        )
        return {
            "convergence_csv": str(dirs["data"] / "fig2_right_history_threshold_convergence_hp.csv"),
            "rows": conv_df.to_dict("records"),
        }

    left_paths: dict[str, str] = {}
    if not skip_left:
        left_records, left_arrays = compute_history_left_outputs(
            r_list=r_list,
            epsilon_vis=epsilon_vis,
            grid_N_left=grid_N_left,
            mp_dps=mp_dps,
            out_dir=out_dir,
            p=p,
            q=q,
            theta=theta,
            omega=omega,
            force=force,
            max_workers=workers,
            logger=logger,
            hp_n_iter=hp_n_iter,
            radial_refinement=radial_refinement,
            hp_target_log10=hp_target_log10,
        )
        left_npz = dirs["data"] / "fig2_left_history_contours_hp.npz"
        left_csv = dirs["data"] / "fig2_left_history_contours_metadata.csv"
        np.savez_compressed(left_npz, **left_arrays)
        pd.DataFrame(left_records).to_csv(left_csv, index=False)
        left_png, left_pdf, zoom_png, zoom_pdf = plot_history_left(left_records, left_arrays, dirs["figures"])
        left_paths = {
            "left_npz": str(left_npz),
            "left_csv": str(left_csv),
            "left_png": str(left_png),
            "left_pdf": str(left_pdf),
            "left_zoom_png": str(zoom_png),
            "left_zoom_pdf": str(zoom_pdf),
        }
    else:
        existing_left = {
            "left_npz": dirs["data"] / "fig2_left_history_contours_hp.npz",
            "left_csv": dirs["data"] / "fig2_left_history_contours_metadata.csv",
            "left_png": dirs["figures"] / "fig2_left_history_pseudospectrum_hp.png",
            "left_pdf": dirs["figures"] / "fig2_left_history_pseudospectrum_hp.pdf",
            "left_zoom_png": dirs["figures"] / "fig2_left_fk_history_zoom_hp.png",
            "left_zoom_pdf": dirs["figures"] / "fig2_left_fk_history_zoom_hp.pdf",
        }
        left_paths = {key: str(path) for key, path in existing_left.items() if path.exists()}

    right_paths: dict[str, str] = {}
    threshold_df = pd.DataFrame()
    convergence_df = pd.DataFrame()
    sector_df = pd.DataFrame()
    if not skip_right:
        threshold_df, convergence_df, sector_df = compute_history_right_thresholds(
            r_list=r_list,
            grid_N_right=grid_N_right,
            epsilon_vis=epsilon_vis,
            hp_target_log10=hp_target_log10,
            mp_dps=mp_dps,
            out_dir=out_dir,
            p=p,
            q=q,
            force=force,
            max_workers=workers,
            logger=logger,
            hp_n_iter=hp_n_iter,
            max_grid_candidates=max_grid_candidates,
        )
        threshold_csv = dirs["data"] / "fig2_right_history_thresholds_hp.csv"
        conv_csv = dirs["data"] / "fig2_right_history_threshold_convergence_hp.csv"
        candidate_grid_csv = dirs["data"] / "fig2_right_history_threshold_grid_candidates_hp.csv"
        sector_csv = dirs["data"] / "sector_estimates_by_r_hp.csv"
        threshold_df.to_csv(threshold_csv, index=False)
        convergence_df.to_csv(candidate_grid_csv, index=False)
        if not conv_csv.exists():
            convergence_df.to_csv(conv_csv, index=False)
        sector_df.to_csv(sector_csv, index=False)
        right_png, right_pdf = plot_history_right(threshold_df, dirs["figures"])
        right_paths = {
            "right_csv": str(threshold_csv),
            "convergence_csv": str(conv_csv),
            "candidate_grid_csv": str(candidate_grid_csv),
            "sector_csv": str(sector_csv),
            "right_png": str(right_png),
            "right_pdf": str(right_pdf),
        }

    metadata = {
        "experiment": "history_accumulation_resolvent_high_precision",
        "run_mode": run_mode,
        "n": 5,
        "m": 2,
        "r_list": r_list,
        "p": p,
        "q": q,
        "Omega": omega,
        "theta": theta,
        "epsilon_vis": epsilon_vis,
        "log10_epsilon_vis": math.log10(float(epsilon_vis)),
        "hp_target_log10": hp_target_log10,
        "mp_dps": mp_dps,
        "grid_N_left": grid_N_left,
        "grid_N_right": grid_N_right,
        "reserve_cores": reserve_cores,
        "max_workers": workers,
        "hp_n_iter": hp_n_iter,
        "radial_refinement": radial_refinement,
        "max_grid_candidates": max_grid_candidates,
        "gate_order": GATE_ORDER,
        "full_fk_definition": "f_global(z;r)=min_x sigma_min(zI-H_x(r)) over all 32 n=5 sectors",
        "mpmath_benchmark": {
            "path": str(dirs["data"] / "hp_solver_benchmark.csv"),
            "all_pass": bool((benchmark_df["status"] == "PASS").all()),
        },
        "paths": {**left_paths, **right_paths},
        "old_random_perturbation_preserved_but_not_used": True,
        "date_time": datetime.now().isoformat(timespec="seconds"),
        "git_commit_hash": _git_commit_hash(),
    }
    metadata_path = dirs["data"] / "experiment2_history_hp_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    logger.info("history HP finished out_dir=%s", out_dir)
    return {
        "out_dir": str(out_dir),
        "metadata": str(metadata_path),
        "benchmark_rows": len(benchmark_df),
        "threshold_rows": threshold_df.to_dict("records") if not threshold_df.empty else [],
        "paths": metadata["paths"],
    }


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return None


def run_experiment_two_resolvent(
    n_left: list[int] | None = None,
    n_right: list[int] | None = None,
    epsilon_vis: float = 1e-15,
    eps_floor: float = 1e-15,
    grid_N: int = 300,
    p: float = 2.0,
    q: float = 4.0,
    theta: float = np.pi / 4,
    omega: float = 1.0,
    out_dir: str | Path = Path("outputs") / "experiment_two_resolvent",
    quick: bool = False,
    force: bool = False,
    skip_left: bool = False,
    skip_right: bool = False,
    plot_only: bool = False,
    max_workers: int | None = None,
    exact_grid_n_max: int | None = None,
) -> dict:
    if quick:
        n_left = [5, 9] if n_left is None else n_left
        n_right = [5, 9, 13] if n_right is None else n_right
        grid_N = min(int(grid_N), 120)
    else:
        n_left = [5, 9, 13, 17] if n_left is None else n_left
        n_right = [5, 9, 13, 17, 21, 25, 29, 33, 37] if n_right is None else n_right
    exact_grid_n_max = 9 if exact_grid_n_max is None else int(exact_grid_n_max)
    _ = force, max_workers if max_workers is not None else default_max_workers()
    dirs = ensure_dirs(out_dir)
    logger = configure_logger(out_dir, append=plot_only)
    logger.info("starting experiment two resolvent quick=%s grid_N=%d", quick, grid_N)

    if plot_only:
        threshold_csv = dirs["data"] / "fig2_right_thresholds.csv"
        left_csv = dirs["data"] / "fig2_left_contours_metadata.csv"
        left_npz = dirs["data"] / "fig2_left_contours.npz"
        if threshold_csv.exists():
            plot_fig2_right(pd.read_csv(threshold_csv), dirs["figures"])
        if left_csv.exists() and left_npz.exists():
            arrs = dict(np.load(left_npz))
            recs = pd.read_csv(left_csv).to_dict("records")
            plot_fig2_left(recs, arrs, dirs["figures"], marker_mode="level")
            plot_fig2_left(recs, arrs, dirs["figures"], marker_mode="scheme")
        return {"out_dir": str(out_dir), "plot_only": True}

    validation_summaries: list[dict] = []
    selection_by_n: dict[int, tuple[int, ...]] = {}
    for n in [5, 9]:
        csv_path = dirs["data"] / f"sector_validation_n{n}.csv"
        df, summary = enumerate_sector_validation(n, p=p, q=q, out_csv=csv_path)
        validation_summaries.append(summary)
        selected_bits = tuple(int(ch) for ch in summary["selected_sector_bitstring"])
        if selected_bits != tuple(build_ck_graph_from_n(n)["x_mis"]):
            selection_by_n[n] = selected_bits
        logger.info(
            "sector validation n=%d: MIS rank gain=%d rank eps=%d selected=%s",
            n,
            summary["rank_mis_by_log_gain"],
            summary["rank_mis_by_epsilon_est"],
            summary["selected_sector_bitstring"],
        )

    left_paths = {}
    if not skip_left:
        left_records, left_arrays = compute_left_contours(
            n_left=n_left,
            epsilon_vis=epsilon_vis,
            grid_N=grid_N,
            p=p,
            q=q,
            theta=theta,
            omega=omega,
            selection_by_n=selection_by_n,
            logger=logger,
        )
        left_npz, left_csv = save_left_outputs(left_records, left_arrays, dirs["data"])
        level_png, level_pdf = plot_fig2_left(left_records, left_arrays, dirs["figures"], marker_mode="level")
        scheme_png, scheme_pdf = plot_fig2_left(left_records, left_arrays, dirs["figures"], marker_mode="scheme")
        left_paths = {
            "left_npz": str(left_npz),
            "left_csv": str(left_csv),
            "left_level_png": str(level_png),
            "left_level_pdf": str(level_pdf),
            "left_scheme_png": str(scheme_png),
            "left_scheme_pdf": str(scheme_pdf),
        }

    right_paths = {}
    threshold_df = pd.DataFrame()
    convergence_df = pd.DataFrame()
    if not skip_right:
        threshold_df, convergence_df = compute_right_thresholds(
            n_right=n_right,
            eps_floor=eps_floor,
            grid_N=grid_N,
            p=p,
            q=q,
            selection_by_n=selection_by_n,
            exact_grid_n_max=exact_grid_n_max,
            logger=logger,
        )
        threshold_csv = dirs["data"] / "fig2_right_thresholds.csv"
        convergence_csv = dirs["data"] / "fig2_right_threshold_convergence.csv"
        threshold_df.to_csv(threshold_csv, index=False)
        convergence_df.to_csv(convergence_csv, index=False)
        right_png, right_pdf = plot_fig2_right(threshold_df, dirs["figures"])
        right_paths = {
            "right_csv": str(threshold_csv),
            "convergence_csv": str(convergence_csv),
            "right_png": str(right_png),
            "right_pdf": str(right_pdf),
        }

    try:
        import scipy

        scipy_version = scipy.__version__
    except Exception:
        scipy_version = None
    metadata = {
        "p": p,
        "q": q,
        "Omega": omega,
        "theta": theta,
        "epsilon_vis": epsilon_vis,
        "eps_floor": eps_floor,
        "n_left": n_left,
        "n_right": n_right,
        "quick": quick,
        "gate_order": GATE_ORDER,
        "grid_settings": {"grid_N": grid_N, "exact_grid_n_max": exact_grid_n_max},
        "solver_settings": {
            "fk_sigma_solver": "banded_eig_banded_of_AstarA",
            "hd_sigma_solver": "stable_analytic_2x2",
            "tiny_floor": TINY_FLOOR,
        },
        "numpy_version": np.__version__,
        "scipy_version": scipy_version,
        "date_time": datetime.now().isoformat(timespec="seconds"),
        "git_commit_hash": _git_commit_hash(),
        "old_experiment_two_preserved_as_legacy": True,
        "sector_selection_rule": "MIS by default; n=5/n=9 can be overridden by exhaustive estimate validation",
        "sector_validation_summaries": validation_summaries,
        "paths": {**left_paths, **right_paths},
        "warnings": [],
    }
    metadata_path = dirs["data"] / "experiment2_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    logger.info("finished experiment two resolvent outputs in %s", out_dir)
    return {
        "out_dir": str(out_dir),
        "metadata": str(metadata_path),
        "left": left_paths,
        "right": right_paths,
        "sector_validation": validation_summaries,
        "threshold_rows": threshold_df.to_dict("records") if not threshold_df.empty else [],
        "convergence_rows": convergence_df.to_dict("records") if not convergence_df.empty else [],
    }
