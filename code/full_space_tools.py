"""Shared full-space utilities for experiments two and three."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix, eye

try:
    from .experiment_one_ck import build_ck_gate_sequence, ck_graph_info
except ImportError:  # pragma: no cover
    from experiment_one_ck import build_ck_gate_sequence, ck_graph_info


def _instance_from_graph(
    instance_name: str,
    graph_type: str,
    n: int,
    r: int,
    gates_one_round: list[tuple],
    edges: list[tuple[int, int]],
    x_star_bits: tuple[int, ...],
    p: float,
    q: float,
    gamma: float,
    m: int | None = None,
    num_edges: int | None = None,
) -> dict:
    gates = gates_one_round * int(r)
    L = len(gates)
    work_dim = 1 << n
    clock_dim = L + 1
    x_star_index = int(sum(bit << i for i, bit in enumerate(x_star_bits)))
    return {
        "instance_name": instance_name,
        "graph_type": graph_type,
        "m": m,
        "n": n,
        "r": int(r),
        "L": L,
        "num_edges": len(edges) if num_edges is None else int(num_edges),
        "edges": list(edges),
        "gates": gates,
        "work_dim": work_dim,
        "clock_dim": clock_dim,
        "full_dim": work_dim * clock_dim,
        "x_star_bits": tuple(x_star_bits),
        "x_star_index": x_star_index,
        "gamma": gamma,
        "p": p,
        "q": q,
    }


def _ck_m2_edges() -> list[tuple[int, int]]:
    return [
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ]


def build_full_space_instance(
    instance_name: str,
    p: float = 2.0,
    q: float = 4.0,
    gamma: float = 10.0,
) -> dict:
    """Build metadata for supported full-space experiment-three instances."""
    if instance_name in {"paper_m2", "ck_m2_r1"}:
        m = 2
        r_override = 1 if instance_name == "ck_m2_r1" else None
        info = ck_graph_info(m)
        n = info["n"]
        r = n if r_override is None else r_override
        gates_one_round = build_ck_gate_sequence(m, 1)
        return _instance_from_graph(
            instance_name=instance_name,
            graph_type="CK",
            n=n,
            r=r,
            gates_one_round=gates_one_round,
            edges=_ck_m2_edges(),
            x_star_bits=tuple(info["x_star"]),
            p=p,
            q=q,
            gamma=gamma,
            m=m,
            num_edges=info["num_edges"],
        )

    if instance_name in {"toy_star3_r1", "toy_star3_r2"}:
        r = 1 if instance_name == "toy_star3_r1" else 2
        edges = [(0, 1), (0, 2)]
        gates_one_round = [
            ("B", 0, 1),
            ("B", 0, 2),
            ("A", 0),
            ("A", 1),
            ("A", 2),
        ]
        return _instance_from_graph(
            instance_name=instance_name,
            graph_type="toy_star3",
            n=3,
            r=r,
            gates_one_round=gates_one_round,
            edges=edges,
            x_star_bits=(0, 1, 1),
            p=p,
            q=q,
            gamma=gamma,
        )

    raise ValueError(f"unknown full-space instance {instance_name!r}")


def build_m2_instance(p: float = 2.0, q: float = 4.0, gamma: float = 10.0) -> dict:
    """Build fixed CK m=2 full-space instance metadata."""
    return build_full_space_instance("paper_m2", p=p, q=q, gamma=gamma)


def _old_build_m2_instance(p: float = 2.0, q: float = 4.0, gamma: float = 10.0) -> dict:
    """Legacy implementation retained for reference during development."""
    m = 2
    info = ck_graph_info(m)
    n = info["n"]
    r = n
    gates = build_ck_gate_sequence(m, r)
    L = len(gates)
    work_dim = 1 << n
    clock_dim = L + 1
    x_star_bits = info["x_star"]
    x_star_index = int(sum(bit << i for i, bit in enumerate(x_star_bits)))
    return {
        "m": m,
        "n": n,
        "r": r,
        "L": L,
        "num_edges": info["num_edges"],
        "gates": gates,
        "work_dim": work_dim,
        "clock_dim": clock_dim,
        "full_dim": work_dim * clock_dim,
        "x_star_bits": x_star_bits,
        "x_star_index": x_star_index,
        "gamma": gamma,
        "p": p,
        "q": q,
    }


def bitstrings_array(n: int) -> np.ndarray:
    """Return bitstrings[x, i] = (x >> i) & 1."""
    return ((np.arange(1 << n, dtype=np.int64)[:, None] >> np.arange(n)) & 1)


def gate_eigenvalues(
    gate: tuple, bitstrings: np.ndarray, p: float = 2.0, q: float = 4.0
) -> np.ndarray:
    """Return diagonal gate eigenvalues v_l(x)."""
    if gate[0] == "A":
        return np.where(bitstrings[:, gate[1]] == 1, p, 1.0).astype(float)
    if gate[0] == "B":
        i, j = gate[1], gate[2]
        violated = bitstrings[:, i] * bitstrings[:, j]
        return np.where(violated == 1, 1.0, q).astype(float)
    raise ValueError(f"unknown gate kind {gate[0]}")


def all_gate_eigenvalues(
    gates: list[tuple], bitstrings: np.ndarray, p: float = 2.0, q: float = 4.0
) -> list[np.ndarray]:
    """Return list of gate eigenvalue arrays; index 0 is mathematical l=1."""
    return [gate_eigenvalues(gate, bitstrings, p=p, q=q) for gate in gates]


def initial_state_full(work_dim: int, clock_dim: int) -> np.ndarray:
    """Return |+>^n x |0> as shape (work_dim, clock_dim)."""
    psi0 = np.zeros((work_dim, clock_dim), dtype=complex)
    psi0[:, 0] = 1.0 / math.sqrt(work_dim)
    return psi0


def success_probability_full(
    psi: np.ndarray, x_star_index: int, final_clock: int
) -> float:
    """Return normalized full-space success probability."""
    arr = np.asarray(psi)
    if arr.ndim == 1:
        clock_dim = final_clock + 1
        if arr.size % clock_dim != 0:
            raise ValueError("flat state size is incompatible with final_clock")
        arr = arr.reshape(arr.size // clock_dim, clock_dim)
    norm = float(np.sum(np.abs(arr) ** 2))
    if norm <= 0.0 or not np.isfinite(norm):
        raise RuntimeError("state norm is not finite and positive")
    return float(abs(arr[x_star_index, final_clock]) ** 2 / norm)


def build_grover_unitary(work_dim: int, x_star_index: int) -> np.ndarray:
    """Build U_G = D O for a unique marked state."""
    plus = np.full(work_dim, 1.0 / math.sqrt(work_dim), dtype=complex)
    diffusion = 2.0 * np.outer(plus, plus.conj()) - np.eye(work_dim, dtype=complex)
    oracle = np.eye(work_dim, dtype=complex)
    oracle[x_star_index, x_star_index] = -1.0
    return diffusion @ oracle


def grover_success_formula(n: int, L: int) -> float:
    """Unique-solution Grover success formula."""
    alpha = np.arcsin(2.0 ** (-0.5 * n))
    return float(np.sin((2 * L + 1) * alpha) ** 2)


def build_hx_plus_projector(n: int):
    """Build H_X = 1/2 sum_i (I - X_i) on work space."""
    work_dim = 1 << n
    rows = []
    cols = []
    data = []
    diag = np.full(work_dim, 0.5 * n, dtype=float)
    for x in range(work_dim):
        rows.append(x)
        cols.append(x)
        data.append(diag[x])
        for i in range(n):
            rows.append(x ^ (1 << i))
            cols.append(x)
            data.append(-0.5)
    return csr_matrix((data, (rows, cols)), shape=(work_dim, work_dim))


def random_complex_matrix(dim: int, rng, scale: str = "ginibre") -> np.ndarray:
    """Return complex Ginibre matrix (A+iB)/sqrt(2)."""
    if scale != "ginibre":
        raise ValueError("only scale='ginibre' is supported")
    return (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) / math.sqrt(2.0)


def estimate_spectral_norm(
    A: np.ndarray, method: str = "power", n_iter: int = 20, rng=None
) -> float:
    """Estimate or exactly compute matrix 2-norm."""
    if method == "exact":
        return float(np.linalg.norm(A, 2))
    if method != "power":
        raise ValueError("method must be 'exact' or 'power'")
    if rng is None:
        rng = np.random.default_rng(1234)
    x = rng.standard_normal(A.shape[1]) + 1j * rng.standard_normal(A.shape[1])
    x = x / np.linalg.norm(x)
    for _ in range(n_iter):
        y = A.conj().T @ (A @ x)
        norm_y = np.linalg.norm(y)
        if norm_y == 0.0:
            return 0.0
        x = y / norm_y
    ax = A @ x
    return float(np.linalg.norm(ax))


def normalize_random_perturbation(
    R: np.ndarray,
    epsilon: float,
    norm_method: str = "power",
    norm_iters: int = 20,
) -> tuple[np.ndarray, float]:
    """Scale random perturbation to approximate/exact spectral norm epsilon."""
    norm_R = estimate_spectral_norm(R, method=norm_method, n_iter=norm_iters)
    if norm_R <= 0.0 or not np.isfinite(norm_R):
        raise RuntimeError("random perturbation norm is not finite and positive")
    return epsilon * R / norm_R, norm_R


def apply_full_fk_propagation_hamiltonian(
    psi: np.ndarray, V_diag: list[np.ndarray]
) -> np.ndarray:
    """Apply full non-Hermitian FK propagation Hamiltonian."""
    work_dim, clock_dim = psi.shape
    L = clock_dim - 1
    if len(V_diag) != L:
        raise ValueError("V_diag length must be clock_dim - 1")
    out = np.zeros_like(psi, dtype=complex)
    for idx, v in enumerate(V_diag):
        left = idx
        right = idx + 1
        out[:, left] += 0.5 * psi[:, left]
        out[:, right] += 0.5 * psi[:, right]
        out[:, right] += -0.5 * v * psi[:, left]
        out[:, left] += -0.5 * (1.0 / v) * psi[:, right]
    return out


def apply_full_fk_path_hamiltonian(
    psi: np.ndarray,
    s: float,
    V_diag: list[np.ndarray],
    include_hx: bool = False,
    hx=None,
) -> np.ndarray:
    """Apply the full FK interpolation Hamiltonian.

    With ``include_hx=True`` this applies
    H_X x |0><0| + (1-s) I_w x sum_{l>0}|l><l| + s H_FK.
    The default keeps the previous clock-only path for diagnostics and
    experiment-one validation code that intentionally omits H_X.
    """
    work_dim, clock_dim = psi.shape
    init_diag = np.ones(clock_dim, dtype=float)
    init_diag[0] = 0.0
    out = (1.0 - s) * psi * init_diag[None, :] + s * apply_full_fk_propagation_hamiltonian(psi, V_diag)
    if include_hx:
        if hx is None:
            n = int(round(math.log2(work_dim)))
            if (1 << n) != work_dim:
                raise ValueError("work_dim must be a power of two when hx is not provided")
            hx = build_hx_plus_projector(n)
        out[:, 0] += hx @ psi[:, 0]
    return out


def apply_full_hd_active_link_hamiltonian(
    psi: np.ndarray,
    l: int,
    theta: float,
    gate_type: str,
    Omega: float = 1.0,
    v: np.ndarray | None = None,
    U: np.ndarray | None = None,
) -> np.ndarray:
    """Apply full-clock active-link HD Hamiltonian with inactive Omega I."""
    work_dim, clock_dim = psi.shape
    if l <= 0 or l >= clock_dim:
        raise ValueError("l must be in 1..clock_dim-1")
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    out = Omega * psi.astype(complex, copy=True)
    left = l - 1
    right = l
    a = psi[:, left]
    b = psi[:, right]
    out[:, left] = 0.0
    out[:, right] = 0.0
    if gate_type == "NHMIS":
        if v is None:
            raise ValueError("v is required for NHMIS active link")
        out[:, left] = Omega * (sin_theta * sin_theta * a - sin_theta * cos_theta * b / v)
        out[:, right] = Omega * (-sin_theta * cos_theta * v * a + cos_theta * cos_theta * b)
    elif gate_type == "HMIS":
        if U is None:
            raise ValueError("U is required for HMIS active link")
        out[:, left] = Omega * (sin_theta * sin_theta * a - sin_theta * cos_theta * (U.conj().T @ b))
        out[:, right] = Omega * (-sin_theta * cos_theta * (U @ a) + cos_theta * cos_theta * b)
    else:
        raise ValueError("gate_type must be 'NHMIS' or 'HMIS'")
    return out


def dense_matrix_from_matvec(matvec: Callable[[np.ndarray], np.ndarray], dim: int) -> np.ndarray:
    """Build dense matrix from a matvec; intended only for m=2 validation/exp2."""
    matrix = np.empty((dim, dim), dtype=complex)
    for j in range(dim):
        basis = np.zeros(dim, dtype=complex)
        basis[j] = 1.0
        matrix[:, j] = matvec(basis)
    return matrix
