"""Experiment one on fixed CK-like random cross-edge deletion graph pools."""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

try:
    from .experiment_one_ck import (
        Gate,
        _fk_expm_num_steps,
        ck_graph_info,
        compute_prefix_weights_ck,
        default_max_workers,
        solve_empty_fk_clock,
        solve_empty_hd_segment,
        success_hmis_hd,
        success_nhmis_fk,
        success_nhmis_hd,
    )
    from .generate_ck_like_random_deletion_graphs import (
        GRAPH_POOL_FILENAME,
        ck_cross_edges,
        graph_hash,
        load_graph_pool,
        validate_graph_pool,
        validate_graph_record,
    )
except ImportError:  # pragma: no cover - used when run as a script path.
    from experiment_one_ck import (
        Gate,
        _fk_expm_num_steps,
        ck_graph_info,
        compute_prefix_weights_ck,
        default_max_workers,
        solve_empty_fk_clock,
        solve_empty_hd_segment,
        success_hmis_hd,
        success_nhmis_fk,
        success_nhmis_hd,
    )
    from generate_ck_like_random_deletion_graphs import (
        GRAPH_POOL_FILENAME,
        ck_cross_edges,
        graph_hash,
        load_graph_pool,
        validate_graph_pool,
        validate_graph_record,
    )


ALGORITHMS = ["NHMIS-FKQAA", "NHMIS-HDQAA", "HMIS-HDQAA"]
RAW_COLUMNS = [
    "algorithm",
    "graph_id",
    "graph_hash",
    "m",
    "n",
    "sample_id",
    "seed",
    "delete_probability",
    "deleted_edge_count",
    "num_edges",
    "r",
    "L",
    "x_star_index",
    "mis_size",
    "x_star_left_count",
    "x_star_right_count",
    "gamma",
    "Omega",
    "p_gate",
    "q_gate",
    "absf2",
    "absg2",
    "success_probability",
    "ideal_NH_circuit",
    "fk_method",
    "fk_num_steps",
    "fk_norm_error",
    "runtime_prepare",
    "runtime_solver",
    "runtime_total",
    "cache_hit_prefix",
    "cache_hit_clock",
    "status",
    "error_message",
]
SUMMARY_COLUMNS = [
    "algorithm",
    "m",
    "n",
    "gamma",
    "n_samples",
    "median_L",
    "min_L",
    "max_L",
    "median_deleted_edge_count",
    "min_deleted_edge_count",
    "max_deleted_edge_count",
    "median_success",
    "min_success",
    "max_success",
]


DeletedCrossEdge = tuple[int, int, int]


def _threadpool_limit_context():
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        return nullcontext()
    return threadpool_limits(limits=1)


def _float_token(value: float | int | None) -> str:
    if value is None:
        return "none"
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return str(value).replace("-", "m").replace(".", "p")


def _normalize_deleted_edges(m: int, deleted_edges) -> tuple[DeletedCrossEdge, ...]:
    valid = set(ck_cross_edges(m))
    normalized = tuple(sorted(tuple(int(value) for value in edge) for edge in deleted_edges))
    if len(set(normalized)) != len(normalized):
        raise ValueError("deleted_edges contains duplicates")
    invalid = [edge for edge in normalized if edge not in valid]
    if invalid:
        raise ValueError(f"invalid deleted CK cross edges: {invalid[:3]}")
    return normalized


def build_ck_like_random_deletion_gate_sequence(
    m: int, r: int, deleted_edges
) -> list[Gate]:
    """Build the experiment-one gate sequence for one CK-like deletion graph."""
    if r < 0:
        raise ValueError("r must be nonnegative")
    deleted = set(_normalize_deleted_edges(m, deleted_edges))
    n = ck_graph_info(m)["n"]
    one_round: list[Gate] = []

    for left in range(m):
        for tri in range(m - 1):
            for u in range(3):
                if (left, tri, u) not in deleted:
                    one_round.append(("B", left, m + 3 * tri + u))

    for tri in range(m - 1):
        v0 = m + 3 * tri
        v1 = v0 + 1
        v2 = v0 + 2
        one_round.append(("B", v0, v1))
        one_round.append(("B", v0, v2))
        one_round.append(("B", v1, v2))

    for vertex in range(n):
        one_round.append(("A", vertex))

    return one_round * r


def ck_like_random_edges_from_gates(gates) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for gate in gates:
        if gate[0] == "B":
            i, j = int(gate[1]), int(gate[2])
            edges.add((min(i, j), max(i, j)))
    return edges


def compute_logw_star_for_bitstring(
    n: int,
    gates: list[Gate],
    x_star_bits: list[int] | tuple[int, ...],
    p: float = 2.0,
    q: float = 4.0,
) -> np.ndarray:
    """Compute ``log w_l(x_star)`` for an arbitrary target bitstring."""
    if len(x_star_bits) != n:
        raise ValueError("x_star_bits length must equal n")
    if p <= 0.0 or q <= 0.0:
        raise ValueError("p and q must be positive")

    bits = tuple(int(bit) for bit in x_star_bits)
    if any(bit not in {0, 1} for bit in bits):
        raise ValueError("x_star_bits must be binary")

    log_p = math.log(p)
    log_q = math.log(q)
    current = -n * math.log(2.0)
    logw_star = np.empty(len(gates) + 1, dtype=float)
    logw_star[0] = current

    for step, gate in enumerate(gates, start=1):
        if not gate:
            raise ValueError("empty gate")
        if gate[0] == "A":
            if len(gate) != 2:
                raise ValueError(f"invalid A gate {gate}")
            vertex = int(gate[1])
            if vertex < 0 or vertex >= n:
                raise ValueError(f"invalid A gate vertex {vertex}")
            if bits[vertex]:
                current += 2.0 * log_p
        elif gate[0] == "B":
            if len(gate) != 3:
                raise ValueError(f"invalid B gate {gate}")
            i, j = int(gate[1]), int(gate[2])
            if i < 0 or i >= n or j < 0 or j >= n:
                raise ValueError(f"invalid B gate endpoints {i}, {j}")
            if not (bits[i] and bits[j]):
                current += 2.0 * log_q
        else:
            raise ValueError(f"unknown gate kind {gate[0]}")
        logw_star[step] = current

    return logw_star


def compute_prefix_weights_ck_like_target(
    m: int,
    gates: list[Gate],
    x_star_bits: list[int] | tuple[int, ...],
    p: float = 2.0,
    q: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute CK-like prefix ``logZ`` plus target-specific ``logw_star``."""
    logZ, _old_logw = compute_prefix_weights_ck(m, gates, p=p, q=q)
    n = ck_graph_info(m)["n"]
    logw_star = compute_logw_star_for_bitstring(n, gates, x_star_bits, p=p, q=q)
    return logZ, logw_star


def _prefix_cache_path(cache_dir: str, record: dict, p_gate: float, q_gate: float) -> str:
    filename = (
        f"prefix_m{int(record['m'])}_s{int(record['sample_id']):02d}_"
        f"{record['graph_hash']}_p{_float_token(p_gate)}_q{_float_token(q_gate)}.npz"
    )
    return os.path.join(cache_dir, "prefix", filename)


def _clock_cache_path(
    cache_dir: str,
    L: int,
    gamma: float,
    fk_method: str,
    fk_num_steps: int | None,
    fk_step_factor: int,
    fk_min_steps: int,
) -> str:
    filename = (
        f"clock_L{L}_gamma{_float_token(gamma)}_{fk_method}"
        f"_num{_float_token(fk_num_steps)}_factor{fk_step_factor}_min{fk_min_steps}.npz"
    )
    return os.path.join(cache_dir, "clock", filename)


def _load_prefix_cache(filename: str) -> dict:
    with np.load(filename) as data:
        return {
            "graph_id": str(data["graph_id"]),
            "graph_hash": str(data["graph_hash"]),
            "m": int(data["m"]),
            "n": int(data["n"]),
            "sample_id": int(data["sample_id"]),
            "seed": int(data["seed"]),
            "delete_probability": float(data["delete_probability"]),
            "deleted_edge_count": int(data["deleted_edge_count"]),
            "num_edges": int(data["num_edges"]),
            "r": int(data["r"]),
            "L": int(data["L"]),
            "x_star_index": int(data["x_star_index"]),
            "mis_size": int(data["mis_size"]),
            "x_star_left_count": int(data["x_star_left_count"]),
            "x_star_right_count": int(data["x_star_right_count"]),
            "logZ": data["logZ"].copy(),
            "logw_star": data["logw_star"].copy(),
            "ideal_NH_circuit": float(data["ideal_NH_circuit"]),
            "cache_hit_prefix": True,
        }


def _save_prefix_cache(instance: dict, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez_compressed(
        filename,
        graph_id=instance["graph_id"],
        graph_hash=instance["graph_hash"],
        m=instance["m"],
        n=instance["n"],
        sample_id=instance["sample_id"],
        seed=instance["seed"],
        delete_probability=instance["delete_probability"],
        deleted_edge_count=instance["deleted_edge_count"],
        num_edges=instance["num_edges"],
        r=instance["r"],
        L=instance["L"],
        x_star_index=instance["x_star_index"],
        mis_size=instance["mis_size"],
        x_star_left_count=instance["x_star_left_count"],
        x_star_right_count=instance["x_star_right_count"],
        logZ=instance["logZ"],
        logw_star=instance["logw_star"],
        ideal_NH_circuit=instance["ideal_NH_circuit"],
    )


def _prob_from_logs(log_num: float, log_den: float) -> float:
    if np.isneginf(log_num) or np.isneginf(log_den):
        return 0.0
    return float(np.clip(np.exp(log_num - log_den), 0.0, 1.0))


def prepare_ck_like_random_instance(
    record: dict,
    p_gate: float = 2.0,
    q_gate: float = 4.0,
    cache_dir: str | None = None,
    force: bool = False,
) -> dict:
    """Prepare one graph-pool record and cache its prefix weights."""
    validate_graph_record(record)
    cache_file = None
    if cache_dir is not None:
        cache_file = _prefix_cache_path(cache_dir, record, p_gate, q_gate)
        if os.path.exists(cache_file) and not force:
            return _load_prefix_cache(cache_file)

    m = int(record["m"])
    n = int(record["n"])
    r = n
    deleted = _normalize_deleted_edges(m, record["deleted_cross_edges"])
    if graph_hash(m, deleted) != record["graph_hash"]:
        raise ValueError(f"graph hash changed for {record['graph_id']}")
    gates = build_ck_like_random_deletion_gate_sequence(m, r, deleted)
    if len(gates) != int(record["L"]):
        raise RuntimeError(f"internal L mismatch: got {len(gates)}, expected {record['L']}")

    logZ, logw_star = compute_prefix_weights_ck_like_target(
        m, gates, record["x_star_bits"], p=p_gate, q=q_gate
    )
    ideal = _prob_from_logs(float(logw_star[-1]), float(logZ[-1]))
    instance = {
        "graph_id": record["graph_id"],
        "graph_hash": record["graph_hash"],
        "m": m,
        "n": n,
        "sample_id": int(record["sample_id"]),
        "seed": int(record["seed"]),
        "delete_probability": float(record["delete_probability"]),
        "deleted_edge_count": int(record["deleted_edge_count"]),
        "num_edges": int(record["num_edges"]),
        "r": r,
        "L": int(record["L"]),
        "x_star_index": int(record["x_star_index"]),
        "mis_size": int(record["mis_size"]),
        "x_star_left_count": int(record["x_star_left_count"]),
        "x_star_right_count": int(record["x_star_right_count"]),
        "logZ": logZ,
        "logw_star": logw_star,
        "ideal_NH_circuit": ideal,
        "cache_hit_prefix": False,
    }
    if cache_file is not None:
        _save_prefix_cache(instance, cache_file)
    return instance


def solve_fk_clock_cached(
    L: int,
    gamma: float,
    fk_method: str,
    fk_num_steps: int | None,
    fk_step_factor: int,
    fk_min_steps: int,
    cache_dir: str | None = None,
    force: bool = False,
) -> tuple[np.ndarray, bool]:
    cache_file = None
    if cache_dir is not None:
        cache_file = _clock_cache_path(
            cache_dir, L, gamma, fk_method, fk_num_steps, fk_step_factor, fk_min_steps
        )
        if os.path.exists(cache_file) and not force:
            with np.load(cache_file) as data:
                return data["c"].copy(), True

    c = solve_empty_fk_clock(
        L,
        gamma,
        method=fk_method,
        num_steps=fk_num_steps,
        step_factor=fk_step_factor,
        min_steps=fk_min_steps,
    )
    if cache_file is not None:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        np.savez_compressed(cache_file, c=c)
    return c, False


def expected_algorithms_for_m(m: int, m_fk_max: int, m_hd_max: int) -> list[str]:
    algorithms: list[str] = []
    if m <= m_fk_max:
        algorithms.append("NHMIS-FKQAA")
    if m <= m_hd_max:
        algorithms.extend(["NHMIS-HDQAA", "HMIS-HDQAA"])
    return algorithms


def _actual_fk_num_steps(
    L: int,
    fk_method: str,
    fk_num_steps: int | None,
    fk_step_factor: int,
    fk_min_steps: int,
) -> float:
    if fk_method != "expm":
        return np.nan
    return float(_fk_expm_num_steps(L, fk_num_steps, fk_step_factor, fk_min_steps))


def _clock_worker(args: tuple) -> tuple[int, bool, float]:
    (
        L,
        gamma,
        fk_method,
        fk_num_steps,
        fk_step_factor,
        fk_min_steps,
        cache_dir,
    ) = args
    with _threadpool_limit_context():
        start = time.perf_counter()
        _c, cache_hit = solve_fk_clock_cached(
            L,
            gamma,
            fk_method,
            fk_num_steps,
            fk_step_factor,
            fk_min_steps,
            cache_dir=cache_dir,
            force=False,
        )
        return int(L), bool(cache_hit), time.perf_counter() - start


def precompute_fk_clock_caches(
    records: list[dict],
    m_fk_max: int,
    gamma: float,
    fk_method: str,
    fk_num_steps: int | None,
    fk_step_factor: int,
    fk_min_steps: int,
    cache_dir: str,
    max_workers: int,
) -> None:
    unique_L = sorted({int(record["L"]) for record in records if int(record["m"]) <= m_fk_max})
    if not unique_L:
        return
    tasks = [
        (L, gamma, fk_method, fk_num_steps, fk_step_factor, fk_min_steps, cache_dir)
        for L in unique_L
    ]
    print(f"precomputing FK clocks for {len(tasks)} unique L values", flush=True)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_clock_worker, task) for task in tasks]
        for idx, future in enumerate(as_completed(futures), start=1):
            L, cache_hit, elapsed = future.result()
            state = "cache hit" if cache_hit else "computed"
            print(f"FK clock {state} {idx}/{len(tasks)}: L={L}, elapsed={elapsed:.3f}s", flush=True)


def run_one_ck_like_random_sample(args: tuple) -> list[dict]:
    (
        record,
        gamma,
        p_gate,
        q_gate,
        Omega,
        m_fk_max,
        m_hd_max,
        fk_method,
        fk_num_steps,
        fk_step_factor,
        fk_min_steps,
        cache_dir,
        force_prefix,
    ) = args

    start_total = time.perf_counter()
    with _threadpool_limit_context():
        try:
            start_prepare = time.perf_counter()
            instance = prepare_ck_like_random_instance(
                record,
                p_gate=p_gate,
                q_gate=q_gate,
                cache_dir=cache_dir,
                force=force_prefix,
            )
            runtime_prepare = time.perf_counter() - start_prepare
            f, g = solve_empty_hd_segment(gamma, Omega)
            absf2 = abs(f) ** 2
            absg2 = abs(g) ** 2
            base = {
                "graph_id": instance["graph_id"],
                "graph_hash": instance["graph_hash"],
                "m": instance["m"],
                "n": instance["n"],
                "sample_id": instance["sample_id"],
                "seed": instance["seed"],
                "delete_probability": instance["delete_probability"],
                "deleted_edge_count": instance["deleted_edge_count"],
                "num_edges": instance["num_edges"],
                "r": instance["r"],
                "L": instance["L"],
                "x_star_index": instance["x_star_index"],
                "mis_size": instance["mis_size"],
                "x_star_left_count": instance["x_star_left_count"],
                "x_star_right_count": instance["x_star_right_count"],
                "gamma": gamma,
                "Omega": Omega,
                "p_gate": p_gate,
                "q_gate": q_gate,
                "absf2": absf2,
                "absg2": absg2,
                "ideal_NH_circuit": instance["ideal_NH_circuit"],
                "fk_method": fk_method,
                "fk_num_steps": np.nan,
                "fk_norm_error": np.nan,
                "runtime_prepare": runtime_prepare,
                "runtime_solver": 0.0,
                "runtime_total": 0.0,
                "cache_hit_prefix": bool(instance["cache_hit_prefix"]),
                "cache_hit_clock": False,
                "status": "success",
                "error_message": "",
            }

            rows: list[dict] = []
            if instance["m"] <= m_fk_max:
                start = time.perf_counter()
                c, cache_hit_clock = solve_fk_clock_cached(
                    instance["L"],
                    gamma,
                    fk_method,
                    fk_num_steps,
                    fk_step_factor,
                    fk_min_steps,
                    cache_dir=cache_dir,
                    force=False,
                )
                runtime = time.perf_counter() - start
                row = {
                    **base,
                    "algorithm": "NHMIS-FKQAA",
                    "success_probability": success_nhmis_fk(
                        c, instance["logZ"], instance["logw_star"]
                    ),
                    "fk_num_steps": _actual_fk_num_steps(
                        instance["L"], fk_method, fk_num_steps, fk_step_factor, fk_min_steps
                    ),
                    "fk_norm_error": float(abs(np.sum(np.abs(c) ** 2) - 1.0)),
                    "runtime_solver": runtime,
                    "cache_hit_clock": cache_hit_clock,
                }
                row["runtime_total"] = time.perf_counter() - start_total
                rows.append(row)

            if instance["m"] <= m_hd_max:
                start = time.perf_counter()
                p_nhhd = success_nhmis_hd(f, g, instance["logZ"], instance["logw_star"])
                runtime = time.perf_counter() - start
                row = {
                    **base,
                    "algorithm": "NHMIS-HDQAA",
                    "success_probability": p_nhhd,
                    "runtime_solver": runtime,
                }
                row["runtime_total"] = time.perf_counter() - start_total
                rows.append(row)

                start = time.perf_counter()
                p_hhd = success_hmis_hd(f, instance["n"], instance["L"])
                runtime = time.perf_counter() - start
                row = {
                    **base,
                    "algorithm": "HMIS-HDQAA",
                    "success_probability": p_hhd,
                    "runtime_solver": runtime,
                }
                row["runtime_total"] = time.perf_counter() - start_total
                rows.append(row)

            return [{column: row.get(column, np.nan) for column in RAW_COLUMNS} for row in rows]
        except Exception as exc:
            row = {
                "graph_id": record.get("graph_id", ""),
                "graph_hash": record.get("graph_hash", ""),
                "m": record.get("m", np.nan),
                "n": record.get("n", np.nan),
                "sample_id": record.get("sample_id", np.nan),
                "seed": record.get("seed", np.nan),
                "delete_probability": record.get("delete_probability", np.nan),
                "deleted_edge_count": record.get("deleted_edge_count", np.nan),
                "num_edges": record.get("num_edges", np.nan),
                "r": record.get("n", np.nan),
                "L": record.get("L", np.nan),
                "x_star_index": record.get("x_star_index", np.nan),
                "mis_size": record.get("mis_size", np.nan),
                "x_star_left_count": record.get("x_star_left_count", np.nan),
                "x_star_right_count": record.get("x_star_right_count", np.nan),
                "gamma": gamma,
                "Omega": Omega,
                "p_gate": p_gate,
                "q_gate": q_gate,
                "runtime_total": time.perf_counter() - start_total,
                "status": "failed",
                "error_message": repr(exc),
            }
            return [
                {**{column: np.nan for column in RAW_COLUMNS}, **row, "algorithm": algorithm}
                for algorithm in expected_algorithms_for_m(
                    int(record.get("m", 0)), m_fk_max, m_hd_max
                )
            ]


def summarize_success(rows: pd.DataFrame | list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    df = df[df["status"] == "success"].copy()
    df = df[np.isfinite(df["success_probability"].to_numpy(dtype=float))]
    if df.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    output: list[dict] = []
    for (algorithm, m), group in df.groupby(["algorithm", "m"], sort=False):
        values = group["success_probability"].to_numpy(dtype=float)
        L_values = group["L"].to_numpy(dtype=float)
        deleted_values = group["deleted_edge_count"].to_numpy(dtype=float)
        output.append(
            {
                "algorithm": algorithm,
                "m": int(m),
                "n": int(group["n"].iloc[0]),
                "gamma": float(group["gamma"].iloc[0]),
                "n_samples": int(len(values)),
                "median_L": float(np.median(L_values)),
                "min_L": int(np.min(L_values)),
                "max_L": int(np.max(L_values)),
                "median_deleted_edge_count": float(np.median(deleted_values)),
                "min_deleted_edge_count": int(np.min(deleted_values)),
                "max_deleted_edge_count": int(np.max(deleted_values)),
                "median_success": float(np.median(values)),
                "min_success": float(np.min(values)),
                "max_success": float(np.max(values)),
            }
        )
    return pd.DataFrame(output, columns=SUMMARY_COLUMNS).sort_values(["algorithm", "m"])


def plot_ck_like_random_summary(
    summary_csv: str,
    out_dir: str,
    png_filename: str | None = None,
    pdf_filename: str | None = None,
):
    import matplotlib.pyplot as plt

    df = pd.read_csv(summary_csv)
    os.makedirs(out_dir, exist_ok=True)
    png_path = png_filename or os.path.join(out_dir, "ck_like_random_deletion_success.png")
    pdf_path = pdf_filename or os.path.join(out_dir, "ck_like_random_deletion_success.pdf")
    markers = {"NHMIS-FKQAA": "o", "NHMIS-HDQAA": "s", "HMIS-HDQAA": "^"}

    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    positive_seen = False
    for algorithm in ALGORITHMS:
        sub = df[df["algorithm"] == algorithm].sort_values("n")
        if sub.empty:
            continue
        x = sub["n"].to_numpy(dtype=int)
        median = sub["median_success"].to_numpy(dtype=float)
        ymin = sub["min_success"].to_numpy(dtype=float)
        ymax = sub["max_success"].to_numpy(dtype=float)
        median_plot = np.where(median > 0.0, median, np.nan)
        ymin_plot = np.where(ymin > 0.0, ymin, np.nan)
        ymax_plot = np.where(ymax > 0.0, ymax, np.nan)
        if not np.any(np.isfinite(median_plot)):
            continue
        positive_seen = True
        ax.plot(
            x,
            median_plot,
            marker=markers.get(algorithm, "o"),
            linewidth=1.8,
            label=algorithm,
        )
        ax.fill_between(x, ymin_plot, ymax_plot, alpha=0.18)

    if not positive_seen:
        ax.plot([1], [1.0], alpha=0.0)
    ax.set_xlabel(r"graph size $n$")
    ax.set_ylabel("success probability")
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", linewidth=0.7)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    return fig, png_path, pdf_path


def _read_existing_raw(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=RAW_COLUMNS)
    df = pd.read_csv(path)
    for column in RAW_COLUMNS:
        if column not in df:
            df[column] = np.nan
    return df[RAW_COLUMNS]


def _completed_task_keys(df: pd.DataFrame, m_fk_max: int, m_hd_max: int) -> set[str]:
    completed: set[str] = set()
    if df.empty:
        return completed
    good = df[df["status"] == "success"]
    for graph_id, group in good.groupby("graph_id"):
        m = int(group["m"].iloc[0])
        expected = set(expected_algorithms_for_m(m, m_fk_max, m_hd_max))
        present = set(group["algorithm"].astype(str))
        if expected and expected.issubset(present):
            completed.add(str(graph_id))
    return completed


def load_and_filter_graph_pool(
    graph_pool: str,
    m_min: int,
    m_hd_max: int,
    num_samples: int,
) -> list[dict]:
    records = load_graph_pool(graph_pool)
    validate_graph_pool(records, m_min, m_hd_max, num_samples)
    filtered = [record for record in records if m_min <= int(record["m"]) <= m_hd_max]
    filtered.sort(key=lambda row: (int(row["m"]), int(row["sample_id"])))
    return filtered


def build_tasks(
    records: list[dict],
    gamma: float,
    p_gate: float,
    q_gate: float,
    Omega: float,
    m_fk_max: int,
    m_hd_max: int,
    fk_method: str,
    fk_num_steps: int | None,
    fk_step_factor: int,
    fk_min_steps: int,
    cache_dir: str,
    force_prefix: bool,
    completed: set[str] | None = None,
) -> list[tuple]:
    completed = completed or set()
    tasks = []
    for record in records:
        if str(record["graph_id"]) in completed:
            continue
        tasks.append(
            (
                record,
                gamma,
                p_gate,
                q_gate,
                Omega,
                m_fk_max,
                m_hd_max,
                fk_method,
                fk_num_steps,
                fk_step_factor,
                fk_min_steps,
                cache_dir,
                force_prefix,
            )
        )
    tasks.sort(key=lambda task: (int(task[0]["m"]), int(task[0]["L"])), reverse=True)
    return tasks


def run_ck_like_random_deletion_experiment(
    graph_pool: str = os.path.join(
        "outputs", "ck_like_random_deletion_graphs", GRAPH_POOL_FILENAME
    ),
    m_min: int = 4,
    m_hd_max: int = 12,
    m_fk_max: int = 10,
    num_samples: int = 20,
    gamma: float = 10.0,
    p_gate: float = 2.0,
    q_gate: float = 4.0,
    Omega: float = 1.0,
    out_dir: str = "outputs/ck_like_random_deletion_exp1_gamma10",
    fk_method: str = "expm",
    fk_num_steps: int | None = None,
    fk_step_factor: int = 8,
    fk_min_steps: int = 200,
    max_workers: int | None = None,
    reserve_cores: int = 1,
    serial: bool = False,
    force: bool = False,
    skip_fk_precompute: bool = False,
) -> tuple[str, str, str, str]:
    if m_min < 1 or m_hd_max < m_min:
        raise ValueError("invalid m range")
    if m_fk_max > m_hd_max:
        raise ValueError("m_fk_max must not exceed m_hd_max")
    if not os.path.exists(graph_pool):
        raise FileNotFoundError(f"graph pool not found: {graph_pool}")

    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "ck_like_results.csv")
    summary_path = os.path.join(out_dir, "ck_like_summary.csv")
    cache_dir = os.path.join(out_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    records = load_and_filter_graph_pool(graph_pool, m_min, m_hd_max, num_samples)
    raw = _read_existing_raw(raw_path)
    completed = _completed_task_keys(raw, m_fk_max, m_hd_max)
    if force:
        raw = pd.DataFrame(columns=RAW_COLUMNS)
        completed = set()

    tasks = build_tasks(
        records,
        gamma,
        p_gate,
        q_gate,
        Omega,
        m_fk_max,
        m_hd_max,
        fk_method,
        fk_num_steps,
        fk_step_factor,
        fk_min_steps,
        cache_dir,
        force_prefix=force,
        completed=completed,
    )

    workers = max_workers if max_workers is not None else default_max_workers(reserve_cores)
    if tasks:
        print(f"remaining graph tasks: {len(tasks)}")
        if not serial and not skip_fk_precompute:
            task_records = [task[0] for task in tasks if int(task[0]["m"]) <= m_fk_max]
            precompute_fk_clock_caches(
                task_records,
                m_fk_max,
                gamma,
                fk_method,
                fk_num_steps,
                fk_step_factor,
                fk_min_steps,
                cache_dir,
                max_workers=workers,
            )
    else:
        print("all graph tasks already complete; rebuilding summary and plot")

    def record_rows(new_rows: list[dict]) -> None:
        nonlocal raw
        new_df = pd.DataFrame(new_rows, columns=RAW_COLUMNS)
        raw = new_df if raw.empty else pd.concat([raw, new_df], ignore_index=True)
        raw = raw.drop_duplicates(
            subset=["algorithm", "graph_id"], keep="last"
        ).sort_values(["m", "sample_id", "algorithm"])
        raw.to_csv(raw_path, index=False)

    if serial:
        for idx, task in enumerate(tasks, start=1):
            rows = run_one_ck_like_random_sample(task)
            record_rows(rows)
            first = rows[0]
            print(
                f"finished {idx}/{len(tasks)}: m={first['m']}, "
                f"sample={first['sample_id']}, status={first['status']}",
                flush=True,
            )
    elif tasks:
        print(
            f"using ProcessPoolExecutor with max_workers = {workers}, "
            f"reserve_cores = {reserve_cores}"
        )
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(run_one_ck_like_random_sample, task) for task in tasks]
            for idx, future in enumerate(as_completed(futures), start=1):
                rows = future.result()
                record_rows(rows)
                first = rows[0]
                print(
                    f"finished {idx}/{len(tasks)}: m={first['m']}, "
                    f"sample={first['sample_id']}, status={first['status']}",
                    flush=True,
                )

    summary = summarize_success(raw)
    summary.to_csv(summary_path, index=False)
    fig, png_path, pdf_path = plot_ck_like_random_summary(summary_path, out_dir)
    fig.clf()
    print(f"saved {raw_path}")
    print(f"saved {summary_path}")
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")
    return raw_path, summary_path, png_path, pdf_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run experiment one on a fixed CK-like random deletion graph pool."
    )
    parser.add_argument(
        "--graph-pool",
        type=str,
        default=os.path.join("outputs", "ck_like_random_deletion_graphs", GRAPH_POOL_FILENAME),
    )
    parser.add_argument("--m-min", type=int, default=4)
    parser.add_argument("--m-hd-max", type=int, default=12)
    parser.add_argument("--m-fk-max", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--p", dest="p_gate", type=float, default=2.0)
    parser.add_argument("--q", dest="q_gate", type=float, default=4.0)
    parser.add_argument("--Omega", type=float, default=1.0)
    parser.add_argument(
        "--out-dir", type=str, default="outputs/ck_like_random_deletion_exp1_gamma10"
    )
    parser.add_argument("--fk-method", choices=["expm", "ivp"], default="expm")
    parser.add_argument("--fk-num-steps", type=int)
    parser.add_argument("--fk-step-factor", type=int, default=8)
    parser.add_argument("--fk-min-steps", type=int, default=200)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--reserve-cores", type=int, default=1)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-fk-precompute", action="store_true")
    return parser.parse_args()


def main() -> int:
    mp.freeze_support()
    args = parse_args()
    run_ck_like_random_deletion_experiment(
        graph_pool=args.graph_pool,
        m_min=args.m_min,
        m_hd_max=args.m_hd_max,
        m_fk_max=args.m_fk_max,
        num_samples=args.num_samples,
        gamma=args.gamma,
        p_gate=args.p_gate,
        q_gate=args.q_gate,
        Omega=args.Omega,
        out_dir=args.out_dir,
        fk_method=args.fk_method,
        fk_num_steps=args.fk_num_steps,
        fk_step_factor=args.fk_step_factor,
        fk_min_steps=args.fk_min_steps,
        max_workers=args.max_workers,
        reserve_cores=args.reserve_cores,
        serial=args.serial,
        force=args.force,
        skip_fk_precompute=args.skip_fk_precompute,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
