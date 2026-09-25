"""Resumable mixed-scaling runner with persistent per-task caches."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

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
        _fk_expm_num_steps,
        ck_graph_info,
        default_max_workers,
        prepare_ck_instance,
        solve_empty_fk_clock,
        solve_empty_hd_segment,
        success_hmis_hd,
        success_nhmis_fk,
        success_nhmis_hd,
    )
    from .plot_mixed_scaling_from_csv import plot_mixed_scaling_from_csv
except ImportError:  # pragma: no cover - used when run as a script path.
    from experiment_one_ck import (
        _fk_expm_num_steps,
        ck_graph_info,
        default_max_workers,
        prepare_ck_instance,
        solve_empty_fk_clock,
        solve_empty_hd_segment,
        success_hmis_hd,
        success_nhmis_fk,
        success_nhmis_hd,
    )
    from plot_mixed_scaling_from_csv import plot_mixed_scaling_from_csv


WIDE_COLUMNS = [
    "m",
    "n",
    "L",
    "gamma",
    "p_FK",
    "p_NHHD",
    "p_HHD",
    "p_Grover",
    "ideal_NH_circuit",
    "computed_FK",
    "computed_NHHD",
    "computed_HHD",
    "fk_norm_error",
    "runtime_FK",
    "runtime_NHHD",
    "runtime_HMIS",
    "runtime_prepare_instance",
    "note",
]

LONG_COLUMNS = [
    "algorithm",
    "m",
    "n",
    "L",
    "gamma",
    "success_probability",
    "computed",
    "runtime_total",
    "status",
    "note",
]


def _threadpool_limit_context():
    try:
        from threadpoolctl import threadpool_limits
    except Exception:
        return nullcontext()
    return threadpool_limits(limits=1)


def gamma_token(gamma: float) -> str:
    value = float(gamma)
    if value.is_integer():
        return str(int(value))
    return str(value).replace("-", "m").replace(".", "p")


def ensure_dirs(out_dir: str) -> tuple[str, str]:
    cache_dir = os.path.join(out_dir, "cache")
    tasks_dir = os.path.join(out_dir, "tasks")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(tasks_dir, exist_ok=True)
    return cache_dir, tasks_dir


def instance_cache_path(out_dir: str, m: int) -> str:
    return os.path.join(out_dir, "cache", f"ck_instance_m{m}.npz")


def prepare_task_json_path(out_dir: str, m: int, gamma: float) -> str:
    return os.path.join(out_dir, "tasks", f"PREPARE_m{m}_gamma{gamma_token(gamma)}.json")


def task_json_path(
    out_dir: str,
    algorithm: str,
    m: int,
    gamma: float,
    fk_method: str = "expm",
    fk_step_factor: int = 16,
) -> str:
    token = gamma_token(gamma)
    if algorithm == "FK":
        return os.path.join(
            out_dir,
            "tasks",
            f"FK_m{m}_gamma{token}_{fk_method}_sf{fk_step_factor}.json",
        )
    return os.path.join(out_dir, "tasks", f"{algorithm}_m{m}_gamma{token}.json")


def save_instance_cache(instance: dict, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savez_compressed(
        filename,
        m=instance["m"],
        n=instance["n"],
        r=instance["r"],
        L=instance["L"],
        logZ=instance["logZ"],
        logw_star=instance["logw_star"],
        ideal_NH_circuit=instance["ideal_NH_circuit"],
    )


def load_instance_cache(filename: str) -> dict:
    with np.load(filename) as data:
        return {
            "m": int(data["m"]),
            "n": int(data["n"]),
            "r": int(data["r"]),
            "L": int(data["L"]),
            "logZ": data["logZ"].copy(),
            "logw_star": data["logw_star"].copy(),
            "ideal_NH_circuit": float(data["ideal_NH_circuit"]),
        }


def save_task_json(result: dict, filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)


def load_task_json(filename: str) -> dict:
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def cached_success(filename: str, force: bool = False) -> dict | None:
    if force or not os.path.exists(filename):
        return None
    row = load_task_json(filename)
    if row.get("status") == "success":
        return row
    return None


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _pure_grover_success(n: int, L: int) -> float:
    alpha = np.arcsin(2.0 ** (-0.5 * n))
    return float(np.sin((2 * L + 1) * alpha) ** 2)


def build_dynamic_tasks(
    m_start: int,
    m_fk_max: int,
    m_nhhd_max: int,
    hmis_same_as_nhhd: bool = True,
    m_hmis_max: int | None = None,
) -> list[tuple[str, int]]:
    """Return fine-grained (algorithm_kind, m) tasks in heavy-first order."""
    hmis_max = m_nhhd_max if hmis_same_as_nhhd else (m_hmis_max or m_nhhd_max)
    tasks: list[tuple[str, int]] = []
    tasks.extend(("FK", m) for m in range(m_fk_max, m_start - 1, -1))
    tasks.extend(("NHHD", m) for m in range(m_nhhd_max, m_start - 1, -1))
    tasks.extend(("HMIS", m) for m in range(hmis_max, m_start - 1, -1))
    return tasks


def _success_result_base(algorithm: str, m: int, n: int, L: int, gamma: float) -> dict:
    return {
        "algorithm": algorithm,
        "m": m,
        "n": n,
        "L": L,
        "gamma": gamma,
        "success_probability": None,
        "p_Grover": None,
        "ideal_NH_circuit": None,
        "fk_method": None,
        "fk_step_factor": None,
        "fk_num_steps": None,
        "fk_norm_error": None,
        "runtime_prepare_instance": 0.0,
        "runtime_solver": 0.0,
        "runtime_total": 0.0,
        "status": "failed",
        "error_message": "",
        "timestamp": now_iso(),
    }


def _prepare_worker(args: tuple) -> dict:
    m, gamma, p, q, out_dir, force = args
    path = instance_cache_path(out_dir, m)
    task_path = prepare_task_json_path(out_dir, m, gamma)
    cached = cached_success(task_path, force=force)
    if cached is not None and os.path.exists(path):
        cached["cache_hit"] = True
        return cached

    start = time.perf_counter()
    result = _success_result_base("PREPARE", m, 0, 0, gamma)
    result["cache_hit"] = False
    with _threadpool_limit_context():
        try:
            instance = prepare_ck_instance(m, p=p, q=q)
            save_instance_cache(instance, path)
            elapsed = time.perf_counter() - start
            result.update(
                {
                    "n": instance["n"],
                    "L": instance["L"],
                    "ideal_NH_circuit": instance["ideal_NH_circuit"],
                    "runtime_prepare_instance": elapsed,
                    "runtime_total": elapsed,
                    "status": "success",
                }
            )
        except Exception as exc:
            result.update(
                {
                    "runtime_total": time.perf_counter() - start,
                    "status": "failed",
                    "error_message": repr(exc),
                }
            )
    save_task_json(result, task_path)
    return result


def _algorithm_worker(args: tuple) -> dict:
    (
        algorithm,
        m,
        gamma,
        Omega,
        out_dir,
        fk_method,
        fk_num_steps,
        fk_step_factor,
        fk_min_steps,
        force,
    ) = args
    task_path = task_json_path(out_dir, algorithm, m, gamma, fk_method, fk_step_factor)
    cached = cached_success(task_path, force=force)
    if cached is not None:
        cached["cache_hit"] = True
        return cached

    info = ck_graph_info(m)
    n = info["n"]
    L = n * (n + info["num_edges"])
    result = _success_result_base(algorithm, m, n, L, gamma)
    result["cache_hit"] = False
    start_total = time.perf_counter()

    with _threadpool_limit_context():
        try:
            start_solver = time.perf_counter()
            f, g = solve_empty_hd_segment(gamma, Omega)

            if algorithm == "HMIS":
                p_grover = _pure_grover_success(n, L)
                success = success_hmis_hd(f, n, L)
                result.update(
                    {
                        "success_probability": success,
                        "p_Grover": p_grover,
                        "runtime_solver": time.perf_counter() - start_solver,
                        "runtime_total": time.perf_counter() - start_total,
                        "status": "success",
                    }
                )
            else:
                instance = load_instance_cache(instance_cache_path(out_dir, m))
                result["ideal_NH_circuit"] = instance["ideal_NH_circuit"]
                if algorithm == "NHHD":
                    success = success_nhmis_hd(
                        f, g, instance["logZ"], instance["logw_star"]
                    )
                    result.update(
                        {
                            "success_probability": success,
                            "runtime_solver": time.perf_counter() - start_solver,
                            "runtime_total": time.perf_counter() - start_total,
                            "status": "success",
                        }
                    )
                elif algorithm == "FK":
                    print(
                        f"FK task start: m={m}, n={instance['n']}, L={instance['L']}, "
                        f"fk_num_steps={_fk_expm_num_steps(instance['L'], fk_num_steps, fk_step_factor, fk_min_steps) if fk_method == 'expm' else 'ivp'}, "
                        f"time={now_iso()}",
                        flush=True,
                    )
                    c = solve_empty_fk_clock(
                        instance["L"],
                        gamma,
                        method=fk_method,
                        num_steps=fk_num_steps,
                        step_factor=fk_step_factor,
                        min_steps=fk_min_steps,
                    )
                    success = success_nhmis_fk(c, instance["logZ"], instance["logw_star"])
                    result.update(
                        {
                            "success_probability": success,
                            "fk_method": fk_method,
                            "fk_step_factor": fk_step_factor,
                            "fk_num_steps": (
                                _fk_expm_num_steps(
                                    instance["L"], fk_num_steps, fk_step_factor, fk_min_steps
                                )
                                if fk_method == "expm"
                                else None
                            ),
                            "fk_norm_error": float(abs(np.sum(np.abs(c) ** 2) - 1.0)),
                            "runtime_solver": time.perf_counter() - start_solver,
                            "runtime_total": time.perf_counter() - start_total,
                            "status": "success",
                        }
                    )
                    print(
                        f"FK task finished: m={m}, elapsed={result['runtime_solver']:.3f}s",
                        flush=True,
                    )
                else:
                    raise ValueError(f"unknown algorithm {algorithm}")
        except Exception as exc:
            result.update(
                {
                    "runtime_total": time.perf_counter() - start_total,
                    "status": "failed",
                    "error_message": repr(exc),
                }
            )
            if algorithm == "FK":
                print(f"FK task failed: m={m}, error={result['error_message']}", flush=True)

    save_task_json(result, task_path)
    return result


def _task_paths_for_request(
    out_dir: str,
    gamma: float,
    m_start: int,
    m_fk_max: int,
    m_nhhd_max: int,
    hmis_same_as_nhhd: bool,
    fk_method: str,
    fk_step_factor: int,
    m_hmis_max: int | None = None,
) -> dict[tuple[str, int], str]:
    hmis_max = m_nhhd_max if hmis_same_as_nhhd else (m_hmis_max or m_nhhd_max)
    paths = {}
    for m in range(m_start, m_fk_max + 1):
        paths[("FK", m)] = task_json_path(out_dir, "FK", m, gamma, fk_method, fk_step_factor)
    for m in range(m_start, m_nhhd_max + 1):
        paths[("NHHD", m)] = task_json_path(out_dir, "NHHD", m, gamma)
    for m in range(m_start, hmis_max + 1):
        paths[("HMIS", m)] = task_json_path(out_dir, "HMIS", m, gamma)
    return paths


def aggregate_task_results(
    out_dir: str,
    gamma: float,
    m_start: int,
    m_fk_max: int,
    m_nhhd_max: int,
    hmis_same_as_nhhd: bool = True,
    fk_method: str = "expm",
    fk_step_factor: int = 16,
    m_hmis_max: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate task JSON files into wide and long data frames."""
    hmis_max = m_nhhd_max if hmis_same_as_nhhd else (m_hmis_max or m_nhhd_max)
    max_m = max(m_fk_max, m_nhhd_max, hmis_max)
    task_paths = _task_paths_for_request(
        out_dir,
        gamma,
        m_start,
        m_fk_max,
        m_nhhd_max,
        hmis_same_as_nhhd,
        fk_method,
        fk_step_factor,
        m_hmis_max=m_hmis_max,
    )

    wide_rows = []
    long_rows = []
    for m in range(m_start, max_m + 1):
        info = ck_graph_info(m)
        n = info["n"]
        L = n * (n + info["num_edges"])
        row = {
            "m": m,
            "n": n,
            "L": L,
            "gamma": gamma,
            "p_FK": np.nan,
            "p_NHHD": np.nan,
            "p_HHD": np.nan,
            "p_Grover": np.nan,
            "ideal_NH_circuit": np.nan,
            "computed_FK": False,
            "computed_NHHD": False,
            "computed_HHD": False,
            "fk_norm_error": np.nan,
            "runtime_FK": 0.0,
            "runtime_NHHD": 0.0,
            "runtime_HMIS": 0.0,
            "runtime_prepare_instance": 0.0,
            "note": "",
        }
        prep_path = prepare_task_json_path(out_dir, m, gamma)
        if os.path.exists(prep_path):
            prep = load_task_json(prep_path)
            row["runtime_prepare_instance"] = prep.get("runtime_prepare_instance", 0.0)
            if prep.get("ideal_NH_circuit") is not None:
                row["ideal_NH_circuit"] = prep["ideal_NH_circuit"]

        for algorithm in ["FK", "NHHD", "HMIS"]:
            path = task_paths.get((algorithm, m))
            task = load_task_json(path) if path and os.path.exists(path) else None
            computed = bool(task and task.get("status") == "success")
            success = task.get("success_probability") if computed else np.nan
            note = "" if computed else "missing"
            if task and task.get("status") == "failed":
                note = task.get("error_message", "failed")

            if algorithm == "FK":
                row["p_FK"] = success
                row["computed_FK"] = computed
                row["runtime_FK"] = task.get("runtime_solver", 0.0) if task else 0.0
                row["fk_norm_error"] = task.get("fk_norm_error") if computed else np.nan
            elif algorithm == "NHHD":
                row["p_NHHD"] = success
                row["computed_NHHD"] = computed
                row["runtime_NHHD"] = task.get("runtime_solver", 0.0) if task else 0.0
                if computed and task.get("ideal_NH_circuit") is not None:
                    row["ideal_NH_circuit"] = task["ideal_NH_circuit"]
            else:
                row["p_HHD"] = success
                row["computed_HHD"] = computed
                row["runtime_HMIS"] = task.get("runtime_solver", 0.0) if task else 0.0
                row["p_Grover"] = task.get("p_Grover") if computed else np.nan

            long_rows.append(
                {
                    "algorithm": algorithm,
                    "m": m,
                    "n": n,
                    "L": L,
                    "gamma": gamma,
                    "success_probability": success,
                    "computed": computed,
                    "runtime_total": task.get("runtime_total", 0.0) if task else 0.0,
                    "status": task.get("status", "missing") if task else "missing",
                    "note": note,
                }
            )

        notes = []
        if not row["computed_FK"]:
            notes.append("FK not computed")
        if not row["computed_NHHD"]:
            notes.append("NHHD not computed")
        if not row["computed_HHD"]:
            notes.append("HMIS not computed")
        row["note"] = "; ".join(notes)
        wide_rows.append(row)

    return (
        pd.DataFrame(wide_rows, columns=WIDE_COLUMNS),
        pd.DataFrame(long_rows, columns=LONG_COLUMNS),
    )


def write_aggregates(
    out_dir: str,
    gamma: float,
    m_start: int,
    m_fk_max: int,
    m_nhhd_max: int,
    hmis_same_as_nhhd: bool,
    fk_method: str,
    fk_step_factor: int,
    partial: bool = False,
    m_hmis_max: int | None = None,
) -> tuple[str, str]:
    wide, long = aggregate_task_results(
        out_dir,
        gamma,
        m_start,
        m_fk_max,
        m_nhhd_max,
        hmis_same_as_nhhd=hmis_same_as_nhhd,
        fk_method=fk_method,
        fk_step_factor=fk_step_factor,
        m_hmis_max=m_hmis_max,
    )
    suffix = "_partial" if partial else ""
    wide_path = os.path.join(out_dir, f"mixed_scaling_gamma10{suffix}.csv")
    long_path = os.path.join(out_dir, f"mixed_scaling_gamma10_long{suffix}.csv")
    wide.to_csv(wide_path, index=False)
    long.to_csv(long_path, index=False)
    return wide_path, long_path


def code_version() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def write_metadata(
    out_dir: str,
    gamma: float,
    p: float,
    q: float,
    Omega: float,
    fk_method: str,
    fk_step_factor: int,
    m_start: int,
    m_fk_max_requested: int,
    m_nhhd_max_requested: int,
    hmis_max_m: int,
    command_line: list[str],
) -> str:
    wide, _long = aggregate_task_results(
        out_dir,
        gamma,
        m_start,
        m_fk_max_requested,
        m_nhhd_max_requested,
        hmis_same_as_nhhd=True,
        fk_method=fk_method,
        fk_step_factor=fk_step_factor,
    )
    fk_done = wide.loc[wide["computed_FK"] == True, "m"]
    nhhd_done = wide.loc[wide["computed_NHHD"] == True, "m"]
    metadata = {
        "gamma": gamma,
        "p": p,
        "q": q,
        "Omega": Omega,
        "fk_method": fk_method,
        "fk_step_factor": fk_step_factor,
        "m_start": m_start,
        "m_fk_max_requested": m_fk_max_requested,
        "m_fk_max_completed": int(fk_done.max()) if len(fk_done) else None,
        "m_nhhd_max_requested": m_nhhd_max_requested,
        "m_nhhd_max_completed": int(nhhd_done.max()) if len(nhhd_done) else None,
        "hmis_max_m": hmis_max_m,
        "created_at": now_iso(),
        "command_line": command_line,
        "code_version_if_available": code_version(),
    }
    path = os.path.join(out_dir, "mixed_scaling_gamma10_metadata.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return path


def _submit_or_load_prepare(
    executor,
    task: tuple,
    out_dir: str,
    force: bool,
):
    m, gamma, _p, _q, _out_dir, _force = task
    path = prepare_task_json_path(out_dir, m, gamma)
    cached = cached_success(path, force=force)
    if cached is not None and os.path.exists(instance_cache_path(out_dir, m)):
        print(f"cache hit: PREPARE m={m}", flush=True)
        return cached, None
    return None, executor.submit(_prepare_worker, task)


def _submit_or_load_algorithm(
    executor,
    task: tuple,
    out_dir: str,
    force: bool,
):
    algorithm, m, gamma, _Omega, _out_dir, fk_method, _fk_num_steps, fk_step_factor, _fk_min_steps, _force = task
    path = task_json_path(out_dir, algorithm, m, gamma, fk_method, fk_step_factor)
    cached = cached_success(path, force=force)
    if cached is not None:
        print(f"cache hit: {algorithm} m={m}", flush=True)
        return cached, None
    return None, executor.submit(_algorithm_worker, task)


def run_extended(
    gamma: float,
    p: float,
    q: float,
    Omega: float,
    m_start: int,
    m_fk_max: int,
    m_nhhd_max: int,
    hmis_same_as_nhhd: bool,
    fk_method: str,
    fk_num_steps: int | None,
    fk_step_factor: int,
    fk_min_steps: int,
    out_dir: str,
    max_workers: int | None,
    reserve_cores: int,
    serial: bool,
    force: bool,
) -> None:
    ensure_dirs(out_dir)
    hmis_max = m_nhhd_max if hmis_same_as_nhhd else m_nhhd_max
    workers = max_workers if max_workers is not None else default_max_workers(reserve_cores)
    if not serial:
        print(
            f"using ProcessPoolExecutor with max_workers = {workers}, "
            f"reserve_cores = {reserve_cores}",
            flush=True,
        )

    prepare_values = list(range(max(m_fk_max, m_nhhd_max), m_start - 1, -1))
    prepare_tasks = [(m, gamma, p, q, out_dir, force) for m in prepare_values]

    if serial:
        for task in prepare_tasks:
            result = _prepare_worker(task)
            print(
                f"{'cache hit' if result.get('cache_hit') else 'finished'}: PREPARE m={result['m']} "
                f"status={result['status']} elapsed={result.get('runtime_total', 0.0):.3f}s",
                flush=True,
            )
            write_aggregates(
                out_dir,
                gamma,
                m_start,
                m_fk_max,
                m_nhhd_max,
                hmis_same_as_nhhd,
                fk_method,
                fk_step_factor,
                partial=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for task in prepare_tasks:
                cached, future = _submit_or_load_prepare(executor, task, out_dir, force)
                if future is not None:
                    futures.append(future)
            for future in as_completed(futures):
                result = future.result()
                print(
                    f"finished: PREPARE m={result['m']} status={result['status']} "
                    f"elapsed={result.get('runtime_total', 0.0):.3f}s",
                    flush=True,
                )
                write_aggregates(
                    out_dir,
                    gamma,
                    m_start,
                    m_fk_max,
                    m_nhhd_max,
                    hmis_same_as_nhhd,
                    fk_method,
                    fk_step_factor,
                    partial=True,
                )

    algorithm_tasks = [
        (
            algorithm,
            m,
            gamma,
            Omega,
            out_dir,
            fk_method,
            fk_num_steps,
            fk_step_factor,
            fk_min_steps,
            force,
        )
        for algorithm, m in build_dynamic_tasks(
            m_start, m_fk_max, m_nhhd_max, hmis_same_as_nhhd=hmis_same_as_nhhd
        )
    ]

    if serial:
        for task in algorithm_tasks:
            result = _algorithm_worker(task)
            print(
                f"{'cache hit' if result.get('cache_hit') else 'finished'}: "
                f"{result['algorithm']} m={result['m']} status={result['status']} "
                f"elapsed={result.get('runtime_total', 0.0):.3f}s",
                flush=True,
            )
            write_aggregates(
                out_dir,
                gamma,
                m_start,
                m_fk_max,
                m_nhhd_max,
                hmis_same_as_nhhd,
                fk_method,
                fk_step_factor,
                partial=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for task in algorithm_tasks:
                cached, future = _submit_or_load_algorithm(executor, task, out_dir, force)
                if future is not None:
                    futures.append(future)
            for future in as_completed(futures):
                result = future.result()
                print(
                    f"finished: {result['algorithm']} m={result['m']} status={result['status']} "
                    f"elapsed={result.get('runtime_total', 0.0):.3f}s",
                    flush=True,
                )
                write_aggregates(
                    out_dir,
                    gamma,
                    m_start,
                    m_fk_max,
                    m_nhhd_max,
                    hmis_same_as_nhhd,
                    fk_method,
                    fk_step_factor,
                    partial=True,
                )

    wide_path, long_path = write_aggregates(
        out_dir,
        gamma,
        m_start,
        m_fk_max,
        m_nhhd_max,
        hmis_same_as_nhhd,
        fk_method,
        fk_step_factor,
        partial=False,
    )
    metadata_path = write_metadata(
        out_dir,
        gamma,
        p,
        q,
        Omega,
        fk_method,
        fk_step_factor,
        m_start,
        m_fk_max,
        m_nhhd_max,
        hmis_max,
        sys.argv,
    )
    png_path = os.path.join(out_dir, "mixed_scaling_gamma10.png")
    pdf_path = os.path.join(out_dir, "mixed_scaling_gamma10.pdf")
    fig = plot_mixed_scaling_from_csv(wide_path, png_path, pdf=pdf_path)
    fig.clf()
    print(f"saved {wide_path}", flush=True)
    print(f"saved {long_path}", flush=True)
    print(f"saved {metadata_path}", flush=True)
    print(f"saved {png_path}", flush=True)
    print(f"saved {pdf_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run resumable mixed scaling.")
    parser.add_argument("--gamma", type=float, default=10.0)
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--q", type=float, default=4.0)
    parser.add_argument("--Omega", type=float, default=1.0)
    parser.add_argument("--m-start", type=int, default=3)
    parser.add_argument("--m-fk-max", type=int, default=10)
    parser.add_argument("--m-nhhd-max", type=int, default=18)
    parser.add_argument("--hmis-same-as-nhhd", action="store_true")
    parser.add_argument("--fk-method", choices=["expm", "ivp"], default="expm")
    parser.add_argument("--fk-step-factor", type=int, default=16)
    parser.add_argument("--fk-min-steps", type=int, default=200)
    parser.add_argument("--fk-num-steps", type=int)
    parser.add_argument("--out-dir", default="outputs/extended_scaling_gamma10")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--reserve-cores", type=int, default=1)
    parser.add_argument("--serial", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_extended(
        gamma=args.gamma,
        p=args.p,
        q=args.q,
        Omega=args.Omega,
        m_start=args.m_start,
        m_fk_max=args.m_fk_max,
        m_nhhd_max=args.m_nhhd_max,
        hmis_same_as_nhhd=args.hmis_same_as_nhhd,
        fk_method=args.fk_method,
        fk_num_steps=args.fk_num_steps,
        fk_step_factor=args.fk_step_factor,
        fk_min_steps=args.fk_min_steps,
        out_dir=args.out_dir,
        max_workers=args.max_workers,
        reserve_cores=args.reserve_cores,
        serial=args.serial,
        force=args.force,
    )


if __name__ == "__main__":
    main()
