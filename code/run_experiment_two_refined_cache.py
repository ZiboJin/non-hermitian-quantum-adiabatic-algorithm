"""Refined cache-first pipeline for experiment two.

This script keeps the existing final-overnight outputs intact and writes a
separate refined run.  Its first-class artifact is the FK logF grid cache:

    logF[i,j] = log10 min_x sigma_min(z_ij I - H_x(r)).

The same pair E0-E1 cache is then reused for the k=1 right threshold and for
the FK part of the left figure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import mpmath as mp
except ImportError:  # pragma: no cover
    mp = None

try:
    from .experiment_two_resolvent_pseudospectrum import (
        GATE_ORDER,
        GridWindow,
        clock_eigenvalues,
        component_contours_from_log_grid,
        compute_log_grid_global_hp,
        ensure_dirs,
        global_condition_estimate_for_r,
        hd_block,
        history_sector_data_for_r,
        local_condition_scales_for_r,
        log10_sigma_min_fk_global_hp,
        plot_history_right,
        pseudospectral_closing_threshold_grid_log,
        radial_log_boundary_global_hp,
        single_component_contours_from_log_grid,
    )
except ImportError:  # pragma: no cover
    from experiment_two_resolvent_pseudospectrum import (
        GATE_ORDER,
        GridWindow,
        clock_eigenvalues,
        component_contours_from_log_grid,
        compute_log_grid_global_hp,
        ensure_dirs,
        global_condition_estimate_for_r,
        hd_block,
        history_sector_data_for_r,
        local_condition_scales_for_r,
        log10_sigma_min_fk_global_hp,
        plot_history_right,
        pseudospectral_closing_threshold_grid_log,
        radial_log_boundary_global_hp,
        single_component_contours_from_log_grid,
    )


def _logger(out_dir: str | Path) -> logging.Logger:
    dirs = ensure_dirs(out_dir)
    logger = logging.getLogger("experiment_two_refined_cache")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(dirs["logs"] / "experiment2_refined_cache.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    return logger


def _append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
        key_cols = [
            col
            for col in ["job_id", "kind", "r", "k", "center_type", "grid_N", "worker_label"]
            if col in df.columns
        ]
        if key_cols:
            df = df.drop_duplicates(subset=key_cols, keep="last")
    df.to_csv(path, index=False)


def _cache_key(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:20]


def _mp_clock_eigenvalue_str(L: int, k: int, mp_dps: int) -> str:
    if mp is None:
        raise RuntimeError("mpmath is required for high-precision clock eigenvalues")
    mp.mp.dps = int(mp_dps)
    value = mp.mpf("1") - mp.cos(mp.mpf(int(k)) * mp.pi / mp.mpf(int(L) + 1))
    return mp.nstr(value, n=int(mp_dps) - 5, strip_zeros=False)


def _local_center_strings(L: int, center_type: str, mp_dps: int) -> tuple[str, str, float]:
    if center_type == "E0":
        return "0", "0", 0.0
    if center_type == "E1":
        real = _mp_clock_eigenvalue_str(L, 1, mp_dps)
        return real, "0", float(real)
    raise ValueError(f"unknown center_type={center_type!r}")


def _shifted_log_grid_worker(args: tuple) -> tuple[int, np.ndarray]:
    (
        row_start,
        ys_chunk,
        xs,
        sector_v_lists,
        center_real_str,
        center_imag_str,
        mp_dps,
        n_iter,
        tol_log10,
    ) = args
    if mp is None:
        raise RuntimeError("mpmath is required for shifted local grids")
    sector_data = [{"v": np.asarray(v, dtype=float)} for v in sector_v_lists]
    mp.mp.dps = int(mp_dps)
    c = mp.mpc(str(center_real_str), str(center_imag_str))
    out = np.empty((len(ys_chunk), len(xs)), dtype=float)
    for iy, y in enumerate(ys_chunk):
        y_mp = mp.mpf(str(float(y)))
        for ix, x in enumerate(xs):
            z = c + mp.mpc(mp.mpf(str(float(x))), y_mp)
            out[iy, ix] = log10_sigma_min_fk_global_hp(
                sector_data, z, mp_dps=mp_dps, n_iter=n_iter, tol_log10=tol_log10
            )
    return int(row_start), out


def compute_shifted_local_log_grid_cache(
    sector_data_list: list[dict],
    *,
    center: complex,
    center_real_str: str | None = None,
    center_imag_str: str | None = None,
    hx: float,
    hy: float,
    grid_N: int,
    mp_dps: int,
    hp_n_iter: int,
    hp_target_log10: float,
    p: float,
    q: float,
    cache_dir: str | Path,
    cache_payload: dict,
    force: bool,
    max_workers: int,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    grid_N = int(grid_N)
    if grid_N % 2 == 0:
        raise ValueError("shifted local grid requires odd grid_N so the center point is exactly E0/E1")
    center_grid_index = grid_N // 2
    xs = np.linspace(-float(hx), float(hx), grid_N, dtype=float)
    ys = np.linspace(-float(hy), float(hy), grid_N, dtype=float)
    xs[center_grid_index] = 0.0
    ys[center_grid_index] = 0.0
    center_real_hp = str(center_real_str) if center_real_str is not None else str(float(np.real(center)))
    center_imag_hp = str(center_imag_str) if center_imag_str is not None else str(float(np.imag(center)))
    payload = dict(cache_payload)
    payload.update(
        {
            "local_coordinates": True,
            "center_real_hp": center_real_hp,
            "center_imag_hp": center_imag_hp,
            "center_grid_index": int(center_grid_index),
            "center_offset_is_exact_zero": True,
            "hx": float(hx),
            "hy": float(hy),
            "grid_N": grid_N,
            "mp_dps": int(mp_dps),
            "hp_n_iter": int(hp_n_iter),
            "hp_target_log10": float(hp_target_log10),
            "p": float(p),
            "q": float(q),
            "gate_order": GATE_ORDER,
            "calculation_kind": "refined_local_shifted",
        }
        )
    cache_path = cache_dir / f"loggrid_{_cache_key(payload)}.npz"
    if cache_path.exists() and not force:
        loaded = np.load(cache_path, allow_pickle=True)
        return (
            loaded["xs"],
            loaded["ys"],
            np.nan_to_num(loaded["logF"], nan=math.inf, posinf=math.inf, neginf=-math.inf),
            {
                "cache_path": str(cache_path),
                "from_cache": True,
                "local_coordinates": True,
                "center_real_hp": center_real_hp,
                "center_imag_hp": center_imag_hp,
            },
        )
    start = time.perf_counter()
    workers = max(1, int(max_workers))
    row_chunk = max(1, int(math.ceil(len(ys) / (workers * 4))))
    chunks = [(idx, ys[idx : idx + row_chunk]) for idx in range(0, len(ys), row_chunk)]
    sector_v_lists = [np.asarray(sector["v"], dtype=float) for sector in sector_data_list]
    logF = np.empty((len(ys), len(xs)), dtype=float)
    if workers == 1:
        for idx, chunk in chunks:
            _, rows = _shifted_log_grid_worker(
                (idx, chunk, xs, sector_v_lists, center_real_hp, center_imag_hp, mp_dps, hp_n_iter, 1e-8)
            )
            logF[idx : idx + len(chunk), :] = rows
            logger.info("shifted local rows %d/%d done", min(idx + len(chunk), len(ys)), len(ys))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _shifted_log_grid_worker,
                    (idx, chunk, xs, sector_v_lists, center_real_hp, center_imag_hp, mp_dps, hp_n_iter, 1e-8),
                )
                for idx, chunk in chunks
            ]
            done = 0
            for future in as_completed(futures):
                idx, rows = future.result()
                logF[idx : idx + rows.shape[0], :] = rows
                done += rows.shape[0]
                logger.info("shifted local rows %d/%d done", done, len(ys))
    logF = np.nan_to_num(logF, nan=math.inf, posinf=math.inf, neginf=-math.inf)
    np.savez_compressed(
        cache_path,
        xs=xs,
        ys=ys,
        logF=logF,
        local_coordinates=np.array([True]),
        center_real=np.array([float(np.real(center))]),
        center_imag=np.array([float(np.imag(center))]),
        center_real_hp=np.array(center_real_hp),
        center_imag_hp=np.array(center_imag_hp),
        center_grid_index=np.array([center_grid_index]),
        center_offset_is_exact_zero=np.array([True]),
    )
    return xs, ys, logF, {
        "cache_path": str(cache_path),
        "from_cache": False,
        "runtime_seconds": float(time.perf_counter() - start),
        "local_coordinates": True,
        "center_real_hp": center_real_hp,
        "center_imag_hp": center_imag_hp,
    }


@dataclass(frozen=True)
class QuadCell:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    level: int


def _point_key(x: float, y: float) -> str:
    return f"{float(x):.18e},{float(y):.18e}"


def _adaptive_points_worker(args: tuple) -> list[tuple[float, float, float]]:
    (
        points,
        sector_v_lists,
        center_real_str,
        center_imag_str,
        mp_dps,
        n_iter,
        tol_log10,
    ) = args
    if mp is None:
        raise RuntimeError("mpmath is required for adaptive local contours")
    sector_data = [{"v": np.asarray(v, dtype=float)} for v in sector_v_lists]
    mp.mp.dps = int(mp_dps)
    c = mp.mpc(str(center_real_str), str(center_imag_str))
    out: list[tuple[float, float, float]] = []
    for x, y in points:
        z = c + mp.mpc(mp.mpf(str(float(x))), mp.mpf(str(float(y))))
        logf = log10_sigma_min_fk_global_hp(
            sector_data, z, mp_dps=mp_dps, n_iter=n_iter, tol_log10=tol_log10
        )
        out.append((float(x), float(y), float(logf)))
    return out


def _batch_eval_adaptive_points(
    points: list[tuple[float, float]],
    *,
    sample_cache: dict[str, float],
    sector_v_lists: list[np.ndarray],
    center_real_hp: str,
    center_imag_hp: str,
    mp_dps: int,
    hp_n_iter: int,
    max_workers: int,
    logger: logging.Logger,
) -> None:
    unique: list[tuple[float, float]] = []
    seen: set[str] = set()
    for x, y in points:
        key = _point_key(x, y)
        if key not in sample_cache and key not in seen:
            unique.append((float(x), float(y)))
            seen.add(key)
    if not unique:
        return
    workers = max(1, min(int(max_workers), len(unique)))
    chunk_size = max(1, int(math.ceil(len(unique) / (workers * 4))))
    chunks = [unique[i : i + chunk_size] for i in range(0, len(unique), chunk_size)]
    if workers == 1:
        for chunk in chunks:
            for x, y, logf in _adaptive_points_worker(
                (chunk, sector_v_lists, center_real_hp, center_imag_hp, mp_dps, hp_n_iter, 1e-8)
            ):
                sample_cache[_point_key(x, y)] = logf
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _adaptive_points_worker,
                    (chunk, sector_v_lists, center_real_hp, center_imag_hp, mp_dps, hp_n_iter, 1e-8),
                )
                for chunk in chunks
            ]
            done = 0
            for future in as_completed(futures):
                for x, y, logf in future.result():
                    sample_cache[_point_key(x, y)] = logf
                    done += 1
                logger.info("adaptive local samples %d/%d done", done, len(unique))


def _sample_logf(sample_cache: dict[str, float], x: float, y: float) -> float:
    return float(sample_cache[_point_key(x, y)])


def _cell_points(cell: QuadCell, include_center: bool = True) -> list[tuple[float, float]]:
    pts = [
        (cell.xmin, cell.ymin),
        (cell.xmax, cell.ymin),
        (cell.xmax, cell.ymax),
        (cell.xmin, cell.ymax),
    ]
    if include_center:
        pts.append((0.5 * (cell.xmin + cell.xmax), 0.5 * (cell.ymin + cell.ymax)))
    return pts


def _cell_g_values(cell: QuadCell, sample_cache: dict[str, float], logeps: float) -> list[float]:
    return [_sample_logf(sample_cache, x, y) - float(logeps) for x, y in _cell_points(cell)]


def _classify_cell(cell: QuadCell, sample_cache: dict[str, float], logeps: float) -> str:
    vals = _cell_g_values(cell, sample_cache, logeps)
    if not all(np.isfinite(vals)):
        return "ambiguous"
    has_inside = any(v <= 0.0 for v in vals)
    has_outside = any(v > 0.0 for v in vals)
    if has_inside and has_outside:
        return "boundary"
    return "inside" if has_inside else "outside"


def _split_cell(cell: QuadCell) -> list[QuadCell]:
    xm = 0.5 * (cell.xmin + cell.xmax)
    ym = 0.5 * (cell.ymin + cell.ymax)
    level = cell.level + 1
    return [
        QuadCell(cell.xmin, xm, cell.ymin, ym, level),
        QuadCell(xm, cell.xmax, cell.ymin, ym, level),
        QuadCell(xm, cell.xmax, ym, cell.ymax, level),
        QuadCell(cell.xmin, xm, ym, cell.ymax, level),
    ]


def _initial_cells(radius: float, coarse_cells: int) -> list[QuadCell]:
    xs = np.linspace(-float(radius), float(radius), int(coarse_cells) + 1)
    ys = np.linspace(-float(radius), float(radius), int(coarse_cells) + 1)
    return [
        QuadCell(float(xs[ix]), float(xs[ix + 1]), float(ys[iy]), float(ys[iy + 1]), 0)
        for iy in range(int(coarse_cells))
        for ix in range(int(coarse_cells))
    ]


def _square_boundary_points(radius: float, per_edge: int = 9) -> list[tuple[float, float]]:
    vals = np.linspace(-float(radius), float(radius), int(per_edge))
    pts: list[tuple[float, float]] = []
    for v in vals:
        pts.extend([(float(v), -float(radius)), (float(v), float(radius)), (-float(radius), float(v)), (float(radius), float(v))])
    return pts


def _edge_root_bisection(
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    sample_cache: dict[str, float],
    sector_v_lists: list[np.ndarray],
    center_real_hp: str,
    center_imag_hp: str,
    logeps: float,
    mp_dps: int,
    hp_n_iter: int,
    max_workers: int,
    iters: int,
    logger: logging.Logger,
) -> tuple[float, float]:
    x0, y0 = p0
    x1, y1 = p1
    g0 = _sample_logf(sample_cache, x0, y0) - logeps
    for _ in range(int(iters)):
        xm = 0.5 * (x0 + x1)
        ym = 0.5 * (y0 + y1)
        _batch_eval_adaptive_points(
            [(xm, ym)],
            sample_cache=sample_cache,
            sector_v_lists=sector_v_lists,
            center_real_hp=center_real_hp,
            center_imag_hp=center_imag_hp,
            mp_dps=mp_dps,
            hp_n_iter=hp_n_iter,
            max_workers=max_workers,
            logger=logger,
        )
        gm = _sample_logf(sample_cache, xm, ym) - logeps
        if (g0 <= 0.0 and gm <= 0.0) or (g0 > 0.0 and gm > 0.0):
            x0, y0, g0 = xm, ym, gm
        else:
            x1, y1 = xm, ym
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1))


def _cell_segments(
    cell: QuadCell,
    *,
    sample_cache: dict[str, float],
    sector_v_lists: list[np.ndarray],
    center_real_hp: str,
    center_imag_hp: str,
    logeps: float,
    mp_dps: int,
    hp_n_iter: int,
    max_workers: int,
    bisection_iters: int,
    logger: logging.Logger,
) -> list[tuple[float, float, float, float]]:
    corners = [
        (cell.xmin, cell.ymin),
        (cell.xmax, cell.ymin),
        (cell.xmax, cell.ymax),
        (cell.xmin, cell.ymax),
    ]
    vals = [_sample_logf(sample_cache, x, y) - logeps for x, y in corners]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    points: list[tuple[float, float]] = []
    for a, b in edges:
        ga, gb = vals[a], vals[b]
        if not (np.isfinite(ga) and np.isfinite(gb)):
            continue
        if ga == 0.0:
            points.append(corners[a])
        if (ga <= 0.0 < gb) or (gb <= 0.0 < ga):
            points.append(
                _edge_root_bisection(
                    corners[a],
                    corners[b],
                    sample_cache=sample_cache,
                    sector_v_lists=sector_v_lists,
                    center_real_hp=center_real_hp,
                    center_imag_hp=center_imag_hp,
                    logeps=logeps,
                    mp_dps=mp_dps,
                    hp_n_iter=hp_n_iter,
                    max_workers=max_workers,
                    iters=bisection_iters,
                    logger=logger,
                )
            )
    if len(points) < 2:
        return []
    if len(points) == 2:
        return [(points[0][0], points[0][1], points[1][0], points[1][1])]
    # Ambiguous four-crossing cells are rare for these local contours.  Use the
    # angular order around the cell center as a stable asymptotic-decider proxy.
    cx = 0.5 * (cell.xmin + cell.xmax)
    cy = 0.5 * (cell.ymin + cell.ymax)
    points = sorted(points, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return [(points[0][0], points[0][1], points[1][0], points[1][1]), (points[2][0], points[2][1], points[3][0], points[3][1])] if len(points) >= 4 else []


def _contour_from_segments(segments: np.ndarray) -> np.ndarray:
    if segments.size == 0:
        return np.empty((0, 2), dtype=float)
    pts = np.vstack([segments[:, :2], segments[:, 2:4]])
    if len(pts) < 3:
        return np.empty((0, 2), dtype=float)
    # Merge near-duplicate vertices by angle.  Local E0/E1 contours are simple
    # closed loops; angular ordering is robust and avoids imposing a dense grid.
    rounded: dict[tuple[int, int], list[np.ndarray]] = {}
    scale = max(float(np.nanmax(np.abs(pts))), 1e-300)
    q = max(scale * 1e-10, 1e-300)
    for pnt in pts:
        key = (int(round(float(pnt[0]) / q)), int(round(float(pnt[1]) / q)))
        rounded.setdefault(key, []).append(np.asarray(pnt, dtype=float))
    unique = np.array([np.mean(v, axis=0) for v in rounded.values()], dtype=float)
    if len(unique) < 3:
        return np.empty((0, 2), dtype=float)
    order = np.argsort(np.arctan2(unique[:, 1], unique[:, 0]))
    contour = unique[order]
    if np.linalg.norm(contour[0] - contour[-1]) > q:
        contour = np.vstack([contour, contour[0]])
    return contour


def _winding_number_seed(contour: np.ndarray) -> float:
    if len(contour) < 4:
        return 0.0
    angles = np.unwrap(np.arctan2(contour[:, 1], contour[:, 0]))
    return float((angles[-1] - angles[0]) / (2.0 * np.pi))


def adaptive_quadtree_local_contour(
    sector_data_list: list[dict],
    *,
    center_real_hp: str,
    center_imag_hp: str,
    center: complex,
    center_type: str,
    r: int,
    grid_N: int,
    epsilon_vis: float,
    mp_dps: int,
    hp_n_iter: int,
    hp_target_log10: float,
    p: float,
    q: float,
    cache_dir: str | Path,
    cache_payload: dict,
    force: bool,
    max_workers: int,
    logger: logging.Logger,
    local_radius_factor: float,
    coarse_cells: int,
    max_levels: int,
    rel_tol: float,
    abs_tol: float,
    max_active_cells: int,
    outer_expand_factor: float,
    outer_max_expansions: int,
    edge_bisection_iters: int,
    radial_bootstrap_angles: int,
    save_debug_cells: bool,
) -> tuple[np.ndarray, dict]:
    del radial_bootstrap_angles  # Reserved for optional sizing only; final contour is quadtree-based.
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    logeps = math.log10(float(epsilon_vis))
    condition_radius = _condition_radius(sector_data_list, center_type, epsilon_vis, factor=local_radius_factor)
    min_radius = max(10.0 * float(epsilon_vis), 1e-300)
    initial_R = max(float(condition_radius), min_radius)
    payload = dict(cache_payload)
    payload.update(
        {
            "cache_kind": "adaptive_quadtree_contour",
            "local_contour_method": "adaptive_quadtree",
            "center_real_hp": str(center_real_hp),
            "center_imag_hp": str(center_imag_hp),
            "epsilon_vis": float(epsilon_vis),
            "grid_N_target": int(grid_N),
            "condition_radius": float(condition_radius),
            "local_radius_factor": float(local_radius_factor),
            "coarse_cells": int(coarse_cells),
            "max_levels": int(max_levels),
            "rel_tol": float(rel_tol),
            "abs_tol": float(abs_tol),
            "max_active_cells": int(max_active_cells),
            "outer_expand_factor": float(outer_expand_factor),
            "outer_max_expansions": int(outer_max_expansions),
            "edge_bisection_iters": int(edge_bisection_iters),
            "mp_dps": int(mp_dps),
            "hp_n_iter": int(hp_n_iter),
            "hp_target_log10": float(hp_target_log10),
            "p": float(p),
            "q": float(q),
            "gate_order": GATE_ORDER,
        }
    )
    cache_path = cache_dir / f"adaptive_local_contour_{_cache_key(payload)}.npz"
    if cache_path.exists() and not force:
        loaded = np.load(cache_path, allow_pickle=True)
        metadata = json.loads(str(loaded["metadata_json"].item()))
        metadata.update({"cache_path": str(cache_path), "from_cache": True})
        return np.asarray(loaded["contour_xy"], dtype=float), metadata

    start = time.perf_counter()
    sector_v_lists = [np.asarray(sector["v"], dtype=float) for sector in sector_data_list]
    sample_cache: dict[str, float] = {}
    _batch_eval_adaptive_points(
        [(0.0, 0.0)],
        sample_cache=sample_cache,
        sector_v_lists=sector_v_lists,
        center_real_hp=center_real_hp,
        center_imag_hp=center_imag_hp,
        mp_dps=mp_dps,
        hp_n_iter=hp_n_iter,
        max_workers=max_workers,
        logger=logger,
    )
    seed_logf = _sample_logf(sample_cache, 0.0, 0.0)
    if np.isnan(seed_logf) or seed_logf == math.inf or seed_logf > logeps:
        metadata = {
            "cache_kind": "adaptive_quadtree_contour",
            "local_contour_method": "adaptive_quadtree",
            "condition_radius": float(condition_radius),
            "initial_R": float(initial_R),
            "outer_R": float(initial_R),
            "outer_expansions": 0,
            "seed_logF": float(seed_logf),
            "seed_was_physical": bool(seed_logf <= logeps),
            "final_failed": True,
            "final_failed_reason": "seed_not_inside",
            "runtime_seconds": float(time.perf_counter() - start),
        }
        np.savez_compressed(
            cache_path,
            contour_xy=np.empty((0, 2), dtype=float),
            sample_xy=np.array([[0.0, 0.0]], dtype=float),
            sample_logF=np.array([seed_logf], dtype=float),
            segments=np.empty((0, 4), dtype=float),
            leaf_cells=np.empty((0, 5), dtype=float),
            metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
        )
        metadata.update({"cache_path": str(cache_path), "from_cache": False})
        return np.empty((0, 2), dtype=float), metadata

    R = float(initial_R)
    outer_expansions = 0
    touches_outer_boundary = False
    for outer_expansions in range(int(outer_max_expansions) + 1):
        boundary_pts = _square_boundary_points(R, per_edge=9)
        _batch_eval_adaptive_points(
            boundary_pts,
            sample_cache=sample_cache,
            sector_v_lists=sector_v_lists,
            center_real_hp=center_real_hp,
            center_imag_hp=center_imag_hp,
            mp_dps=mp_dps,
            hp_n_iter=hp_n_iter,
            max_workers=max_workers,
            logger=logger,
        )
        boundary_g = [_sample_logf(sample_cache, x, y) - logeps for x, y in boundary_pts]
        touches_outer_boundary = any(np.isfinite(g) and g <= 0.0 for g in boundary_g)
        if not touches_outer_boundary:
            break
        R *= float(outer_expand_factor)

    cells = _initial_cells(R, int(coarse_cells))
    active = cells
    final_boundary: list[QuadCell] = []
    inside_count = 0
    outside_count = 0
    levels_used = 0
    stopped_reason = "max_levels"
    leaf_diag = math.sqrt(2.0) * 2.0 * R / max(int(coarse_cells), 1)
    for level in range(int(max_levels) + 1):
        levels_used = level
        pts: list[tuple[float, float]] = []
        for cell in active:
            pts.extend(_cell_points(cell))
        _batch_eval_adaptive_points(
            pts,
            sample_cache=sample_cache,
            sector_v_lists=sector_v_lists,
            center_real_hp=center_real_hp,
            center_imag_hp=center_imag_hp,
            mp_dps=mp_dps,
            hp_n_iter=hp_n_iter,
            max_workers=max_workers,
            logger=logger,
        )
        next_active: list[QuadCell] = []
        for cell in active:
            status = _classify_cell(cell, sample_cache, logeps)
            if status == "inside":
                inside_count += 1
                continue
            if status == "outside":
                outside_count += 1
                continue
            diag = math.hypot(cell.xmax - cell.xmin, cell.ymax - cell.ymin)
            leaf_diag = min(leaf_diag, diag)
            if (
                level >= int(max_levels)
                or (float(abs_tol) > 0.0 and diag <= float(abs_tol))
                or diag <= float(rel_tol) * float(R)
            ):
                final_boundary.append(cell)
            else:
                next_active.extend(_split_cell(cell))
        if len(next_active) > int(max_active_cells):
            final_boundary.extend(next_active[: int(max_active_cells)])
            stopped_reason = "too_many_active_cells"
            break
        if not next_active:
            stopped_reason = "converged"
            break
        active = next_active

    segment_rows: list[tuple[float, float, float, float]] = []
    for cell in final_boundary:
        segment_rows.extend(
            _cell_segments(
                cell,
                sample_cache=sample_cache,
                sector_v_lists=sector_v_lists,
                center_real_hp=center_real_hp,
                center_imag_hp=center_imag_hp,
                logeps=logeps,
                mp_dps=mp_dps,
                hp_n_iter=hp_n_iter,
                max_workers=max_workers,
                bisection_iters=edge_bisection_iters,
                logger=logger,
            )
        )
    segments = np.asarray(segment_rows, dtype=float).reshape((-1, 4)) if segment_rows else np.empty((0, 4), dtype=float)
    contour_xy = _contour_from_segments(segments)
    winding = _winding_number_seed(contour_xy)
    contour_closed = bool(len(contour_xy) >= 4 and np.linalg.norm(contour_xy[0] - contour_xy[-1]) <= max(10.0 * leaf_diag, 1e-300))
    contour_contains_seed = bool(abs(winding) > 0.5)
    final_failed_reason = ""
    if touches_outer_boundary:
        final_failed_reason = "touches_outer_boundary"
    elif stopped_reason == "too_many_active_cells":
        final_failed_reason = "too_many_active_cells"
    elif len(contour_xy) < 4:
        final_failed_reason = "no_closed_polyline"
    elif not contour_contains_seed:
        final_failed_reason = "contour_does_not_contain_seed"
    final_failed = bool(final_failed_reason)

    if len(contour_xy):
        bbox_x_min = float(np.nanmin(contour_xy[:, 0]))
        bbox_x_max = float(np.nanmax(contour_xy[:, 0]))
        bbox_y_min = float(np.nanmin(contour_xy[:, 1]))
        bbox_y_max = float(np.nanmax(contour_xy[:, 1]))
        local_hx = float(np.nanmax(np.abs(contour_xy[:, 0])))
        local_hy = float(np.nanmax(np.abs(contour_xy[:, 1])))
    else:
        bbox_x_min = bbox_x_max = bbox_y_min = bbox_y_max = math.nan
        local_hx = local_hy = math.nan
    contour_g = []
    for x, y in contour_xy[:: max(1, len(contour_xy) // 32)] if len(contour_xy) else []:
        _batch_eval_adaptive_points(
            [(float(x), float(y))],
            sample_cache=sample_cache,
            sector_v_lists=sector_v_lists,
            center_real_hp=center_real_hp,
            center_imag_hp=center_imag_hp,
            mp_dps=mp_dps,
            hp_n_iter=hp_n_iter,
            max_workers=max_workers,
            logger=logger,
        )
        contour_g.append(abs(_sample_logf(sample_cache, float(x), float(y)) - logeps))
    sample_items = sorted(sample_cache.items())
    sample_xy = np.array([[float(k.split(",")[0]), float(k.split(",")[1])] for k, _ in sample_items], dtype=float)
    sample_logf = np.array([float(v) for _, v in sample_items], dtype=float)
    metadata = {
        "cache_kind": "adaptive_quadtree_contour",
        "local_contour_method": "adaptive_quadtree",
        "condition_radius": float(condition_radius),
        "initial_R": float(initial_R),
        "outer_R": float(R),
        "outer_expansions": int(outer_expansions),
        "local_quadtree_coarse_cells": int(coarse_cells),
        "local_quadtree_max_levels": int(max_levels),
        "local_quadtree_levels_used": int(levels_used),
        "local_contour_rel_tol": float(rel_tol),
        "local_contour_abs_tol": float(abs_tol),
        "local_max_active_cells": int(max_active_cells),
        "n_samples_total": int(len(sample_cache)),
        "n_boundary_leaf_cells": int(len(final_boundary)),
        "n_inside_leaf_cells": int(inside_count),
        "n_outside_leaf_cells": int(outside_count),
        "n_segments": int(len(segments)),
        "n_contour_vertices": int(len(contour_xy)),
        "contour_closed": bool(contour_closed),
        "contour_contains_seed": bool(contour_contains_seed),
        "winding_number_seed": float(winding),
        "max_abs_g_on_contour_sample": float(np.nanmax(contour_g)) if contour_g else math.nan,
        "median_abs_g_on_contour_sample": float(np.nanmedian(contour_g)) if contour_g else math.nan,
        "seed_logF": float(seed_logf),
        "seed_was_physical": bool(seed_logf <= logeps),
        "touches_outer_boundary": bool(touches_outer_boundary),
        "final_failed": bool(final_failed),
        "final_failed_reason": final_failed_reason,
        "runtime_seconds": float(time.perf_counter() - start),
        "local_hx": local_hx,
        "local_hy": local_hy,
        "outer_R": float(R),
        "contour_bbox_x_min": bbox_x_min,
        "contour_bbox_x_max": bbox_x_max,
        "contour_bbox_y_min": bbox_y_min,
        "contour_bbox_y_max": bbox_y_max,
        "center_real_hp": str(center_real_hp),
        "center_imag_hp": str(center_imag_hp),
    }
    leaf_cells = (
        np.array([[c.xmin, c.xmax, c.ymin, c.ymax, c.level] for c in final_boundary], dtype=float)
        if save_debug_cells
        else np.empty((0, 5), dtype=float)
    )
    np.savez_compressed(
        cache_path,
        contour_xy=contour_xy,
        sample_xy=sample_xy,
        sample_logF=sample_logf,
        segments=segments,
        leaf_cells=leaf_cells,
        metadata_json=np.array(json.dumps(metadata, sort_keys=True)),
    )
    metadata.update({"cache_path": str(cache_path), "from_cache": False})
    return contour_xy, metadata


def _pair_window(r: int, grid_N: int, p: float, q: float, expand_factor: float = 2.0) -> tuple[int, float, GridWindow]:
    sectors = history_sector_data_for_r(r, n=5, p=p, q=q)
    L = len(sectors[0]["v"])
    lam = float(clock_eigenvalues(L)[1])
    window = GridWindow(-0.2 * lam, 1.2 * lam, -0.8 * lam, 0.8 * lam, int(grid_N), int(grid_N))
    if float(expand_factor) != 1.0:
        window = window.expanded(float(expand_factor))
    if not (window.x_min <= 0.0 <= window.x_max and window.x_min <= lam <= window.x_max):
        raise RuntimeError(f"pair window does not contain E0/E1 for r={r}: {window}")
    return L, lam, window


def run_pair_cache_jobs(
    *,
    r_list: list[int],
    out_dir: str | Path,
    grid_N: int,
    grid_N_r5: int | None,
    mp_dps: int,
    hp_n_iter: int,
    hp_target_log10: float,
    epsilon_vis: float,
    max_workers: int,
    force: bool,
    worker_label: str,
    p: float,
    q: float,
    pair_window_expand: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    dirs = ensure_dirs(out_dir)
    rows: list[dict] = []
    for r in r_list:
        sectors = history_sector_data_for_r(r, n=5, p=p, q=q)
        grid_N_this = int(grid_N_r5) if int(r) == 5 and grid_N_r5 is not None else int(grid_N)
        L, lam, window = _pair_window(r, grid_N_this, p=p, q=q, expand_factor=pair_window_expand)
        expand_tag = f"x{float(pair_window_expand):g}".replace(".", "p")
        job_id = f"pair_r{r}_k1_N{grid_N_this}_{expand_tag}"
        t0 = time.perf_counter()
        xs, ys, logF, info = compute_log_grid_global_hp(
            sectors,
            window,
            mp_dps=mp_dps,
            cache_dir=dirs["cache"],
            cache_payload={
                "kind": "refined_pair_E0_E1",
                "r": int(r),
                "k": 1,
                "grid_N": int(grid_N_this),
                "grid_N_pair_default": int(grid_N),
                "grid_N_pair_r5": None if grid_N_r5 is None else int(grid_N_r5),
                "pair_window_expand": float(pair_window_expand),
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
            calculation_kind="refined_pair_E0_E1",
            logger=logger,
        )
        threshold = pseudospectral_closing_threshold_grid_log(
            logF,
            xs,
            ys,
            eigenvalues=[0.0 + 0j, complex(lam, 0.0)],
            ground_indices=[0],
            excited_indices=[1],
        )
        component = component_contours_from_log_grid(logF, xs, ys, (0.0 + 0j, complex(lam, 0.0)), math.log10(epsilon_vis))
        runtime = time.perf_counter() - t0
        rows.append(
            {
                "job_id": job_id,
                "kind": "pair_E0_E1",
                "worker_label": worker_label,
                "r": int(r),
                "L": int(L),
                "k": 1,
                "grid_N": int(grid_N_this),
                "grid_N_pair_default": int(grid_N),
                "grid_N_pair_r5": None if grid_N_r5 is None else int(grid_N_r5),
                "mp_dps": int(mp_dps),
                "hp_n_iter": int(hp_n_iter),
                "hp_target_log10": float(hp_target_log10),
                "epsilon_vis": float(epsilon_vis),
                "pair_window_expand": float(pair_window_expand),
                "window_x_min": window.x_min,
                "window_x_max": window.x_max,
                "window_y_min": window.y_min,
                "window_y_max": window.y_max,
                "contains_E0": True,
                "contains_Ek": True,
                "cache_path": str(info["cache_path"]),
                "from_cache": bool(info.get("from_cache", False)),
                "runtime_seconds": float(runtime),
                "log10_epsilon_c_grid_hp": float(threshold["log10_epsilon_c"]),
                "touches_boundary": bool(threshold["touches_boundary"]),
                "E0_E1_connected_at_epsilon_vis": bool(component["connected"]),
                "contour_status_at_epsilon_vis": str(component["contour_status"]),
                "contour_touches_boundary_at_epsilon_vis": bool(component["touches_boundary"]),
                "gate_order": GATE_ORDER,
                "status": "success",
            }
        )
        logger.info(
            "refined pair r=%d L=%d grid=%d log10_threshold=%.6g connected@eps=%s cache=%s",
            r,
            L,
            grid_N_this,
            threshold["log10_epsilon_c"],
            component["connected"],
            info["cache_path"],
        )
    path = dirs["data"] / "pair_grid_manifest.csv"
    _append_csv(path, rows)
    return pd.DataFrame(rows)


def _condition_radius(sectors: list[dict], center_type: str, epsilon_vis: float, factor: float = 8.0) -> float:
    scales = local_condition_scales_for_r(sectors, len(sectors[0]["v"]) - 1)
    k = 0 if center_type == "E0" else 1
    return max(float(factor) * float(epsilon_vis) * float(scales[k]), 1e-300)


def _local_window_from_radial(
    sectors: list[dict],
    center: complex,
    center_type: str,
    epsilon_vis: float,
    mp_dps: int,
    hp_n_iter: int,
    max_workers: int,
    local_radius_factor: float,
    logger: logging.Logger,
) -> tuple[GridWindow, dict]:
    base_radius = _condition_radius(sectors, center_type, epsilon_vis, factor=local_radius_factor)
    # Keep cache precomputation purely grid-based.  A previous implementation
    # used radial high-precision sizing here; it was too slow to be a job
    # scheduler primitive.  The radius is intentionally tighter than the old
    # factor-50 window so that 81x81 points resolve the epsilon contour.
    hx = base_radius
    hy = base_radius
    window = GridWindow(
        float(np.real(center)) - hx,
        float(np.real(center)) + hx,
        float(np.imag(center)) - hy,
        float(np.imag(center)) + hy,
        1,
        1,
    )
    logger.info(
        "local condition sizing center=%s type=%s base_radius=%.3e hx=%.3e hy=%.3e",
        center,
        center_type,
        base_radius,
        hx,
        hy,
    )
    return window, {"base_radius": base_radius, "radial_hx": math.nan, "radial_hy": math.nan}


def run_local_cache_jobs(
    *,
    r_list: list[int],
    center_list: list[str],
    out_dir: str | Path,
    grid_N: int,
    mp_dps: int,
    hp_n_iter: int,
    hp_target_log10: float,
    epsilon_vis: float,
    max_workers: int,
    force: bool,
    worker_label: str,
    p: float,
    q: float,
    local_radius_factor: float,
    local_contour_method: str,
    local_quadtree_coarse_cells: int,
    local_quadtree_max_levels: int,
    local_contour_rel_tol: float,
    local_contour_abs_tol: float,
    local_max_active_cells: int,
    local_outer_expand_factor: float,
    local_outer_max_expansions: int,
    local_edge_bisection_iters: int,
    local_radial_bootstrap_angles: int,
    save_adaptive_debug_cells: bool,
    logger: logging.Logger,
) -> pd.DataFrame:
    dirs = ensure_dirs(out_dir)
    grid_N = int(grid_N)
    if grid_N % 2 == 0:
        raise ValueError("local phase requires odd --grid-N-local so the central grid point is exactly E0/E1")
    rows: list[dict] = []
    for r in r_list:
        sectors = history_sector_data_for_r(r, n=5, p=p, q=q)
        L = len(sectors[0]["v"])
        for center_type in center_list:
            center_real_hp, center_imag_hp, center_real_float = _local_center_strings(L, center_type, mp_dps)
            center = complex(center_real_float, 0.0)
            condition_radius = _condition_radius(sectors, center_type, epsilon_vis, factor=local_radius_factor)
            job_id = f"local_{local_contour_method}_r{r}_{center_type}_N{grid_N}"
            t0 = time.perf_counter()
            if local_contour_method == "adaptive_quadtree":
                contour_xy, info = adaptive_quadtree_local_contour(
                    sectors,
                    center_real_hp=center_real_hp,
                    center_imag_hp=center_imag_hp,
                    center=center,
                    center_type=center_type,
                    r=int(r),
                    grid_N=grid_N,
                    epsilon_vis=epsilon_vis,
                    mp_dps=mp_dps,
                    hp_n_iter=hp_n_iter,
                    hp_target_log10=hp_target_log10,
                    p=p,
                    q=q,
                    cache_dir=dirs["cache"],
                    cache_payload={
                        "kind": "refined_local_adaptive",
                        "r": int(r),
                        "center_type": center_type,
                        "center_precision": "mp",
                    },
                    force=force,
                    max_workers=max_workers,
                    logger=logger,
                    local_radius_factor=local_radius_factor,
                    coarse_cells=local_quadtree_coarse_cells,
                    max_levels=local_quadtree_max_levels,
                    rel_tol=local_contour_rel_tol,
                    abs_tol=local_contour_abs_tol,
                    max_active_cells=local_max_active_cells,
                    outer_expand_factor=local_outer_expand_factor,
                    outer_max_expansions=local_outer_max_expansions,
                    edge_bisection_iters=local_edge_bisection_iters,
                    radial_bootstrap_angles=local_radial_bootstrap_angles,
                    save_debug_cells=save_adaptive_debug_cells,
                )
                component = {
                    "contour_status": "ok" if not bool(info.get("final_failed", True)) else "adaptive_failed",
                    "touches_boundary": bool(info.get("touches_outer_boundary", False)),
                    "contours": [contour_xy] if len(contour_xy) else [],
                    "physical_points_total": int(info.get("n_samples_total", -1)),
                    "component_physical_points": int(info.get("n_inside_leaf_cells", -1)),
                    "seed_was_physical": bool(info.get("seed_was_physical", False)),
                }
                hx = float(info.get("local_hx", math.nan))
                hy = float(info.get("local_hy", math.nan))
                cache_kind = "adaptive_quadtree_contour"
                contour_source = "adaptive_quadtree_boundary_trace"
            elif local_contour_method == "grid":
                sizing_window, sizing = _local_window_from_radial(
                    sectors, center, center_type, epsilon_vis, mp_dps, hp_n_iter, max_workers, local_radius_factor, logger
                )
                hx = float(sizing["base_radius"])
                hy = float(sizing["base_radius"])
                xs, ys, logF, info = compute_shifted_local_log_grid_cache(
                    sectors,
                    center=center,
                    center_real_str=center_real_hp,
                    center_imag_str=center_imag_hp,
                    hx=hx,
                    hy=hy,
                    grid_N=grid_N,
                    mp_dps=mp_dps,
                    cache_dir=dirs["cache"],
                    cache_payload={
                        "kind": "refined_local",
                        "r": int(r),
                        "center_type": center_type,
                        "center_precision": "mp",
                        "center_real_hp": center_real_hp,
                        "center_imag_hp": center_imag_hp,
                        "grid_N": int(grid_N),
                        "local_radius_factor": float(local_radius_factor),
                        "hp_target_log10": float(hp_target_log10),
                        "p": float(p),
                        "q": float(q),
                    },
                    force=force,
                    max_workers=max_workers,
                    hp_n_iter=hp_n_iter,
                    hp_target_log10=hp_target_log10,
                    p=p,
                    q=q,
                    logger=logger,
                )
                component = single_component_contours_from_log_grid(
                    logF, xs, ys, 0.0 + 0.0j, math.log10(epsilon_vis), force_seed=False
                )
                cache_kind = "refined_local_shifted_grid"
                contour_source = "refined_local_shifted_grid_cache"
            else:
                raise ValueError(f"unknown local_contour_method={local_contour_method!r}")
            runtime = time.perf_counter() - t0
            rows.append(
                {
                    "job_id": job_id,
                    "kind": "local_hpcenter",
                    "cache_kind": cache_kind,
                    "local_contour_method": local_contour_method,
                    "contour_source": contour_source,
                    "worker_label": worker_label,
                    "r": int(r),
                    "L": int(L),
                    "center_type": center_type,
                    "center_precision": "mp",
                    "center_value": float(np.real(center)),
                    "center_value_hp": center_real_hp,
                    "center_imag_hp": center_imag_hp,
                    "grid_N": int(grid_N),
                    "mp_dps": int(mp_dps),
                    "hp_n_iter": int(hp_n_iter),
                    "hp_target_log10": float(hp_target_log10),
                    "epsilon_vis": float(epsilon_vis),
                    "local_radius_factor": float(local_radius_factor),
                    "condition_radius": float(condition_radius),
                    "window_x_min": float(np.real(center)) - hx if np.isfinite(hx) else math.nan,
                    "window_x_max": float(np.real(center)) + hx if np.isfinite(hx) else math.nan,
                    "window_y_min": float(np.imag(center)) - hy if np.isfinite(hy) else math.nan,
                    "window_y_max": float(np.imag(center)) + hy if np.isfinite(hy) else math.nan,
                    "local_coordinates": True,
                    "center_grid_index": int(grid_N // 2),
                    "center_offset_is_exact_zero": local_contour_method == "grid",
                    "force_seed": False,
                    "local_x_min": -hx if np.isfinite(hx) else math.nan,
                    "local_x_max": hx if np.isfinite(hx) else math.nan,
                    "local_y_min": -hy if np.isfinite(hy) else math.nan,
                    "local_y_max": hy if np.isfinite(hy) else math.nan,
                    "local_hx": hx,
                    "local_hy": hy,
                    "base_radius": float(condition_radius),
                    "radial_hx": math.nan,
                    "radial_hy": math.nan,
                    "cache_path": str(info["cache_path"]),
                    "adaptive_cache_path": str(info["cache_path"]) if cache_kind == "adaptive_quadtree_contour" else "",
                    "from_cache": bool(info.get("from_cache", False)),
                    "runtime_seconds": float(runtime),
                    "contour_status": str(component["contour_status"]),
                    "contour_touches_boundary": bool(component["touches_boundary"]),
                    "n_contours": int(len(component["contours"])),
                    "physical_points_total": int(component.get("physical_points_total", -1)),
                    "component_physical_points": int(component.get("component_physical_points", -1)),
                    "seed_was_physical": bool(component.get("seed_was_physical", False)),
                    "initial_R": float(info.get("initial_R", math.nan)),
                    "outer_R": float(info.get("outer_R", math.nan)),
                    "outer_expansions": int(info.get("outer_expansions", -1)),
                    "local_quadtree_coarse_cells": int(info.get("local_quadtree_coarse_cells", local_quadtree_coarse_cells)),
                    "local_quadtree_max_levels": int(info.get("local_quadtree_max_levels", local_quadtree_max_levels)),
                    "local_quadtree_levels_used": int(info.get("local_quadtree_levels_used", -1)),
                    "local_contour_rel_tol": float(info.get("local_contour_rel_tol", local_contour_rel_tol)),
                    "local_contour_abs_tol": float(info.get("local_contour_abs_tol", local_contour_abs_tol)),
                    "local_max_active_cells": int(info.get("local_max_active_cells", local_max_active_cells)),
                    "n_samples_total": int(info.get("n_samples_total", -1)),
                    "n_boundary_leaf_cells": int(info.get("n_boundary_leaf_cells", -1)),
                    "n_inside_leaf_cells": int(info.get("n_inside_leaf_cells", -1)),
                    "n_outside_leaf_cells": int(info.get("n_outside_leaf_cells", -1)),
                    "n_segments": int(info.get("n_segments", -1)),
                    "n_contour_vertices": int(info.get("n_contour_vertices", -1)),
                    "contour_closed": bool(info.get("contour_closed", False)),
                    "contour_contains_seed": bool(info.get("contour_contains_seed", False)),
                    "winding_number_seed": float(info.get("winding_number_seed", math.nan)),
                    "max_abs_g_on_contour_sample": float(info.get("max_abs_g_on_contour_sample", math.nan)),
                    "median_abs_g_on_contour_sample": float(info.get("median_abs_g_on_contour_sample", math.nan)),
                    "seed_logF": float(info.get("seed_logF", math.nan)),
                    "final_failed": bool(info.get("final_failed", component["contour_status"] != "ok")),
                    "final_failed_reason": str(info.get("final_failed_reason", "")),
                    "contour_bbox_x_min": float(info.get("contour_bbox_x_min", math.nan)),
                    "contour_bbox_x_max": float(info.get("contour_bbox_x_max", math.nan)),
                    "contour_bbox_y_min": float(info.get("contour_bbox_y_min", math.nan)),
                    "contour_bbox_y_max": float(info.get("contour_bbox_y_max", math.nan)),
                    "gate_order": GATE_ORDER,
                    "status": "success",
                }
            )
            logger.info(
                "refined local r=%d %s grid=%d status=%s n_contours=%d cache=%s",
                r,
                center_type,
                grid_N,
                component["contour_status"],
                len(component["contours"]),
                info["cache_path"],
            )
    path = dirs["data"] / "local_grid_manifest.csv"
    _append_csv(path, rows)
    return pd.DataFrame(rows)


def _mp_sigma_min_2x2(H: np.ndarray, z: complex, dps: int = 90) -> mp.mpf:
    if mp is None:
        raise RuntimeError("mpmath is required for refined HD contours")
    mp.mp.dps = int(dps)
    if isinstance(H, (list, tuple)):
        h00 = mp.mpc(H[0][0])
        h01 = mp.mpc(H[0][1])
        h10 = mp.mpc(H[1][0])
        h11 = mp.mpc(H[1][1])
    else:
        h00 = mp.mpc(H[0, 0])
        h01 = mp.mpc(H[0, 1])
        h10 = mp.mpc(H[1, 0])
        h11 = mp.mpc(H[1, 1])
    zz = mp.mpc(z)
    a = zz - h00
    b = -h01
    c = -h10
    d = zz - h11
    T = abs(a) ** 2 + abs(b) ** 2 + abs(c) ** 2 + abs(d) ** 2
    D = abs(a * d - b * c) ** 2
    disc = mp.sqrt(max(T * T - 4 * D, mp.mpf("0")))
    eig_min = (T - disc) / 2
    return mp.sqrt(max(eig_min, mp.mpf("0")))


def hd_refined_contour_scaled(
    sigma: float,
    center: float,
    epsilon_vis: float,
    theta: float,
    omega: float,
    n_angles: int = 241,
    mp_dps: int = 90,
) -> np.ndarray:
    mp.mp.dps = int(mp_dps)
    sigma_mp = mp.mpf(str(sigma))
    theta_mp = mp.mpf(str(theta))
    omega_mp = mp.mpf(str(omega))
    st = mp.sin(theta_mp)
    ct = mp.cos(theta_mp)
    H = [
        [omega_mp * st * st, -omega_mp * st * ct / sigma_mp],
        [-omega_mp * st * ct * sigma_mp, omega_mp * ct * ct],
    ]
    angles = np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=True)
    pts = np.empty((len(angles), 2), dtype=float)
    eps = mp.mpf(str(epsilon_vis))
    center_mp = mp.mpf(str(center))
    for idx, phi in enumerate(angles):
        direction = mp.mpc(str(math.cos(float(phi))), str(math.sin(float(phi))))
        lo = mp.mpf("0")
        hi = mp.mpf("1")
        for _ in range(120):
            z = mp.mpc(center_mp, mp.mpf("0")) + eps * hi * direction
            if _mp_sigma_min_2x2(H, z, dps=mp_dps) > eps:
                break
            hi *= 2
        for _ in range(100):
            mid = (lo + hi) / 2
            z = mp.mpc(center_mp, mp.mpf("0")) + eps * mid * direction
            if _mp_sigma_min_2x2(H, z, dps=mp_dps) <= eps:
                lo = mid
            else:
                hi = mid
        rho = float((lo + hi) / 2)
        pts[idx] = [rho * math.cos(float(phi)), rho * math.sin(float(phi))]
    return pts


def assemble_refined_outputs(
    *,
    out_dir: str | Path,
    epsilon_vis: float,
    theta: float,
    omega: float,
    q: float,
    logger: logging.Logger,
) -> None:
    dirs = ensure_dirs(out_dir)
    pair_path = dirs["data"] / "pair_grid_manifest.csv"
    local_path = dirs["data"] / "local_grid_manifest.csv"
    if not pair_path.exists():
        raise FileNotFoundError(pair_path)
    pair_df = pd.read_csv(pair_path)
    if local_path.exists():
        local_df = pd.read_csv(local_path)
    else:
        local_df = pd.DataFrame()
    if local_df.empty:
        local_df = pd.DataFrame(columns=["r", "center_type", "contour_status", "contour_touches_boundary", "grid_N"])

    def best_pair_rows(df: pd.DataFrame) -> pd.DataFrame:
        selected = []
        for _, sub in df.groupby("r", sort=True):
            work = sub.copy()
            if "pair_window_expand" not in work.columns:
                work["pair_window_expand"] = 1.0
            work["pair_window_expand"] = pd.to_numeric(work["pair_window_expand"], errors="coerce").fillna(1.0)
            work["touches_boundary"] = work["touches_boundary"].astype(bool)
            candidates = work[~work["touches_boundary"]]
            if candidates.empty:
                candidates = work
            sort_cols = [col for col in ["grid_N", "pair_window_expand", "runtime_seconds"] if col in candidates.columns]
            selected.append(candidates.sort_values(sort_cols).iloc[-1] if sort_cols else candidates.iloc[-1])
        return pd.DataFrame(selected).sort_values("r")

    # Right figure: k=1 thresholds from shared pair grids.
    right_rows = []
    best_pair_df = best_pair_rows(pair_df)
    for _, row in best_pair_df.iterrows():
        summary, _ = global_condition_estimate_for_r(int(row["r"]), n=5, p=2.0, q=q)
        right_rows.append(
            {
                "r": int(row["r"]),
                "L": int(row["L"]),
                "n": 5,
                "m": 2,
                "Delta_FK": float(summary["Delta_FK"]),
                "log10_epsilon_vis": math.log10(float(epsilon_vis)),
                "hp_target_log10": float(row["hp_target_log10"]),
                "mp_dps": int(row["mp_dps"]),
                "grid_N_right": int(row["grid_N"]),
                "log10_epsilon_c_grid_hp": float(row["log10_epsilon_c_grid_hp"]),
                "k_grid_best": 1,
                "candidate_k_list": "1",
                "log10_epsilon_est_global": float(summary["log10_epsilon_est_global"]),
                "k_star_est": int(summary["k_star_est"]),
                "sector_est_min": int(summary["sector_est_min"]),
                "sector_est_min_bitstring": str(summary["sector_est_min_bitstring"]),
                "reliable_hp_grid": not bool(row["touches_boundary"]),
                "touches_boundary": bool(row["touches_boundary"]),
                "status": "success",
            }
        )
    right_df = pd.DataFrame(right_rows)
    right_df.to_csv(dirs["data"] / "thresholds_numerical.csv", index=False)
    plot_history_right(right_df, dirs["figures"])
    for src_name, dst_name in [
        ("fig2_right_fk_history_threshold_hp.pdf", "fig2_right_fk_history_threshold_refined_k1.pdf"),
        ("fig2_right_fk_history_threshold_hp.png", "fig2_right_fk_history_threshold_refined_k1.png"),
    ]:
        src = dirs["figures"] / src_name
        if src.exists():
            src.replace(dirs["figures"] / dst_name)

    arrays: dict[str, np.ndarray] = {}
    contour_rows: list[dict] = []
    contour_idx = 0
    logeps = math.log10(float(epsilon_vis))

    def clean_str(value, default: str) -> str:
        try:
            if pd.isna(value):
                return default
        except TypeError:
            pass
        return str(value)

    def clean_float(value, default: float) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return float(default)
        return out if np.isfinite(out) else float(default)

    # r=1..4 local contours from local cache; r=5 connected contour from pair cache.
    for _, pair in best_pair_df.iterrows():
        r = int(pair["r"])
        L = int(pair["L"])
        pair_mp_dps = int(pair["mp_dps"]) if "mp_dps" in pair else 80
        E0_hp = "0"
        E1_hp = _mp_clock_eigenvalue_str(L, 1, pair_mp_dps)
        E1 = float(E1_hp)
        if r == 5:
            data = np.load(pair["cache_path"])
            component = component_contours_from_log_grid(
                data["logF"], data["xs"], data["ys"], (0.0 + 0j, complex(E1, 0.0)), logeps
            )
            contours = list(component["contours"])
            if not contours:
                contours = [None]
            for contour in contours:
                key = ""
                if contour is not None:
                    key = f"contour_{contour_idx}"
                    arrays[key] = contour
                    contour_idx += 1
                contour_rows.append(
                    {
                        "contour_key": key,
                        "algorithm": "NHMIS-FKQAA-full-direct-sum",
                        "r": r,
                        "L": L,
                        "E0": 0.0,
                        "E1": E1,
                        "E0_hp": E0_hp,
                        "E1_hp": E1_hp,
                        "center_type": "connected_E0_E1",
                        "center_value": np.nan,
                        "center_value_hp": "",
                        "epsilon_vis": epsilon_vis,
                        "E0_E1_connected_at_epsilon_vis": bool(component["connected"]),
                        "contour_status": component["contour_status"],
                        "contour_source": "refined_pair_grid_cache",
                        "contour_touches_boundary": bool(component["touches_boundary"]),
                        "grid_N": int(pair["grid_N"]),
                        "cache_path": pair["cache_path"],
                        "physical_points_total": int(component.get("physical_points_total", -1)),
                        "component_physical_points": int(component.get("component_physical_points", -1)),
                        "seed0_was_physical": bool(component.get("seed0_was_physical", False)),
                        "seed1_was_physical": bool(component.get("seed1_was_physical", False)),
                        "final_failed": bool(component["contour_status"] != "ok"),
                    }
                )
            continue
        for center_type in ["E0", "E1"]:
            sub = local_df[(local_df["r"] == r) & (local_df["center_type"] == center_type)]
            if sub.empty:
                continue
            if "center_precision" in sub.columns:
                hp_sub = sub[sub["center_precision"].astype(str).eq("mp")]
                if not hp_sub.empty:
                    sub = hp_sub
            elif "job_id" in sub.columns:
                hp_sub = sub[sub["job_id"].astype(str).str.contains("hpcenter", na=False)]
                if not hp_sub.empty:
                    sub = hp_sub
            ok_sub = sub[
                (sub["contour_status"] == "ok")
                & (~sub["contour_touches_boundary"].astype(bool))
            ]
            if not ok_sub.empty:
                row = ok_sub.sort_values(["grid_N", "runtime_seconds"]).iloc[-1]
            else:
                row = sub.sort_values("grid_N").iloc[-1]
            default_center_hp, _, default_center_float = _local_center_strings(L, center_type, int(row["mp_dps"]))
            center_value_hp = clean_str(row.get("center_value_hp", default_center_hp), default_center_hp)
            center = clean_float(row.get("center_value", default_center_float), default_center_float)
            cache_kind = clean_str(row.get("cache_kind", ""), "")
            data = np.load(row["cache_path"], allow_pickle=True)
            if cache_kind == "adaptive_quadtree_contour":
                contour = np.asarray(data["contour_xy"], dtype=float)
                if "metadata_json" in data.files:
                    adaptive_meta = json.loads(str(data["metadata_json"].item()))
                else:
                    adaptive_meta = {}
                component = {
                    "contours": [contour] if len(contour) else [],
                    "contour_status": "ok" if not bool(adaptive_meta.get("final_failed", False)) and len(contour) else "adaptive_failed",
                    "touches_boundary": bool(adaptive_meta.get("touches_outer_boundary", False)),
                    "physical_points_total": int(adaptive_meta.get("n_samples_total", -1)),
                    "component_physical_points": int(adaptive_meta.get("n_inside_leaf_cells", -1)),
                    "seed_was_physical": bool(adaptive_meta.get("seed_was_physical", False)),
                }
                local_coordinates = True
            else:
                local_coordinates = bool(np.asarray(data["local_coordinates"]).ravel()[0]) if "local_coordinates" in data.files else False
                seed_center = 0.0 + 0.0j if local_coordinates else complex(center, 0.0)
                component = single_component_contours_from_log_grid(
                    data["logF"],
                    data["xs"],
                    data["ys"],
                    seed_center,
                    logeps,
                    force_seed=not local_coordinates,
                )
            contours = list(component["contours"])
            if not contours:
                contours = [None]
            for contour in contours:
                key = ""
                if contour is not None:
                    key = f"contour_{contour_idx}"
                    contour_plot = np.asarray(contour, dtype=float).copy()
                    arrays[key] = contour_plot
                    contour_idx += 1
                contour_rows.append(
                    {
                        "contour_key": key,
                        "algorithm": "NHMIS-FKQAA-full-direct-sum",
                        "r": r,
                        "L": L,
                        "E0": 0.0,
                        "E1": E1,
                        "E0_hp": E0_hp,
                        "E1_hp": E1_hp,
                        "center_type": center_type,
                        "center_value": center,
                        "center_value_hp": center_value_hp,
                        "epsilon_vis": epsilon_vis,
                        "E0_E1_connected_at_epsilon_vis": False,
                        "contour_status": component["contour_status"],
                        "contour_source": clean_str(row.get("contour_source", ""), "") or ("refined_local_shifted_grid_cache" if local_coordinates else "refined_local_grid_cache"),
                        "contour_coordinate_frame": "local_centered" if local_coordinates else "absolute",
                        "contour_touches_boundary": bool(component["touches_boundary"]),
                        "grid_N": int(row["grid_N"]),
                        "cache_kind": cache_kind,
                        "local_contour_method": clean_str(row.get("local_contour_method", ""), ""),
                        "center_grid_index": int(row.get("center_grid_index", int(row["grid_N"]) // 2)),
                        "center_offset_is_exact_zero": bool(row.get("center_offset_is_exact_zero", local_coordinates)),
                        "force_seed": bool(not local_coordinates),
                        "local_hx": float(row.get("local_hx", np.nan)),
                        "local_hy": float(row.get("local_hy", np.nan)),
                        "local_x_min": float(row.get("local_x_min", np.nan)),
                        "local_x_max": float(row.get("local_x_max", np.nan)),
                        "local_y_min": float(row.get("local_y_min", np.nan)),
                        "local_y_max": float(row.get("local_y_max", np.nan)),
                        "cache_path": row["cache_path"],
                        "physical_points_total": int(component.get("physical_points_total", -1)),
                        "component_physical_points": int(component.get("component_physical_points", -1)),
                        "seed_was_physical": bool(component.get("seed_was_physical", False)),
                        "n_samples_total": int(row.get("n_samples_total", component.get("physical_points_total", -1))),
                        "n_contour_vertices": int(row.get("n_contour_vertices", len(contour) if contour is not None else -1)),
                        "contour_closed": bool(row.get("contour_closed", False)),
                        "contour_contains_seed": bool(row.get("contour_contains_seed", False)),
                        "final_failed": bool(row.get("final_failed", component["contour_status"] != "ok")),
                        "final_failed_reason": clean_str(row.get("final_failed_reason", ""), ""),
                    }
                )

    # HD refined local contours in scaled coordinates, stored as real z vertices.
    for algorithm, sigma in [("NHMIS-HDQAA", q), ("HMIS-HDQAA", 1.0)]:
        for center_type, center in [("E0", 0.0), ("E1", float(omega))]:
            scaled = hd_refined_contour_scaled(
                sigma=sigma,
                center=center,
                epsilon_vis=epsilon_vis,
                theta=theta,
                omega=omega,
                n_angles=241,
                mp_dps=90,
            )
            key = f"contour_{contour_idx}"
            arrays[key] = scaled
            contour_rows.append(
                {
                    "contour_key": key,
                    "algorithm": algorithm,
                    "r": "independent_of_r",
                    "L": 1,
                    "E0": 0.0,
                    "E1": float(omega),
                    "center_type": center_type,
                    "center_value": center,
                    "center_value_hp": mp.nstr(mp.mpf(str(center)), n=80, strip_zeros=False),
                    "epsilon_vis": epsilon_vis,
                    "E0_E1_connected_at_epsilon_vis": False,
                    "contour_status": "ok",
                    "contour_source": "refined_hd_mp_radial_local",
                    "contour_coordinate_frame": "local_scaled_by_epsilon",
                    "contour_touches_boundary": False,
                    "grid_N": 0,
                    "cache_path": "",
                    "final_failed": False,
                }
            )
            contour_idx += 1

    contour_df = pd.DataFrame(contour_rows)
    contour_df.to_csv(dirs["data"] / "contours_metadata.csv", index=False)
    np.savez_compressed(dirs["data"] / "contours.npz", **arrays)
    plot_refined_fk_montage(contour_df, arrays, dirs["figures"])
    plot_refined_hd_local_panels(contour_df, arrays, dirs["figures"])
    logger.info("assembled refined outputs")


def plot_refined_fk_montage(contour_df: pd.DataFrame, arrays: dict[str, np.ndarray], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    fk = contour_df[contour_df["algorithm"] == "NHMIS-FKQAA-full-direct-sum"].copy()
    colors = {1: "#5b2a86", 2: "#3b5b92", 3: "#2a9d8f", 4: "#53c45e", 5: "#f1d302"}
    center_markers = {"E0": "s", "E1": "^"}
    fig = plt.figure(figsize=(15.5, 8.8))
    gs = fig.add_gridspec(3, 6, width_ratios=[1.4, 1.4, 1, 1, 1, 1], hspace=0.72, wspace=0.62)
    ax_global = fig.add_subplot(gs[:, :2])
    for _, row in fk.iterrows():
        if row["center_type"] == "connected_E0_E1" and row["contour_key"] in arrays:
            r = int(row["r"])
            arr = arrays[row["contour_key"]]
            ax_global.plot(arr[:, 0], arr[:, 1], color=colors[r], lw=2.0, label="r=5 connected contour")
    for r in range(1, 6):
        sub = fk[fk["r"].astype(str) == str(r)]
        if sub.empty:
            continue
        E1 = float(sub.iloc[0]["E1"])
        ax_global.scatter([0.0], [0.0], marker=center_markers["E0"], s=66, color=colors[r], edgecolor="black", linewidth=0.5, zorder=5)
        ax_global.scatter([E1], [0.0], marker=center_markers["E1"], s=76, facecolor="white", edgecolor=colors[r], linewidth=1.7, zorder=5)
        ax_global.text(E1, 0.00005 + 0.00002 * (r % 2), f"r={r}", color=colors[r], fontsize=8, ha="center")
    ax_global.axhline(0.0, color="0.7", lw=0.8)
    ax_global.set_title("Panel A: FK global summary", fontsize=13)
    ax_global.set_xlabel("Re z")
    ax_global.set_ylabel("Im z")
    ax_global.grid(True, ls=":", color="0.8")
    ax_global.set_ylim(-0.00075, 0.00075)
    ax_global.legend(loc="upper right", fontsize=8, frameon=True)
    ax_global.text(
        0.02,
        0.04,
        "squares: E0\ntriangles: E1\nlocal r=1..4 contours moved to zoom panels",
        transform=ax_global.transAxes,
        fontsize=9,
        va="bottom",
    )

    def plot_local(ax, row, title):
        key = row["contour_key"]
        center = float(row["center_value"])
        if key in arrays:
            arr = arrays[key].astype(float)
            if str(row.get("contour_coordinate_frame", "absolute")) == "local_centered":
                xy = np.column_stack([arr[:, 0], arr[:, 1]])
            else:
                xy = np.column_stack([arr[:, 0] - center, arr[:, 1]])
            spread = float(np.nanmax(np.abs(xy)))
            if not np.isfinite(spread) or spread <= 0:
                spread = 1.0
            ax.plot(xy[:, 0], xy[:, 1], color=colors[int(row["r"])], lw=1.8)
            marker = center_markers.get(str(row["center_type"]), "s")
            ax.scatter([0], [0], marker=marker, facecolor="white", edgecolor="black", s=28, lw=0.9, zorder=4)
            xmin, xmax = float(np.nanmin(xy[:, 0])), float(np.nanmax(xy[:, 0]))
            ymin, ymax = float(np.nanmin(xy[:, 1])), float(np.nanmax(xy[:, 1]))
            xpad = max(0.12 * (xmax - xmin), 1e-300)
            ypad = max(0.12 * (ymax - ymin), 1e-300)
            ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(ymin - ypad, ymax + ypad)
        else:
            ax.text(0.5, 0.5, "no reliable contour", transform=ax.transAxes, fontsize=8, ha="center", va="center")
        source = str(row.get("contour_source", ""))
        method_label = "adaptive contour trace" if "adaptive" in source else "2D grid fallback"
        ax.text(0.04, 0.92, method_label, transform=ax.transAxes, fontsize=7, va="top")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Re(z-center)", fontsize=7)
        ax.set_ylabel("Im(z-center)", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize(6)
        ax.yaxis.get_offset_text().set_fontsize(6)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, ls=":", color="0.85")

    for idx, r in enumerate([1, 2, 3, 4]):
        row_base = idx // 2
        col_base = 2 + 2 * (idx % 2)
        for j, center_type in enumerate(["E0", "E1"]):
            ax = fig.add_subplot(gs[row_base, col_base + j])
            sub = fk[(fk["r"].astype(str) == str(r)) & (fk["center_type"] == center_type)]
            if not sub.empty:
                drawable = sub[sub["contour_key"].astype(str).isin(arrays.keys())]
                if not drawable.empty:
                    plot_local(ax, drawable.iloc[0], f"r={r} {center_type}")
                else:
                    plot_local(ax, sub.iloc[0], f"r={r} {center_type}")
            else:
                ax.axis("off")

    sub5 = fk[(fk["r"].astype(str) == "5") & (fk["center_type"] == "connected_E0_E1")]
    ax5 = fig.add_subplot(gs[2, 2:6])
    if not sub5.empty:
        drawable5 = sub5[sub5["contour_key"].astype(str).isin(arrays.keys())]
        row = drawable5.iloc[0] if not drawable5.empty else sub5.iloc[0]
        key = row["contour_key"]
        delta = float(row["E1"])
        if key in arrays:
            arr = arrays[key]
            ax5.plot(arr[:, 0], arr[:, 1], color=colors[5], lw=2.0)
        ax5.scatter([0.0], [0.0], marker=center_markers["E0"], facecolor="white", edgecolor="black", s=35, zorder=4)
        ax5.scatter([delta], [0.0], marker=center_markers["E1"], facecolor="white", edgecolor=colors[5], s=42, zorder=4)
    ax5.set_title("r=5 connected, real coordinates", fontsize=10)
    ax5.set_xlabel("Re z")
    ax5.set_ylabel("Im z")
    ax5.set_aspect("equal", adjustable="box")
    ax5.grid(True, ls=":", color="0.85")

    fig.suptitle("FK full-sector union: refined shared-grid montage", fontsize=16, y=0.985)
    fig.savefig(figures_dir / "fig2_left_fk_montage_refined.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / "fig2_left_fk_montage_refined.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_refined_hd_local_panels(contour_df: pd.DataFrame, arrays: dict[str, np.ndarray], figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    hd = contour_df[contour_df["algorithm"].isin(["NHMIS-HDQAA", "HMIS-HDQAA"])].copy()
    if hd.empty:
        return
    colors = {"NHMIS-HDQAA": "#b83280", "HMIS-HDQAA": "#2f6f4e"}
    labels = {"NHMIS-HDQAA": "NHMIS-HD local block", "HMIS-HDQAA": "HMIS-HD local block"}
    fig = plt.figure(figsize=(10.4, 5.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.05, 1.0], hspace=0.54, wspace=0.48)
    for col, algorithm in enumerate(["NHMIS-HDQAA", "HMIS-HDQAA"]):
        color = colors[algorithm]
        ax_main = fig.add_subplot(gs[0, 2 * col : 2 * col + 2])
        ax_main.plot([0.0, 1.0], [0.0, 0.0], color="0.72", lw=1.0)
        ax_main.scatter([0.0, 1.0], [0.0, 0.0], color=[color, "white"], edgecolor=color, s=64, zorder=3)
        ax_main.text(0.0, 0.018, "E0", color=color, fontsize=9, ha="center")
        ax_main.text(1.0, 0.018, "E1", color=color, fontsize=9, ha="center")
        ax_main.set_title(labels[algorithm], fontsize=12)
        ax_main.set_xlim(-0.12, 1.12)
        ax_main.set_ylim(-0.06, 0.06)
        ax_main.set_xlabel("Re z")
        ax_main.set_yticks([0.0])
        ax_main.set_ylabel("Im z")
        ax_main.grid(True, ls=":", color="0.86")
        ax_main.text(0.5, -0.045, "local contours are shown below", ha="center", fontsize=9, color="0.25")
        for j, center_type in enumerate(["E0", "E1"]):
            ax = fig.add_subplot(gs[1, 2 * col + j])
            sub = hd[(hd["algorithm"] == algorithm) & (hd["center_type"] == center_type)]
            if not sub.empty:
                row = sub.iloc[0]
                key = row["contour_key"]
                if key in arrays:
                    arr = arrays[key].astype(float)
                    if str(row.get("contour_coordinate_frame", "")) == "local_scaled_by_epsilon":
                        eps = float(row["epsilon_vis"])
                        xy = eps * arr
                    else:
                        center = float(row["center_value"])
                        xy = np.column_stack([arr[:, 0] - center, arr[:, 1]])
                    spread = float(np.nanmax(np.abs(xy)))
                    if not np.isfinite(spread) or spread <= 0:
                        spread = 1.0
                    ax.plot(xy[:, 0], xy[:, 1], color=color, lw=1.8)
                    pad = 1.15 * spread
                    ax.set_xlim(-pad, pad)
                    ax.set_ylim(-pad, pad)
            ax.scatter([0.0], [0.0], marker="x", color="black", s=18, lw=0.9, zorder=4)
            ax.set_title(f"{center_type} inset", fontsize=9)
            ax.set_xlabel(f"Re(z-{center_type})", fontsize=8)
            ax.set_ylabel(f"Im(z-{center_type})", fontsize=8)
            ax.tick_params(labelsize=8)
            ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
            ax.xaxis.get_offset_text().set_fontsize(7)
            ax.yaxis.get_offset_text().set_fontsize(7)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, ls=":", color="0.86")
    fig.suptitle("HD local-block epsilon contours", fontsize=14, y=0.985)
    fig.savefig(figures_dir / "fig2_left_hd_local_refined.png", dpi=220, bbox_inches="tight")
    fig.savefig(figures_dir / "fig2_left_hd_local_refined.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run refined cache-first experiment two.")
    parser.add_argument("--phase", choices=["pair", "local", "assemble", "all"], required=True)
    parser.add_argument("--r-list", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--center-list", nargs="+", default=["E0", "E1"])
    parser.add_argument("--out-dir", default=os.path.join("outputs", "experiment_two_resolvent_hp_refined"))
    parser.add_argument("--grid-N-pair", type=int, default=81)
    parser.add_argument("--grid-N-pair-r5", type=int, default=241)
    parser.add_argument("--grid-N-local", type=int, default=81)
    parser.add_argument("--pair-window-expand", type=float, default=2.0)
    parser.add_argument("--local-radius-factor", type=float, default=8.0)
    parser.add_argument("--local-contour-method", choices=["adaptive_quadtree", "grid"], default="adaptive_quadtree")
    parser.add_argument("--local-quadtree-coarse-cells", type=int, default=8)
    parser.add_argument("--local-quadtree-max-levels", type=int, default=9)
    parser.add_argument("--local-contour-rel-tol", type=float, default=1e-3)
    parser.add_argument("--local-contour-abs-tol", type=float, default=0.0)
    parser.add_argument("--local-max-active-cells", type=int, default=20000)
    parser.add_argument("--local-outer-expand-factor", type=float, default=2.0)
    parser.add_argument("--local-outer-max-expansions", type=int, default=6)
    parser.add_argument("--local-edge-bisection-iters", type=int, default=32)
    parser.add_argument("--local-radial-bootstrap-angles", type=int, default=0)
    parser.add_argument("--save-adaptive-debug-cells", action="store_true")
    parser.add_argument("--mp-dps", type=int, default=70)
    parser.add_argument("--hp-n-iter", type=int, default=40)
    parser.add_argument("--hp-target-log10", type=float, default=-34.0)
    parser.add_argument("--epsilon-vis", type=float, default=1e-32)
    parser.add_argument("--max-workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--reserve-cores", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker-label", default="local")
    parser.add_argument("--p", type=float, default=2.0)
    parser.add_argument("--q", type=float, default=4.0)
    parser.add_argument("--theta", type=float, default=math.pi / 4.0)
    parser.add_argument("--omega", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = _logger(args.out_dir)
    logger.info(
        "refined phase=%s r_list=%s pair_N=%d pair_N_r5=%d local_N=%d mp_dps=%d hp_n_iter=%d workers=%d",
        args.phase,
        args.r_list,
        args.grid_N_pair,
        args.grid_N_pair_r5,
        args.grid_N_local,
        args.mp_dps,
        args.hp_n_iter,
        args.max_workers,
    )
    if args.phase in {"pair", "all"}:
        run_pair_cache_jobs(
            r_list=args.r_list,
            out_dir=args.out_dir,
            grid_N=args.grid_N_pair,
            grid_N_r5=args.grid_N_pair_r5,
            mp_dps=args.mp_dps,
            hp_n_iter=args.hp_n_iter,
            hp_target_log10=args.hp_target_log10,
            epsilon_vis=args.epsilon_vis,
            max_workers=args.max_workers,
            force=args.force,
            worker_label=args.worker_label,
            p=args.p,
            q=args.q,
            pair_window_expand=args.pair_window_expand,
            logger=logger,
        )
    if args.phase in {"local", "all"}:
        run_local_cache_jobs(
            r_list=[r for r in args.r_list if r < 5],
            center_list=args.center_list,
            out_dir=args.out_dir,
            grid_N=args.grid_N_local,
            mp_dps=args.mp_dps,
            hp_n_iter=args.hp_n_iter,
            hp_target_log10=args.hp_target_log10,
            epsilon_vis=args.epsilon_vis,
            max_workers=args.max_workers,
            force=args.force,
            worker_label=args.worker_label,
            p=args.p,
            q=args.q,
            local_radius_factor=args.local_radius_factor,
            local_contour_method=args.local_contour_method,
            local_quadtree_coarse_cells=args.local_quadtree_coarse_cells,
            local_quadtree_max_levels=args.local_quadtree_max_levels,
            local_contour_rel_tol=args.local_contour_rel_tol,
            local_contour_abs_tol=args.local_contour_abs_tol,
            local_max_active_cells=args.local_max_active_cells,
            local_outer_expand_factor=args.local_outer_expand_factor,
            local_outer_max_expansions=args.local_outer_max_expansions,
            local_edge_bisection_iters=args.local_edge_bisection_iters,
            local_radial_bootstrap_angles=args.local_radial_bootstrap_angles,
            save_adaptive_debug_cells=args.save_adaptive_debug_cells,
            logger=logger,
        )
    if args.phase in {"assemble", "all"}:
        assemble_refined_outputs(
            out_dir=args.out_dir,
            epsilon_vis=args.epsilon_vis,
            theta=args.theta,
            omega=args.omega,
            q=args.q,
            logger=logger,
        )


if __name__ == "__main__":
    main()
