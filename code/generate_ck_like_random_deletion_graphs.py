"""Generate fixed CK-like random-deletion graph pools for experiment one.

The graph class keeps the CK block structure: ``m`` left vertices and
``m - 1`` right-side triangles.  Only left-to-right cross edges are deleted.
Each accepted graph is checked to have a unique maximum independent set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd

try:
    from .experiment_one_ck import ck_graph_info
except ImportError:  # pragma: no cover - used when run as a script path.
    from experiment_one_ck import ck_graph_info


DeletedCrossEdge = tuple[int, int, int]
DEFAULT_DELETE_PROBABILITIES = (0.03, 0.05, 0.07, 0.09, 0.11)
GRAPH_POOL_FILENAME = "graph_pool.jsonl"
GRAPH_POOL_SUMMARY_FILENAME = "graph_pool_summary.csv"
GRAPH_POOL_COLUMNS = [
    "graph_id",
    "graph_hash",
    "m",
    "n",
    "sample_id",
    "seed",
    "delete_probability",
    "probability_bucket_index",
    "attempt_index",
    "deleted_cross_edges",
    "deleted_edge_count",
    "num_edges",
    "L",
    "x_star_bits",
    "x_star_index",
    "mis_size",
    "x_star_left_count",
    "x_star_right_count",
]


def ck_cross_edges(m: int) -> list[DeletedCrossEdge]:
    """Return CK cross edges as ``(left, triangle, triangle_vertex)``."""
    if m < 1:
        raise ValueError("m must be positive")
    return [(left, tri, u) for left in range(m) for tri in range(m - 1) for u in range(3)]


def deleted_right_vertex(m: int, deleted_edge: DeletedCrossEdge) -> int:
    _left, tri, u = deleted_edge
    return m + 3 * tri + u


def ck_like_edges(m: int, deleted_edges: set[DeletedCrossEdge] | frozenset[DeletedCrossEdge]):
    """Return actual graph edges after deleting the given cross edges."""
    n = ck_graph_info(m)["n"]
    edges: list[tuple[int, int]] = []
    for left in range(m):
        for tri in range(m - 1):
            for u in range(3):
                if (left, tri, u) not in deleted_edges:
                    edges.append((left, m + 3 * tri + u))

    for tri in range(m - 1):
        v0 = m + 3 * tri
        v1 = v0 + 1
        v2 = v0 + 2
        edges.extend([(v0, v1), (v0, v2), (v1, v2)])

    if any(i < 0 or j < 0 or i >= n or j >= n for i, j in edges):
        raise RuntimeError("internal edge construction produced an invalid endpoint")
    return edges


def _normalize_deleted_edges(m: int, deleted_edges) -> tuple[DeletedCrossEdge, ...]:
    valid_edges = set(ck_cross_edges(m))
    normalized = tuple(sorted(tuple(int(value) for value in edge) for edge in deleted_edges))
    if len(set(normalized)) != len(normalized):
        raise ValueError("deleted_edges contains duplicates")
    invalid = [edge for edge in normalized if edge not in valid_edges]
    if invalid:
        raise ValueError(f"invalid deleted CK cross edges: {invalid[:3]}")
    return normalized


def graph_hash(m: int, deleted_edges) -> str:
    deleted = _normalize_deleted_edges(m, deleted_edges)
    payload = json.dumps({"m": int(m), "deleted": deleted}, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()[:16]


def x_star_index_from_bits(bits: list[int] | tuple[int, ...]) -> int:
    return int(sum(int(bit) << index for index, bit in enumerate(bits)))


def _missing_left_masks(
    m: int, deleted_edges: tuple[DeletedCrossEdge, ...]
) -> tuple[tuple[int, int, int], ...]:
    masks = [[0, 0, 0] for _ in range(m - 1)]
    for left, tri, u in deleted_edges:
        masks[tri][u] |= 1 << left
    return tuple(tuple(row) for row in masks)


def find_unique_mis_ck_like(m: int, deleted_edges) -> dict:
    """Find the exact MIS of a CK-like deletion graph.

    The right side is a product of ``m - 1`` triangles, so every maximum
    independent set is represented by choosing no vertex or one vertex from
    each triangle, then taking all compatible left vertices.
    """
    deleted = _normalize_deleted_edges(m, deleted_edges)
    n = ck_graph_info(m)["n"]
    all_left_mask = (1 << m) - 1
    missing_masks = _missing_left_masks(m, deleted)

    @lru_cache(maxsize=None)
    def solve_suffix(tri: int, compatible_left_mask: int) -> tuple[int, int, int, int]:
        if tri == m - 1:
            return compatible_left_mask.bit_count(), 1, compatible_left_mask, 0

        best_size = -1
        best_count = 0
        best_left = 0
        best_right = 0

        choices: list[tuple[int, int | None]] = [(compatible_left_mask, None)]
        choices.extend((compatible_left_mask & missing_masks[tri][u], u) for u in range(3))

        for next_left_mask, u in choices:
            sub_size, sub_count, sub_left, sub_right = solve_suffix(tri + 1, next_left_mask)
            size = sub_size if u is None else sub_size + 1
            if u is None:
                right_mask = sub_right
            else:
                right_vertex = m + 3 * tri + u
                right_mask = sub_right | (1 << right_vertex)

            if size > best_size:
                best_size = size
                best_count = sub_count
                best_left = sub_left
                best_right = right_mask
            elif size == best_size:
                best_count = min(2, best_count + sub_count)

        return best_size, best_count, best_left, best_right

    mis_size, count_capped, left_mask, right_mask = solve_suffix(0, all_left_mask)
    bits = [0] * n
    if count_capped == 1:
        for left in range(m):
            bits[left] = (left_mask >> left) & 1
        for vertex in range(m, n):
            bits[vertex] = (right_mask >> vertex) & 1

    return {
        "is_unique": count_capped == 1,
        "mis_size": int(mis_size),
        "num_mis_capped": int(count_capped),
        "x_star_bits": bits if count_capped == 1 else None,
        "x_star_index": x_star_index_from_bits(bits) if count_capped == 1 else None,
        "x_star_left_count": int(sum(bits[:m])) if count_capped == 1 else None,
        "x_star_right_count": int(sum(bits[m:])) if count_capped == 1 else None,
    }


def deletion_probability_quotas(
    num_samples: int, probabilities: tuple[float, ...]
) -> list[int]:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not probabilities:
        raise ValueError("at least one delete probability is required")
    if any(probability <= 0.0 or probability >= 1.0 for probability in probabilities):
        raise ValueError("delete probabilities must be in (0, 1)")

    base = num_samples // len(probabilities)
    remainder = num_samples % len(probabilities)
    return [base + (1 if index < remainder else 0) for index in range(len(probabilities))]


def random_deleted_cross_edges(
    m: int, delete_probability: float, seed: int
) -> tuple[DeletedCrossEdge, ...]:
    edges = ck_cross_edges(m)
    rng = np.random.default_rng(int(seed))
    mask = rng.random(len(edges)) < float(delete_probability)
    return tuple(edge for edge, should_delete in zip(edges, mask) if bool(should_delete))


def make_graph_record(
    m: int,
    sample_id: int,
    seed: int,
    delete_probability: float,
    probability_bucket_index: int,
    attempt_index: int,
    deleted_edges,
) -> dict | None:
    deleted = _normalize_deleted_edges(m, deleted_edges)
    if not deleted:
        return None

    mis = find_unique_mis_ck_like(m, deleted)
    if not mis["is_unique"]:
        return None

    info = ck_graph_info(m)
    n = int(info["n"])
    num_edges = int(info["num_edges"]) - len(deleted)
    digest = graph_hash(m, deleted)
    return {
        "graph_id": f"cklr_m{m}_s{sample_id:02d}_{digest}",
        "graph_hash": digest,
        "m": int(m),
        "n": n,
        "sample_id": int(sample_id),
        "seed": int(seed),
        "delete_probability": float(delete_probability),
        "probability_bucket_index": int(probability_bucket_index),
        "attempt_index": int(attempt_index),
        "deleted_cross_edges": [list(edge) for edge in deleted],
        "deleted_edge_count": int(len(deleted)),
        "num_edges": num_edges,
        "L": int(n * (n + num_edges)),
        "x_star_bits": [int(bit) for bit in mis["x_star_bits"]],
        "x_star_index": int(mis["x_star_index"]),
        "mis_size": int(mis["mis_size"]),
        "x_star_left_count": int(mis["x_star_left_count"]),
        "x_star_right_count": int(mis["x_star_right_count"]),
    }


def validate_graph_record(record: dict) -> None:
    m = int(record["m"])
    info = ck_graph_info(m)
    n = int(info["n"])
    deleted = _normalize_deleted_edges(m, record["deleted_cross_edges"])
    expected_hash = graph_hash(m, deleted)
    if record["graph_hash"] != expected_hash:
        raise ValueError(f"graph_hash mismatch for {record.get('graph_id')}")
    if int(record["n"]) != n:
        raise ValueError(f"n mismatch for {record.get('graph_id')}")
    if int(record["deleted_edge_count"]) != len(deleted):
        raise ValueError(f"deleted_edge_count mismatch for {record.get('graph_id')}")
    expected_edges = int(info["num_edges"]) - len(deleted)
    if int(record["num_edges"]) != expected_edges:
        raise ValueError(f"num_edges mismatch for {record.get('graph_id')}")
    if int(record["L"]) != n * (n + expected_edges):
        raise ValueError(f"L mismatch for {record.get('graph_id')}")

    bits = [int(bit) for bit in record["x_star_bits"]]
    if len(bits) != n or any(bit not in {0, 1} for bit in bits):
        raise ValueError(f"invalid x_star_bits for {record.get('graph_id')}")
    if int(record["x_star_index"]) != x_star_index_from_bits(bits):
        raise ValueError(f"x_star_index mismatch for {record.get('graph_id')}")

    mis = find_unique_mis_ck_like(m, deleted)
    if not mis["is_unique"]:
        raise ValueError(f"graph is not unique-MIS: {record.get('graph_id')}")
    if bits != mis["x_star_bits"]:
        raise ValueError(f"x_star_bits mismatch for {record.get('graph_id')}")
    if int(record["mis_size"]) != int(mis["mis_size"]):
        raise ValueError(f"mis_size mismatch for {record.get('graph_id')}")


def validate_graph_pool(
    records: list[dict], m_min: int, m_max: int, num_samples: int
) -> None:
    expected_m = set(range(int(m_min), int(m_max) + 1))
    seen_ids: set[str] = set()
    counts = {m: 0 for m in expected_m}
    for record in records:
        validate_graph_record(record)
        graph_id = str(record["graph_id"])
        if graph_id in seen_ids:
            raise ValueError(f"duplicate graph_id: {graph_id}")
        seen_ids.add(graph_id)
        m = int(record["m"])
        if m in counts:
            counts[m] += 1
    bad = {m: count for m, count in counts.items() if count != num_samples}
    if bad:
        raise ValueError(f"graph pool sample counts do not match num_samples={num_samples}: {bad}")


def generate_graph_pool(
    m_min: int = 4,
    m_max: int = 12,
    num_samples: int = 20,
    seed: int = 20260615,
    delete_probabilities: tuple[float, ...] = DEFAULT_DELETE_PROBABILITIES,
    max_attempts_per_probability: int = 10000,
) -> list[dict]:
    if m_min < 1 or m_max < m_min:
        raise ValueError("invalid m range")
    quotas = deletion_probability_quotas(num_samples, tuple(delete_probabilities))
    rng = np.random.default_rng(int(seed))
    records: list[dict] = []

    for m in range(m_min, m_max + 1):
        sample_id = 0
        seen_hashes: set[str] = set()
        for prob_index, (probability, quota) in enumerate(zip(delete_probabilities, quotas)):
            accepted = 0
            attempts = 0
            while accepted < quota and attempts < max_attempts_per_probability:
                attempts += 1
                candidate_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                deleted = random_deleted_cross_edges(m, probability, candidate_seed)
                if not deleted:
                    continue
                digest = graph_hash(m, deleted)
                if digest in seen_hashes:
                    continue
                record = make_graph_record(
                    m=m,
                    sample_id=sample_id,
                    seed=candidate_seed,
                    delete_probability=probability,
                    probability_bucket_index=prob_index,
                    attempt_index=attempts,
                    deleted_edges=deleted,
                )
                if record is None:
                    continue
                records.append(record)
                seen_hashes.add(digest)
                sample_id += 1
                accepted += 1

            if accepted != quota:
                raise RuntimeError(
                    f"could not accept {quota} unique-MIS graphs for m={m}, "
                    f"delete_probability={probability} after {attempts} attempts"
                )

    validate_graph_pool(records, m_min, m_max, num_samples)
    return records


def write_graph_pool(records: list[dict], out_dir: str) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, GRAPH_POOL_FILENAME)
    summary_path = os.path.join(out_dir, GRAPH_POOL_SUMMARY_FILENAME)

    ordered = sorted(records, key=lambda row: (int(row["m"]), int(row["sample_id"])))
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for record in ordered:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")

    csv_rows = []
    for record in ordered:
        row = {column: record[column] for column in GRAPH_POOL_COLUMNS}
        row["deleted_cross_edges"] = json.dumps(row["deleted_cross_edges"], separators=(",", ":"))
        row["x_star_bits"] = json.dumps(row["x_star_bits"], separators=(",", ":"))
        csv_rows.append(row)
    pd.DataFrame(csv_rows, columns=GRAPH_POOL_COLUMNS).to_csv(summary_path, index=False)
    return jsonl_path, summary_path


def load_graph_pool(filename: str) -> list[dict]:
    records = []
    with open(filename, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(json.loads(stripped))
    return records


def generate_or_load_graph_pool(
    m_min: int = 4,
    m_max: int = 12,
    num_samples: int = 20,
    seed: int = 20260615,
    delete_probabilities: tuple[float, ...] = DEFAULT_DELETE_PROBABILITIES,
    max_attempts_per_probability: int = 10000,
    out_dir: str = "outputs/ck_like_random_deletion_graphs",
    force: bool = False,
) -> tuple[str, str]:
    jsonl_path = os.path.join(out_dir, GRAPH_POOL_FILENAME)
    if os.path.exists(jsonl_path) and not force:
        records = load_graph_pool(jsonl_path)
        validate_graph_pool(records, m_min, m_max, num_samples)
        _jsonl, summary_path = write_graph_pool(records, out_dir)
        print(f"loaded existing graph pool: {jsonl_path}")
        print(f"saved {summary_path}")
        return jsonl_path, summary_path

    records = generate_graph_pool(
        m_min=m_min,
        m_max=m_max,
        num_samples=num_samples,
        seed=seed,
        delete_probabilities=delete_probabilities,
        max_attempts_per_probability=max_attempts_per_probability,
    )
    jsonl_path, summary_path = write_graph_pool(records, out_dir)
    print(f"saved {jsonl_path}")
    print(f"saved {summary_path}")
    return jsonl_path, summary_path


def parse_delete_probabilities(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in text.split(",") if part.strip())
    deletion_probability_quotas(1, values)
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CK-like random cross-edge deletion graphs with unique MIS."
    )
    parser.add_argument("--m-min", type=int, default=4)
    parser.add_argument("--m-max", type=int, default=12)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument(
        "--delete-probabilities",
        type=parse_delete_probabilities,
        default=DEFAULT_DELETE_PROBABILITIES,
        help="Comma-separated probabilities, e.g. 0.03,0.05,0.07,0.09,0.11",
    )
    parser.add_argument("--max-attempts-per-probability", type=int, default=10000)
    parser.add_argument(
        "--out-dir", type=str, default="outputs/ck_like_random_deletion_graphs"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate_or_load_graph_pool(
        m_min=args.m_min,
        m_max=args.m_max,
        num_samples=args.num_samples,
        seed=args.seed,
        delete_probabilities=args.delete_probabilities,
        max_attempts_per_probability=args.max_attempts_per_probability,
        out_dir=args.out_dir,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
