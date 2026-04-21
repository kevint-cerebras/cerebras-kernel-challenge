#!/usr/bin/env cs_python
"""Host runner for the Top-K k-NN starter kernel.

This first pass launches a PE-local top-K kernel across the P x P grid and
merges the per-PE candidates on the host. It is useful for wiring up
compilation, memcpy layout, and correctness checking while the in-fabric merge
is still under construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from cerebras.sdk.runtime.sdkruntimepybind import MemcpyDataType  # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import MemcpyOrder  # pylint: disable=no-name-in-module
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime  # pylint: disable=no-name-in-module

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reference import make_all_equal
from reference import make_baseline
from reference import make_duplicates
from reference import make_k_eq_1
from reference import make_k_large
from reference import make_uneven
from reference import topk_reference


CASE_BUILDERS = {
    "baseline": make_baseline,
    "k_eq_1": make_k_eq_1,
    "k_large": make_k_large,
    "uneven": make_uneven,
    "all_equal": make_all_equal,
    "duplicates": make_duplicates,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="compile output directory")
    parser.add_argument("--case", required=True, choices=sorted(CASE_BUILDERS))
    parser.add_argument("--cmaddr", help="IP:port for CS system")
    return parser.parse_args()


def load_compile_params(build_dir: Path) -> dict[str, int]:
    with (build_dir / "out.json").open(encoding="utf-8") as infile:
        data = json.load(infile)
    return {key: int(value) for key, value in data["params"].items()}


def build_case(case_name: str) -> dict:
    return CASE_BUILDERS[case_name]()


def shard_database(D: np.ndarray, P: int, rows_per_pe: int) -> tuple[np.ndarray, np.ndarray]:
    """Pack database rows into a [P, P, rows_per_pe * d] tensor plus valid counts."""
    d_dim = D.shape[1]
    shards = np.zeros((P, P, rows_per_pe, d_dim), dtype=np.float32)
    valid_rows = np.zeros((P, P, 1), dtype=np.int32)

    pe_linear = 0
    for py in range(P):
        for px in range(P):
            start = pe_linear * rows_per_pe
            end = min(start + rows_per_pe, D.shape[0])
            count = max(end - start, 0)
            if count > 0:
                shards[py, px, :count, :] = D[start:end]
            valid_rows[py, px, 0] = count
            pe_linear += 1

    return shards.reshape(P, P, rows_per_pe * d_dim), valid_rows


def tile_query(q: np.ndarray, P: int) -> np.ndarray:
    return np.broadcast_to(q.reshape(1, 1, -1), (P, P, q.shape[0])).astype(np.float32).copy()


def merge_candidates(local_indices: np.ndarray, local_distances: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray]:
    flat_idx = local_indices.reshape(-1)
    flat_dist = local_distances.reshape(-1)

    candidates: list[tuple[float, int]] = []
    for idx, dist in zip(flat_idx.tolist(), flat_dist.tolist()):
        if idx < 0:
            continue
        candidates.append((float(dist), int(idx)))

    candidates.sort(key=lambda item: (item[0], item[1]))
    top = candidates[:K]

    indices = np.array([idx for _, idx in top], dtype=np.int32)
    distances = np.array([dist for dist, _ in top], dtype=np.float32)
    return indices, distances


def main() -> None:
    args = parse_args()
    build_dir = Path(args.name)
    params = load_compile_params(build_dir)
    case = build_case(args.case)

    P = params["P"]
    d_dim = params["d_dim"]
    rows_per_pe = params["rows_per_pe"]
    K = params["K"]

    assert case["P"] == P, f"case P={case['P']} != compiled P={P}"
    assert case["q"].shape[0] == d_dim, f"case d={case['q'].shape[0]} != compiled d={d_dim}"
    assert case["K"] == K, f"case K={case['K']} != compiled K={K}"

    D = case["D"].astype(np.float32, copy=False)
    q = case["q"].astype(np.float32, copy=False)
    oracle_idx, oracle_dist = topk_reference(D, q, K)

    shards, valid_rows = shard_database(D, P, rows_per_pe)
    q_tiled = tile_query(q, P)

    memcpy_32 = MemcpyDataType.MEMCPY_32BIT
    runner = SdkRuntime(str(build_dir), cmaddr=args.cmaddr)

    sym_D = runner.get_id("D_shard")
    sym_q = runner.get_id("q")
    sym_valid = runner.get_id("valid_rows")
    sym_idx = runner.get_id("local_indices")
    sym_dist = runner.get_id("local_distances")

    runner.load()
    runner.run()

    runner.memcpy_h2d(
        sym_D,
        shards.ravel(),
        0,
        0,
        P,
        P,
        rows_per_pe * d_dim,
        streaming=False,
        data_type=memcpy_32,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    runner.memcpy_h2d(
        sym_q,
        q_tiled.ravel(),
        0,
        0,
        P,
        P,
        d_dim,
        streaming=False,
        data_type=memcpy_32,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    runner.memcpy_h2d(
        sym_valid,
        valid_rows.ravel(),
        0,
        0,
        P,
        P,
        1,
        streaming=False,
        data_type=memcpy_32,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )

    runner.launch("compute", nonblock=False)

    local_indices = np.zeros((P, P, K), dtype=np.int32)
    local_distances = np.zeros((P, P, K), dtype=np.float32)

    runner.memcpy_d2h(
        local_indices.ravel(),
        sym_idx,
        0,
        0,
        P,
        P,
        K,
        streaming=False,
        data_type=memcpy_32,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )
    runner.memcpy_d2h(
        local_distances.ravel(),
        sym_dist,
        0,
        0,
        P,
        P,
        K,
        streaming=False,
        data_type=memcpy_32,
        order=MemcpyOrder.ROW_MAJOR,
        nonblock=False,
    )

    runner.stop()

    got_idx, got_dist = merge_candidates(local_indices, local_distances, K)

    np.testing.assert_array_equal(got_idx, oracle_idx)
    np.testing.assert_allclose(got_dist, oracle_dist, atol=1e-3, rtol=1e-3)
    print(f"PASS: {args.case}")


if __name__ == "__main__":
    main()
