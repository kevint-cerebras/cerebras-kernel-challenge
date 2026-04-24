"""NumPy reference oracle for the Top-K k-NN challenge.

Produces deterministic top-K with stable tie-breaking by original index.
This is the ground truth the grader compares submissions against.
"""

from __future__ import annotations

import numpy as np


def topk_reference(
    D: np.ndarray,        # shape (N, d), fp32
    q: np.ndarray,        # shape (d,), fp32
    K: int,
    squared: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, distances) of the K rows of D closest to q.

    Tie-breaking: on equal distance, smaller original index wins.
    Output is sorted ascending by (distance, index).

    If squared=True, returns squared L2 distances (faster, same ordering).
    """
    assert D.ndim == 2, "D must be (N, d)"
    assert q.ndim == 1, "q must be (d,)"
    assert D.shape[1] == q.shape[0], "dim mismatch"
    assert 1 <= K <= D.shape[0]

    diff = D - q[None, :]
    d2 = np.einsum("nd,nd->n", diff, diff).astype(np.float32)

    # Lexsort by (distance, index) so ties resolve to smaller index first.
    # np.lexsort sorts by the LAST key as primary → distance is last.
    idx_all = np.arange(D.shape[0], dtype=np.int32)
    order = np.lexsort((idx_all, d2))
    top_order = order[:K]

    indices = idx_all[top_order].astype(np.int32)
    distances = d2[top_order] if squared else np.sqrt(d2[top_order])
    return indices, distances.astype(np.float32)


# ---------------------------------------------------------------------------
# Deterministic test-case generator. Same seed → same inputs across machines.
# ---------------------------------------------------------------------------


def make_baseline(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    # 128 rows/PE * 32 f32 = 16 KB shard, fits comfortably in 48 KB PE SRAM.
    N, d = 2048, 32
    D = rng.standard_normal((N, d), dtype=np.float32)
    q = rng.standard_normal(d, dtype=np.float32)
    return {"name": "baseline", "D": D, "q": q, "K": 16, "P": 4}


def make_k_eq_1(seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    N, d = 1024, 32
    D = rng.standard_normal((N, d), dtype=np.float32)
    q = rng.standard_normal(d, dtype=np.float32)
    return {"name": "k=1", "D": D, "q": q, "K": 1, "P": 2}


def make_k_large(seed: int = 2) -> dict:
    rng = np.random.default_rng(seed)
    # Intentionally small: K=256 exhausts PE SRAM fast via the all_pairs
    # buffer (2*K*P*P u32). With P=2 we have 4 KB for pairs + 16 KB shard.
    N, d = 1024, 16
    D = rng.standard_normal((N, d), dtype=np.float32)
    q = rng.standard_normal(d, dtype=np.float32)
    return {"name": "k=256", "D": D, "q": q, "K": 256, "P": 2}


def make_uneven(seed: int = 3) -> dict:
    """N not divisible by P² — last shard is short."""
    rng = np.random.default_rng(seed)
    # 4x4 = 16 PEs; 1009 is prime, so unbalanced shards.
    N, d = 1009, 32
    D = rng.standard_normal((N, d), dtype=np.float32)
    q = rng.standard_normal(d, dtype=np.float32)
    return {"name": "uneven", "D": D, "q": q, "K": 16, "P": 4}


def make_all_equal() -> dict:
    """Every row identical → every tie-break must favor smaller index.
    Correct output is exactly [0, 1, ..., K-1]."""
    N, d = 1024, 16
    D = np.ones((N, d), dtype=np.float32) * 0.5
    q = np.zeros(d, dtype=np.float32)
    return {"name": "all-equal", "D": D, "q": q, "K": 16, "P": 2}


def make_duplicates(seed: int = 5) -> dict:
    """Duplicate rows at non-adjacent indices. Exposes naive merges that
    drop one of a colliding pair."""
    rng = np.random.default_rng(seed)
    N, d = 1024, 16
    D = rng.standard_normal((N, d), dtype=np.float32)
    # Make rows 100 and 500 exact duplicates; 250 and 800 too.
    D[500] = D[100]
    D[800] = D[250]
    # Pick q so rows 100 and 250 are the two closest.
    q = D[100] + np.float32(1e-3) * rng.standard_normal(d, dtype=np.float32)
    return {"name": "duplicates", "D": D, "q": q, "K": 8, "P": 2}


ALL_CASES = [
    make_baseline,
    make_k_eq_1,
    make_k_large,
    make_uneven,
    make_all_equal,
    make_duplicates,
]


if __name__ == "__main__":
    for maker in ALL_CASES:
        case = maker()
        idx, dist = topk_reference(case["D"], case["q"], case["K"])
        print(
            f"{case['name']:12s}  N={case['D'].shape[0]:>8d}  "
            f"d={case['D'].shape[1]:>4d}  K={case['K']:>3d}  "
            f"top-3 idx={idx[:3].tolist()}  top-3 dist={dist[:3].tolist()}"
        )
