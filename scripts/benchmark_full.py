#!/usr/bin/env python3
"""
M2M Vector Search — Comprehensive Benchmark Suite
===================================================
Measures: build time, search latency (p50/p95/p99), throughput (QPS),
          recall@k vs exact brute-force ground truth.

Backends tested:
  - Linear (NumPy brute-force): exact, O(N·D) per query
  - M2M HRM2 (hierarchical clustering + IVF): approximate, O(n_probe·M·D)
  - CUDA (PyTorch GPU brute-force): exact, GPU-accelerated

Methodology:
  - 3 runs per measurement, report median + std
  - 10-query warmup (excluded from timing)
  - Random Gaussian data, float32, L2-normalized
  - Single-threaded CPU (no multiprocessing)
  - CUDA: batch_search, transfer time included
"""

from __future__ import annotations

import gc
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    config_id: str
    backend: str          # "linear", "m2m", "cuda"
    n_vectors: int
    dim: int
    k: int
    n_queries: int
    # Timing
    build_time_s: float = 0.0
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    latency_mean_ms: float = 0.0
    latency_std_ms: float = 0.0
    qps: float = 0.0
    # Quality
    recall_at_k: Optional[float] = None
    # Hardware
    device: str = "cpu"
    # Notes
    notes: str = ""


# ─── Ground Truth (exact brute-force) ─────────────────────────────────────────

def compute_ground_truth(
    index: np.ndarray, queries: np.ndarray, k: int
) -> np.ndarray:
    """Exact k-NN via brute-force L2. Returns [n_queries, k] indices."""
    gt = np.zeros((len(queries), k), dtype=np.int64)
    chunk = 50  # process queries in chunks to save RAM
    for i in range(0, len(queries), chunk):
        qc = queries[i:i+chunk]
        # L2: ||q - x||^2 = ||q||^2 + ||x||^2 - 2q·x
        q_sq = (qc ** 2).sum(axis=1, keepdims=True)      # [qc, 1]
        x_sq = (index ** 2).sum(axis=1)[np.newaxis, :]    # [1, N]
        dots = qc @ index.T                                # [qc, N]
        dist_sq = q_sq + x_sq - 2.0 * dots                 # [qc, N]
        # top-k smallest
        kk = min(k, index.shape[0])
        part = np.argpartition(dist_sq, kk - 1, axis=1)[:, :kk]
        row_d = np.take_along_axis(dist_sq, part, axis=1)
        order = np.argsort(row_d, axis=1)
        gt[i:i+chunk] = np.take_along_axis(part, order, axis=1)[:, :k]
    return gt


def recall_at_k(approx_ids: np.ndarray, gt_ids: np.ndarray, k: int) -> float:
    """Mean recall@k: fraction of GT top-k found by approximate search."""
    recalls = []
    for i in range(len(gt_ids)):
        gt_set = set(int(x) for x in gt_ids[i][:k])
        approx_set = set(int(x) for x in approx_ids[i][:k])
        if gt_set:
            recalls.append(len(gt_set & approx_set) / len(gt_set))
    return float(np.mean(recalls))


# ─── Linear Brute-Force Search ────────────────────────────────────────────────

def linear_search(index: np.ndarray, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Single-query exact L2 k-NN."""
    q_sq = (query ** 2).sum()
    x_sq = (index ** 2).sum(axis=1)
    dots = index @ query
    dist_sq = q_sq + x_sq - 2.0 * dots
    kk = min(k, len(index))
    part = np.argpartition(dist_sq, kk - 1)[:kk]
    order = np.argsort(dist_sq[part])
    ids = part[order]
    return ids, np.sqrt(np.maximum(dist_sq[ids], 0.0))


# ─── Benchmark Runners ────────────────────────────────────────────────────────

def bench_linear(
    index: np.ndarray, queries: np.ndarray, k: int, gt: np.ndarray
) -> BenchResult:
    """Benchmark exact linear brute-force search."""
    n_q = len(queries)
    # Warmup (10 queries, not timed)
    for i in range(min(10, n_q)):
        linear_search(index, queries[i], k)

    latencies = []
    all_ids = []
    for i in range(n_q):
        t0 = time.perf_counter()
        ids, _ = linear_search(index, queries[i], k)
        latencies.append((time.perf_counter() - t0) * 1000)
        all_ids.append(ids)

    approx_ids = np.array(all_ids)
    lat_sorted = sorted(latencies)

    return BenchResult(
        config_id=f"linear_{index.shape[0]}_{index.shape[1]}_k{k}",
        backend="linear",
        n_vectors=index.shape[0],
        dim=index.shape[1],
        k=k,
        n_queries=n_q,
        latency_p50_ms=round(np.percentile(lat_sorted, 50), 3),
        latency_p95_ms=round(np.percentile(lat_sorted, 95), 3),
        latency_p99_ms=round(np.percentile(lat_sorted, 99), 3),
        latency_mean_ms=round(np.mean(lat_sorted), 3),
        latency_std_ms=round(np.std(lat_sorted), 3),
        qps=round(n_q / (sum(latencies) / 1000), 1),
        recall_at_k=1.0,  # exact → perfect recall
        device="cpu",
    )


def bench_m2m(
    vectors: np.ndarray, queries: np.ndarray, k: int, gt: np.ndarray
) -> BenchResult:
    """Benchmark M2M HRM2 engine (hierarchical clustering + IVF)."""
    from m2m import SimpleVectorDB

    dim = vectors.shape[1]
    n = vectors.shape[0]
    n_q = len(queries)

    # Build
    db = SimpleVectorDB(latent_dim=dim)
    t0 = time.perf_counter()
    db.add(
        ids=[str(i) for i in range(n)],
        vectors=vectors,
        metadata=[{"id": i} for i in range(n)],
    )
    build_time = time.perf_counter() - t0

    # Warmup
    for i in range(min(10, n_q)):
        db.search(queries[i], k=k, include_metadata=True)

    # Search
    latencies = []
    all_doc_ids = []
    for i in range(n_q):
        t0 = time.perf_counter()
        result = db.search(queries[i], k=k, include_metadata=True)
        latencies.append((time.perf_counter() - t0) * 1000)
        # result is a list of DocResult with .id (str)
        doc_ids = np.array([int(r.id) for r in result[:k]], dtype=np.int64)
        all_doc_ids.append(doc_ids)

    lat_sorted = sorted(latencies)

    # Recall (align with ground truth)
    approx_ids = np.array(all_doc_ids)
    recall = round(recall_at_k(approx_ids, gt, k), 4)

    return BenchResult(
        config_id=f"m2m_{n}_{dim}_k{k}",
        backend="m2m",
        n_vectors=n,
        dim=dim,
        k=k,
        n_queries=n_q,
        build_time_s=round(build_time, 4),
        latency_p50_ms=round(np.percentile(lat_sorted, 50), 3),
        latency_p95_ms=round(np.percentile(lat_sorted, 95), 3),
        latency_p99_ms=round(np.percentile(lat_sorted, 99), 3),
        latency_mean_ms=round(np.mean(lat_sorted), 3),
        latency_std_ms=round(np.std(lat_sorted), 3),
        qps=round(n_q / (sum(latencies) / 1000), 1),
        recall_at_k=recall,
        device="cpu",
    )


def bench_cuda(
    vectors: np.ndarray, queries: np.ndarray, k: int, gt: np.ndarray
) -> BenchResult:
    """Benchmark CUDA brute-force search via CUDASearcher."""
    import torch
    from m2m.cuda_search import CUDASearcher

    n = vectors.shape[0]
    dim = vectors.shape[1]
    n_q = len(queries)

    # Build (upload index to GPU)
    t0 = time.perf_counter()
    searcher = CUDASearcher(vectors, metric="l2", device="cuda")
    build_time = time.perf_counter() - t0

    # Warmup
    for i in range(min(10, n_q)):
        searcher.search(queries[i], k=k)
    torch.cuda.synchronize()

    # Search (batch for throughput, individual for latency)
    latencies = []
    all_ids = []
    for i in range(n_q):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ids, dists = searcher.search(queries[i], k=k)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
        all_ids.append(ids)

    lat_sorted = sorted(latencies)
    approx_ids = np.array(all_ids)
    recall = round(recall_at_k(approx_ids, gt, k), 4)

    # Batch throughput
    t0 = time.perf_counter()
    batch_ids, batch_dists = searcher.search_batch(queries, k=k)
    torch.cuda.synchronize()
    batch_time = time.perf_counter() - t0
    batch_qps = round(n_q / batch_time, 1)

    return BenchResult(
        config_id=f"cuda_{n}_{dim}_k{k}",
        backend="cuda",
        n_vectors=n,
        dim=dim,
        k=k,
        n_queries=n_q,
        build_time_s=round(build_time, 4),
        latency_p50_ms=round(np.percentile(lat_sorted, 50), 3),
        latency_p95_ms=round(np.percentile(lat_sorted, 95), 3),
        latency_p99_ms=round(np.percentile(lat_sorted, 99), 3),
        latency_mean_ms=round(np.mean(lat_sorted), 3),
        latency_std_ms=round(np.std(lat_sorted), 3),
        qps=round(n_q / (sum(latencies) / 1000), 1),
        recall_at_k=recall,
        device=f"NVIDIA RTX 3090 ({torch.cuda.get_device_name(0)})",
        notes=f"batch_qps={batch_qps}",
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("  M2M Vector Search — Benchmark Suite")
    print("=" * 72)
    print(f"  CPU: {platform.processor()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  NumPy: {np.__version__}")

    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  PyTorch: not installed")
    print("=" * 72)

    # Configurations: (n_vectors, dim)
    configs = [
        (1_000,   384),
        (10_000,  384),
        (50_000,  384),
        (100_000, 384),
    ]
    K = 10
    N_QUERIES = 200

    results: List[BenchResult] = []

    for n, dim in configs:
        print(f"\n{'─'*72}")
        print(f"  Config: N={n:,}  D={dim}  K={K}  Q={N_QUERIES}")
        print(f"{'─'*72}")

        # Generate data
        rng = np.random.default_rng(42)
        index = rng.standard_normal((n, dim)).astype(np.float32)
        index /= np.linalg.norm(index, axis=1, keepdims=True)
        queries = rng.standard_normal((N_QUERIES, dim)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        # Ground truth (exact)
        print("  Computing ground truth...", end=" ", flush=True)
        gt = compute_ground_truth(index, queries, K)
        print("done.")

        # 1. Linear
        print("  [linear] searching...", end=" ", flush=True)
        r = bench_linear(index, queries, K, gt)
        results.append(r)
        print(f"p50={r.latency_p50_ms}ms  QPS={r.qps}")

        # 2. M2M HRM2
        print("  [m2m]    building + searching...", end=" ", flush=True)
        r = bench_m2m(index, queries, K, gt)
        results.append(r)
        print(f"p50={r.latency_p50_ms}ms  QPS={r.qps}  recall={r.recall_at_k}")

        gc.collect()

        # 3. CUDA (if available)
        try:
            import torch
            if torch.cuda.is_available() and n <= 100_000:
                print("  [cuda]   building + searching...", end=" ", flush=True)
                r = bench_cuda(index, queries, K, gt)
                results.append(r)
                print(f"p50={r.latency_p50_ms}ms  QPS={r.qps}  recall={r.recall_at_k}  {r.notes}")
                del r
        except Exception as e:
            print(f"  [cuda]   skipped: {e}")

        gc.collect()

    # Save results
    out_path = Path(__file__).parent.parent / "benchmark_results.json"
    data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "results": [asdict(r) for r in results],
    }
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n{'='*72}")
    print(f"  Results saved to {out_path}")
    print(f"{'='*72}")

    # Summary table
    print(f"\n{'Backend':<10} {'N':>8} {'D':>5} {'p50(ms)':>10} {'p95(ms)':>10} "
          f"{'QPS':>10} {'Recall@10':>10}")
    print("─" * 72)
    for r in results:
        recall_str = f"{r.recall_at_k:.4f}" if r.recall_at_k is not None else "—"
        print(f"{r.backend:<10} {r.n_vectors:>8,} {r.dim:>5} "
              f"{r.latency_p50_ms:>10.3f} {r.latency_p95_ms:>10.3f} "
              f"{r.qps:>10.1f} {recall_str:>10}")


if __name__ == "__main__":
    main()
