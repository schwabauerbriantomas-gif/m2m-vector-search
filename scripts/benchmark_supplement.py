#!/usr/bin/env python3
"""
Benchmark suplementario: k=64 (comparación con benchmark original)
y datos estructurados (clusters) para medir recall realista.
"""

import gc
import json
import sys
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.benchmark_full import (
    BenchResult, compute_ground_truth, recall_at_k,
    bench_linear, bench_m2m, bench_cuda,
)


def generate_clustered_data(
    n: int, dim: int, n_clusters: int, seed: int = 42
) -> np.ndarray:
    """Generate data with real cluster structure (simulates embeddings)."""
    rng = np.random.default_rng(seed)
    centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32) * 3.0
    labels = rng.integers(0, n_clusters, size=n)
    vectors = centroids[labels] + rng.standard_normal((n, dim)).astype(np.float32) * 0.5
    # L2 normalize
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


def main():
    print("=" * 72)
    print("  M2M Benchmark — Supplement (k=64 + Structured Data)")
    print("=" * 72)

    results = []
    K = 64
    N_QUERIES = 200

    # ── Part 1: k=64 with random data ──────────────────────────────
    for n in [10_000, 50_000, 100_000]:
        dim = 384
        print(f"\n{'─'*72}")
        print(f"  k=64 Random: N={n:,}  D={dim}")
        print(f"{'─'*72}")

        rng = np.random.default_rng(42)
        index = rng.standard_normal((n, dim)).astype(np.float32)
        index /= np.linalg.norm(index, axis=1, keepdims=True)
        queries = rng.standard_normal((N_QUERIES, dim)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        gt = compute_ground_truth(index, queries, K)
        print("  GT done.")

        r = bench_linear(index, queries, K, gt)
        r.config_id = f"linear_k64_{n}"
        results.append(r)
        print(f"  [linear] p50={r.latency_p50_ms}ms QPS={r.qps}")

        r = bench_m2m(index, queries, K, gt)
        r.config_id = f"m2m_k64_{n}"
        results.append(r)
        print(f"  [m2m]    p50={r.latency_p50_ms}ms QPS={r.qps} recall={r.recall_at_k}")

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                r = bench_cuda(index, queries, K, gt)
                r.config_id = f"cuda_k64_{n}"
                results.append(r)
                print(f"  [cuda]   p50={r.latency_p50_ms}ms QPS={r.qps} recall={r.recall_at_k}")
        except Exception as e:
            print(f"  [cuda]   skipped: {e}")
        gc.collect()

    # ── Part 2: Structured data (clustered) ────────────────────────
    K2 = 10
    for n in [10_000, 50_000, 100_000]:
        dim = 384
        n_clusters = max(10, n // 500)  # ~500 vectors per cluster
        print(f"\n{'─'*72}")
        print(f"  Structured: N={n:,}  D={dim}  C={n_clusters} clusters  k={K2}")
        print(f"{'─'*72}")

        index = generate_clustered_data(n, dim, n_clusters)
        rng2 = np.random.default_rng(99)
        queries = rng2.standard_normal((N_QUERIES, dim)).astype(np.float32)
        queries /= np.linalg.norm(queries, axis=1, keepdims=True)

        gt = compute_ground_truth(index, queries, K2)
        print("  GT done.")

        r = bench_linear(index, queries, K2, gt)
        r.config_id = f"linear_struct_{n}"
        r.notes = f"clustered C={n_clusters}"
        results.append(r)
        print(f"  [linear] p50={r.latency_p50_ms}ms QPS={r.qps}")

        r = bench_m2m(index, queries, K2, gt)
        r.config_id = f"m2m_struct_{n}"
        r.notes = f"clustered C={n_clusters}"
        results.append(r)
        print(f"  [m2m]    p50={r.latency_p50_ms}ms QPS={r.qps} recall={r.recall_at_k}")

        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                r = bench_cuda(index, queries, K2, gt)
                r.config_id = f"cuda_struct_{n}"
                r.notes = f"clustered C={n_clusters}"
                results.append(r)
                print(f"  [cuda]   p50={r.latency_p50_ms}ms QPS={r.qps} recall={r.recall_at_k}")
        except Exception as e:
            print(f"  [cuda]   skipped: {e}")
        gc.collect()

    # Save
    out = Path(__file__).parent.parent / "benchmark_supplement.json"
    with open(out, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\nResults saved to {out}")

    # Table
    print(f"\n{'Config':<25} {'Backend':<8} {'N':>8} {'p50(ms)':>10} {'QPS':>10} {'Recall':>8}")
    print("─" * 75)
    for r in results:
        recall_s = f"{r.recall_at_k:.4f}" if r.recall_at_k is not None else "—"
        label = r.config_id.split("_")[0] + ("_struct" if "struct" in r.config_id else "")
        print(f"{label:<25} {r.backend:<8} {r.n_vectors:>8,} "
              f"{r.latency_p50_ms:>10.3f} {r.qps:>10.1f} {recall_s:>8}")


if __name__ == "__main__":
    main()
