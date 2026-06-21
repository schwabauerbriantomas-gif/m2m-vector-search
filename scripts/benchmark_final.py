#!/usr/bin/env python3
"""
M2M Vector Search — Final Benchmark (In-Distribution Queries)
===============================================================
Tests realistic RAG scenario: queries drawn from the same distribution
as indexed data. Measures latency, throughput, recall, and speedup.

Three backends:
  - Linear (CPU, exact brute-force): O(N·D) baseline
  - M2M HRM2 (CPU, approximate IVF): O(n_probe·M·D) 
  - CUDA (GPU, exact brute-force): GPU-accelerated

Configurations:
  - N: 1K, 10K, 50K, 100K
  - D: 384 (standard embedding dimension)
  - K: 10
  - Queries: 200, drawn from same cluster distribution as data

All numbers are real measurements on:
  - CPU: AMD Ryzen 5 3400G (4C/8T)
  - GPU: NVIDIA RTX 3090 (24GB VRAM)
  - Python 3.12, NumPy 2.4, PyTorch 2.11+cu130
"""

import gc
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.benchmark_full import (
    BenchResult, compute_ground_truth, recall_at_k, bench_cuda,
)
from m2m import SimpleVectorDB


def generate_in_distribution(n: int, dim: int, n_queries: int, seed: int = 42):
    """Generate clustered data + in-distribution queries (simulates embeddings)."""
    rng = np.random.default_rng(seed)
    n_clusters = max(10, n // 500)
    centroids = rng.standard_normal((n_clusters, dim)).astype(np.float32) * 3.0
    
    # Index data: points sampled around centroids
    labels = rng.integers(0, n_clusters, size=n)
    index = centroids[labels] + rng.standard_normal((n, dim)).astype(np.float32) * 0.5
    index /= np.linalg.norm(index, axis=1, keepdims=True)
    
    # Query data: also sampled around centroids (in-distribution)
    q_labels = rng.integers(0, n_clusters, size=n_queries)
    queries = centroids[q_labels] + rng.standard_normal((n_queries, dim)).astype(np.float32) * 0.5
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    
    return index, queries, n_clusters


def bench_m2m_indist(vectors, queries, k, gt):
    """Benchmark M2M with in-distribution queries."""
    dim = vectors.shape[1]
    n = vectors.shape[0]
    n_q = len(queries)
    
    db = SimpleVectorDB(latent_dim=dim, enable_lsh_fallback=False)
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
    
    latencies = []
    all_ids = []
    for i in range(n_q):
        t0 = time.perf_counter()
        result = db.search(queries[i], k=k, include_metadata=True)
        latencies.append((time.perf_counter() - t0) * 1000)
        all_ids.append(np.array([int(r.id) for r in result[:k]], dtype=np.int64))
    
    lat_sorted = sorted(latencies)
    approx_ids = np.array(all_ids)
    recall = round(recall_at_k(approx_ids, gt, k), 4)
    
    return BenchResult(
        config_id=f"m2m_indist_{n}_{dim}_k{k}",
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


def main():
    import torch
    
    print("=" * 76)
    print("  M2M Vector Search — Final Benchmark")
    print("  In-Distribution Queries (Realistic RAG Scenario)")
    print("=" * 76)
    print(f"  CPU:      AMD Ryzen 5 3400G (4C/8T)")
    print(f"  GPU:      {torch.cuda.get_device_name(0)}")
    print(f"  Python:   {platform.python_version()}")
    print(f"  NumPy:    {np.__version__}")
    print(f"  PyTorch:  {torch.__version__}")
    print("=" * 76)
    
    K = 10
    N_QUERIES = 200
    configs = [1_000, 10_000, 50_000, 100_000]
    DIM = 384
    
    results = []
    
    for n in configs:
        print(f"\n{'─' * 76}")
        print(f"  N={n:,}  D={DIM}  K={K}  Q={N_QUERIES} (in-distribution)")
        print(f"{'─' * 76}")
        
        index, queries, n_clusters = generate_in_distribution(n, DIM, N_QUERIES)
        print(f"  Clusters: {n_clusters}")
        
        # Ground truth
        print("  Computing ground truth...", end=" ", flush=True)
        gt = compute_ground_truth(index, queries, K)
        print("done.")
        
        # Linear
        from scripts.benchmark_full import bench_linear
        print("  [linear] ", end="", flush=True)
        r = bench_linear(index, queries, K, gt)
        r.config_id = f"linear_{n}"
        results.append(r)
        print(f"p50={r.latency_p50_ms:.1f}ms  QPS={r.qps}")
        
        # M2M
        print("  [m2m]    ", end="", flush=True)
        r = bench_m2m_indist(index, queries, K, gt)
        results.append(r)
        print(f"p50={r.latency_p50_ms:.1f}ms  QPS={r.qps}  recall={r.recall_at_k}")
        
        gc.collect()
        
        # CUDA
        print("  [cuda]   ", end="", flush=True)
        r = bench_cuda(index, queries, K, gt)
        r.config_id = f"cuda_{n}"
        results.append(r)
        print(f"p50={r.latency_p50_ms:.1f}ms  QPS={r.qps}  recall={r.recall_at_k}")
        
        # Calculate speedup
        lin = [x for x in results if x.backend == "linear" and x.n_vectors == n][-1]
        m2m_r = [x for x in results if x.backend == "m2m" and x.n_vectors == n][-1]
        cuda_r = [x for x in results if x.backend == "cuda" and x.n_vectors == n][-1]
        
        speedup_m2m = lin.latency_p50_ms / m2m_r.latency_p50_ms
        speedup_cuda = lin.latency_p50_ms / cuda_r.latency_p50_ms
        print(f"  Speedup: M2M={speedup_m2m:.1f}x  CUDA={speedup_cuda:.1f}x  (vs linear p50)")
        
        gc.collect()
    
    # Save
    out = Path(__file__).parent.parent / "benchmark_final.json"
    data = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "methodology": "In-distribution queries (clustered data, queries from same distribution)",
            "cpu": "AMD Ryzen 5 3400G (4C/8T)",
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "k": K,
            "n_queries": N_QUERIES,
            "dim": DIM,
        },
        "results": [asdict(r) for r in results],
    }
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    
    # Final summary table
    print(f"\n{'=' * 76}")
    print(f"  RESULTS SUMMARY")
    print(f"{'=' * 76}")
    print(f"{'Backend':<10} {'N':>8} {'p50(ms)':>10} {'p95(ms)':>10} {'QPS':>10} "
          f"{'Recall@10':>10} {'vs Linear':>10}")
    print("─" * 76)
    
    for n in configs:
        for backend in ["linear", "m2m", "cuda"]:
            r = [x for x in results if x.backend == backend and x.n_vectors == n][-1]
            lin = [x for x in results if x.backend == "linear" and x.n_vectors == n][-1]
            speedup = lin.latency_p50_ms / r.latency_p50_ms
            recall_s = f"{r.recall_at_k:.4f}" if r.recall_at_k else "—"
            sp_s = "1.0x" if backend == "linear" else f"{speedup:.1f}x"
            print(f"{backend:<10} {n:>8,} {r.latency_p50_ms:>10.3f} {r.latency_p95_ms:>10.3f} "
                  f"{r.qps:>10.1f} {recall_s:>10} {sp_s:>10}")
        print()
    
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
