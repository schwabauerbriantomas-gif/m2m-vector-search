#!/usr/bin/env python3
"""M2M Auto-Benchmark — Quick performance profiling for M2M Vector Search.

Usage:
    python scripts/auto_benchmark.py [--dim 640] [--max-n 50000] [--k 10]

Outputs a markdown table with latency and QPS for different N sizes.
"""
import argparse
import sys
import time
import numpy as np

sys.path.insert(0, "src")

from m2m.config import M2MConfig
from m2m.splats import SplatStore


def benchmark_find_neighbors(n_splats, dim, k, n_queries=10, n_warmup=3):
    """Benchmark find_neighbors for a given number of splats."""
    config = M2MConfig.simple()
    config.latent_dim = dim
    config.max_splats = n_splats + 1000

    store = SplatStore(config)

    # Generate and add splats
    vecs = np.random.randn(n_splats, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs /= norms
    store.add_splat(vecs)
    store.build_index()

    # Generate queries
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    norms = np.linalg.norm(queries, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    queries /= norms

    # Warmup
    for i in range(n_warmup):
        store.find_neighbors(queries[i % n_queries], k=k)

    # Benchmark
    times = []
    for i in range(n_queries):
        t0 = time.perf_counter()
        store.find_neighbors(queries[i], k=k)
        dt = (time.perf_counter() - t0) * 1000  # ms
        times.append(dt)

    avg = np.mean(times)
    p50 = np.percentile(times, 50)
    p95 = np.percentile(times, 95)
    p99 = np.percentile(times, 99)
    qps = 1000.0 / avg

    return {
        "n": n_splats,
        "dim": dim,
        "k": k,
        "avg_ms": avg,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "qps": qps,
    }


def main():
    parser = argparse.ArgumentParser(description="M2M Auto-Benchmark")
    parser.add_argument("--dim", type=int, default=640, help="Embedding dimension")
    parser.add_argument("--k", type=int, default=10, help="K neighbors")
    parser.add_argument("--sizes", type=int, nargs="+",
                        default=[100, 500, 1000, 5000, 10000, 20000, 50000],
                        help="N sizes to benchmark")
    args = parser.parse_args()

    print(f"M2M Auto-Benchmark (dim={args.dim}, k={args.k})")
    print(f"Sizes: {args.sizes}")
    print()

    results = []
    for n in args.sizes:
        print(f"  Benchmarking N={n}...", end=" ", flush=True)
        r = benchmark_find_neighbors(n, args.dim, args.k)
        results.append(r)
        print(f"{r['avg_ms']:.2f}ms ({r['qps']:.0f} QPS)")

    # Output as markdown table
    print("\n## Benchmark Results")
    print()
    print("| N | Dim | K | Avg (ms) | P50 (ms) | P95 (ms) | P99 (ms) | QPS |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['n']} | {r['dim']} | {r['k']} | {r['avg_ms']:.2f} | {r['p50_ms']:.2f} | {r['p95_ms']:.2f} | {r['p99_ms']:.2f} | {r['qps']:.0f} |")

    # Save results
    output_path = "benchmark_auto_results.json"
    import json
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
