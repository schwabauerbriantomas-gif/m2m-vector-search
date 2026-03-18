"""Benchmark script for M2M Vector Search."""
import time
import json
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from m2m import SimpleVectorDB

def benchmark(n_vectors=10000, dim=384, k=10, n_queries=100):
    results = {}
    
    # Generate data
    np.random.seed(42)
    vectors = np.random.randn(n_vectors, dim).astype(np.float32)
    queries = np.random.randn(n_queries, dim).astype(np.float32)
    
    # Build index
    db = SimpleVectorDB(latent_dim=dim)
    t0 = time.perf_counter()
    db.add(
        ids=[str(i) for i in range(n_vectors)],
        vectors=vectors,
        metadata=[{"id": i} for i in range(n_vectors)],
    )
    build_time = time.perf_counter() - t0
    results["build_time_s"] = round(build_time, 4)
    results["vectors_per_second"] = round(n_vectors / build_time, 1)
    
    # Search benchmark
    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        db.search(q, k=k)
        latencies.append((time.perf_counter() - t0) * 1000)
    
    latencies.sort()
    results["n_queries"] = n_queries
    results["k"] = k
    results["n_vectors"] = n_vectors
    results["dim"] = dim
    results["latency_avg_ms"] = round(np.mean(latencies), 4)
    results["latency_p50_ms"] = round(np.percentile(latencies, 50), 4)
    results["latency_p95_ms"] = round(np.percentile(latencies, 95), 4)
    results["latency_p99_ms"] = round(np.percentile(latencies, 99), 4)
    results["qps"] = round(1000 / np.mean(latencies), 1)
    
    return results

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    r = benchmark(n_vectors=n)
    print(json.dumps(r, indent=2))
    
    # Save
    out = os.path.join(os.path.dirname(__file__), '..', 'benchmark_results.json')
    with open(out, 'w') as f:
        json.dump(r, f, indent=2)
    print(f"Saved to {out}")
