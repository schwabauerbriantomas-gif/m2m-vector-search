#!/usr/bin/env python3
"""
Comprehensive benchmark: M2M HRM2 vs CPU Linear Scan vs CUDA Brute-Force.

Tests three search methods across multiple dataset sizes:
  1. M2M HRM2 (CPU)     — hierarchical routing with mixture models
  2. CPU Linear Scan     — exact brute-force k-NN via numpy
  3. CUDA Brute-Force    — GPU-accelerated exact k-NN via PyTorch

Key design: Build phase is separated from query phase.
Only per-query search latency is measured.

All data is real, measured on the specified hardware. No estimates.
"""

import argparse
import gc
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore", message=".*silhouette.*")

# ─── Data Generation ──────────────────────────────────────────────────────────

def generate_clustered_data(n, dim=384, n_clusters=20, seed=42):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, dim).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8

    cluster_assignments = rng.randint(0, n_clusters, size=n)
    vectors = np.zeros((n, dim), dtype=np.float32)
    for i in range(n):
        c = cluster_assignments[i]
        vectors[i] = centers[c] + rng.randn(dim) * 0.3

    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8

    n_queries = 100
    query_clusters = rng.randint(0, n_clusters, size=n_queries)
    queries = np.zeros((n_queries, dim), dtype=np.float32)
    for i in range(n_queries):
        c = query_clusters[i]
        queries[i] = centers[c] + rng.randn(dim) * 0.3

    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8
    return vectors, queries


def compute_ground_truth(vectors, queries, k=10):
    vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    queries_norm = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-8)
    gt = np.zeros((len(queries), k), dtype=np.int64)
    for i in range(len(queries)):
        sims = vectors_norm @ queries_norm[i]
        gt[i] = np.argsort(-sims)[:k]
    return gt


def recall_at_k(found_indices, ground_truth, k=10):
    total_hits = 0
    total_possible = 0
    for i in range(len(ground_truth)):
        true_set = set(ground_truth[i][:k].tolist())
        found_set = set(found_indices[i][:k].tolist())
        total_hits += len(true_set & found_set)
        total_possible += k
    return total_hits / total_possible if total_possible > 0 else 0.0


# ─── Index Wrappers (build once, query many) ─────────────────────────────────

class CPULinearIndex:
    """Pre-computed normalized index for CPU brute-force."""
    def __init__(self, vectors):
        self.vectors = vectors.astype(np.float32)
        self.norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        self.normalized = vectors / self.norms

    def search(self, query, k=10):
        q = query / (np.linalg.norm(query) + 1e-8)
        sims = self.normalized @ q
        idx = np.argpartition(-sims, k)[:k]
        order = np.argsort(-sims[idx])
        return idx[order]

    def search_batch(self, queries, k=10):
        results = np.zeros((len(queries), k), dtype=np.int64)
        for i in range(len(queries)):
            results[i] = self.search(queries[i], k)
        return results


class CUDAIndex:
    """GPU brute-force index using PyTorch CUDA tensors."""
    def __init__(self, vectors):
        import torch
        self.device = torch.device("cuda")
        self.t = torch.from_numpy(vectors.astype(np.float32)).to(self.device)
        self.norms = self.t.norm(dim=1, keepdim=True) + 1e-8
        self.normalized = self.t / self.norms
        self.k = 10

    def search(self, query, k=10):
        import torch
        q = torch.from_numpy(query.astype(np.float32)).to(self.device)
        q = q / (q.norm() + 1e-8)
        sims = self.normalized @ q
        topk = torch.topk(sims, k=min(k, sims.shape[0]))
        return topk.indices.cpu().numpy()

    def search_batch(self, queries, k=10):
        import torch
        q = torch.from_numpy(queries.astype(np.float32)).to(self.device)
        q = q / (q.norm(dim=1, keepdim=True) + 1e-8)
        sims = q @ self.normalized.T
        topk = torch.topk(sims, k=min(k, sims.shape[1]), dim=1)
        return topk.indices.cpu().numpy()


class M2MIndex:
    """M2M SimpleVectorDB wrapper — build once, query many."""
    def __init__(self, vectors, dim=384):
        from m2m import SimpleVectorDB
        self.db = SimpleVectorDB(latent_dim=dim)
        self.db.add(vectors=vectors, ids=[str(i) for i in range(len(vectors))])

    def search(self, query, k=10):
        res = self.db.search(query, k=k, include_metadata=True)
        return np.array([int(r.id) for r in res[:k]], dtype=np.int64)

    def search_batch(self, queries, k=10):
        results = np.zeros((len(queries), k), dtype=np.int64)
        for i in range(len(queries)):
            results[i] = self.search(queries[i], k)
        return results


# ─── Timing ───────────────────────────────────────────────────────────────────

def percentile(data, p):
    if not data:
        return 0.0
    s = sorted(data)
    idx = min(int(len(s) * p / 100), len(s) - 1)
    return s[idx]


def benchmark_index(index, queries, k, name=""):
    """Time per-query search latency (index already built)."""
    # Warmup
    try:
        index.search(queries[0], k=k)
    except Exception:
        pass
    try:
        import torch; torch.cuda.synchronize()
    except Exception:
        pass

    latencies = []
    for i in range(len(queries)):
        try:
            import torch; torch.cuda.synchronize()
        except Exception:
            pass
        t0 = time.perf_counter()
        index.search(queries[i], k=k)
        try:
            import torch; torch.cuda.synchronize()
        except Exception:
            pass
        latencies.append((time.perf_counter() - t0) * 1000)

    # Recall via batch
    found = index.search_batch(queries, k=k)
    return {
        "p50": round(percentile(latencies, 50), 2),
        "p95": round(percentile(latencies, 95), 2),
        "p99": round(percentile(latencies, 99), 2),
        "qps": round(1000.0 / np.mean(latencies), 1) if np.mean(latencies) > 0 else 0,
    }, found


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="1000,10000,50000,100000")
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", default="benchmark_comprehensive.json")
    args = parser.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    dim, k = args.dim, args.k

    import platform, multiprocessing

    system_info = {
        "timestamp": datetime.now().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_cores": multiprocessing.cpu_count(),
        "dim": dim,
        "k": k,
        "n_queries": 100,
        "data_type": "synthetic_gaussian_clusters",
        "n_clusters": 20,
        "query_distribution": "in-distribution",
    }

    has_cuda = False
    gpu_name = "N/A"
    try:
        import torch
        if torch.cuda.is_available():
            has_cuda = True
            gpu_name = torch.cuda.get_device_name(0)
            system_info["gpu"] = gpu_name
            system_info["gpu_memory_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            system_info["cuda_version"] = torch.version.cuda
            system_info["pytorch_version"] = torch.__version__
    except ImportError:
        pass

    system_info["has_cuda"] = has_cuda

    print(f"\n{'='*70}")
    print(f"  M2M COMPREHENSIVE BENCHMARK")
    print(f"  CPU: {system_info['cpu_cores']} cores | GPU: {gpu_name}")
    print(f"  dim={dim}, k={k}, queries=100 (in-distribution)")
    print(f"  Build timed separately from search")
    print(f"{'='*70}\n")

    results = {"system": system_info, "results": []}

    for size in sizes:
        print(f"\n{'─'*55}")
        print(f"  N = {size:,}")
        print(f"{'─'*55}")

        vectors, queries = generate_clustered_data(size, dim=dim)
        ground_truth = compute_ground_truth(vectors, queries, k=k)
        entry = {"n": size}

        # ── 1. CPU Linear ──
        print(f"\n  [CPU Linear] building...", end=" ", flush=True)
        t0 = time.perf_counter()
        cpu_idx = CPULinearIndex(vectors)
        cpu_build = time.perf_counter() - t0
        print(f"{cpu_build:.2f}s")

        cpu_stats, cpu_found = benchmark_index(cpu_idx, queries, k, "CPU")
        cpu_recall = recall_at_k(cpu_found, ground_truth, k)
        cpu_stats.update({"recall": round(cpu_recall, 4), "build_s": round(cpu_build, 2)})
        entry["cpu_linear"] = cpu_stats
        print(f"    p50={cpu_stats['p50']}ms  QPS={cpu_stats['qps']}  recall={cpu_stats['recall']}")

        del cpu_idx
        gc.collect()

        # ── 2. M2M HRM2 ──
        print(f"  [M2M HRM2] building...", end=" ", flush=True)
        t0 = time.perf_counter()
        m2m_idx = M2MIndex(vectors, dim=dim)
        m2m_build = time.perf_counter() - t0
        print(f"{m2m_build:.2f}s")

        m2m_stats, m2m_found = benchmark_index(m2m_idx, queries, k, "M2M")
        m2m_recall = recall_at_k(m2m_found, ground_truth, k)
        m2m_stats.update({"recall": round(m2m_recall, 4), "build_s": round(m2m_build, 2)})
        entry["m2m_hrm2"] = m2m_stats
        print(f"    p50={m2m_stats['p50']}ms  QPS={m2m_stats['qps']}  recall={m2m_stats['recall']}")

        del m2m_idx
        gc.collect()

        # ── 3. CUDA ──
        if has_cuda:
            print(f"  [CUDA GPU] building...", end=" ", flush=True)
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass

            t0 = time.perf_counter()
            cuda_idx = CUDAIndex(vectors)
            cuda_build = time.perf_counter() - t0
            print(f"{cuda_build:.2f}s")

            cuda_stats, cuda_found = benchmark_index(cuda_idx, queries, k, "CUDA")
            cuda_recall = recall_at_k(cuda_found, ground_truth, k)
            cuda_stats.update({"recall": round(cuda_recall, 4), "build_s": round(cuda_build, 2)})
            entry["cuda_bruteforce"] = cuda_stats
            print(f"    p50={cuda_stats['p50']}ms  QPS={cuda_stats['qps']}  recall={cuda_stats['recall']}")

            del cuda_idx
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass
            gc.collect()

        # Speedups
        cpu_qps = cpu_stats["qps"]
        m2m_qps = m2m_stats["qps"]
        entry["speedup_m2m_vs_cpu"] = round(m2m_qps / cpu_qps, 2) if cpu_qps > 0 else 0
        if "cuda_bruteforce" in entry:
            cuda_qps = entry["cuda_bruteforce"]["qps"]
            entry["speedup_cuda_vs_cpu"] = round(cuda_qps / cpu_qps, 2) if cpu_qps > 0 else 0
            entry["speedup_cuda_vs_m2m"] = round(cuda_qps / m2m_qps, 2) if m2m_qps > 0 else 0

        results["results"].append(entry)

        # Comparison table
        print(f"\n  ┌{'─'*55}┐")
        print(f"  │ N={size:>7,}  {'CPU Linear':>13} {'M2M HRM2':>13}", end="")
        if "cuda_bruteforce" in entry:
            print(f" {'CUDA GPU':>13}", end="")
        print(" │")
        print(f"  │ p50 (ms)  {cpu_stats['p50']:>13.2f} {m2m_stats['p50']:>13.2f}", end="")
        if "cuda_bruteforce" in entry:
            print(f" {entry['cuda_bruteforce']['p50']:>13.3f}", end="")
        print(" │")
        print(f"  │ QPS       {cpu_stats['qps']:>13.1f} {m2m_stats['qps']:>13.1f}", end="")
        if "cuda_bruteforce" in entry:
            print(f" {entry['cuda_bruteforce']['qps']:>13.1f}", end="")
        print(" │")
        print(f"  │ recall@{k}  {cpu_stats['recall']:>13.4f} {m2m_stats['recall']:>13.4f}", end="")
        if "cuda_bruteforce" in entry:
            print(f" {entry['cuda_bruteforce']['recall']:>13.4f}", end="")
        print(" │")
        print(f"  └{'─'*55}┘")

    # Save
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved to {args.output}")

    # Final summary
    print(f"\n{'='*70}")
    print("  SUMMARY (QPS comparison)")
    print(f"{'='*70}")
    for e in results["results"]:
        n = e["n"]
        print(f"  N={n:>7,}  CPU: {e['cpu_linear']['qps']:>8.1f} QPS  "
              f"M2M: {e['m2m_hrm2']['qps']:>8.1f} QPS ({e['speedup_m2m_vs_cpu']}x vs CPU)", end="")
        if "cuda_bruteforce" in e:
            print(f"  GPU: {e['cuda_bruteforce']['qps']:>8.1f} QPS ({e['speedup_cuda_vs_cpu']}x vs CPU)")
        else:
            print()
    print()


if __name__ == "__main__":
    main()
