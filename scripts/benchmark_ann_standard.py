#!/usr/bin/env python3
"""
M2M Vector Search — Standard ANN-Benchmarks Suite
==================================================
Uses the same datasets as ann-benchmarks.com (FAISS, HNSW, Annoy, ScaNN, etc.)
for direct cross-comparison with established vector search systems.

Datasets (from /mnt/d/splatdb/bench-data/ann/):
  - SIFT-128-EUCLIDEAN: 1,000,000 vectors, 128D, L2 distance
  - GLOVE-100-ANGULAR:  1,183,514 vectors, 100D, cosine distance
  - NYTIMES-256-ANGULAR: 290,000 vectors, 256D, cosine distance

Each dataset includes precomputed ground truth (neighbors + distances for 10K queries).

The M2M engine is essentially an IVF (Inverted File Index) — same algorithmic
family as FAISS IndexIVFFlat. This benchmark sweeps n_probe to show the
recall/latency tradeoff curve, exactly as ann-benchmarks does.

Usage:
    python scripts/benchmark_ann_standard.py
    python scripts/benchmark_ann_standard.py --datasets nytimes
    python scripts/benchmark_ann_standard.py --queries 1000 --n_probes 5 10 20
"""

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

from m2m import SimpleVectorDB

BENCH_DIR = Path("/mnt/d/splatdb/bench-data/ann")

DATASETS = {
    "sift": {
        "file": "sift-128-euclidean.hdf5",
        "metric": "euclidean",
        "dim": 128,
        "n_train": 1_000_000,
    },
    "glove": {
        "file": "glove-100-angular.hdf5",
        "metric": "angular",
        "dim": 100,
        "n_train": 1_183_514,
    },
    "nytimes": {
        "file": "nytimes-256-angular.hdf5",
        "metric": "angular",
        "dim": 256,
        "n_train": 290_000,
    },
}


@dataclass
class ANNResult:
    dataset: str
    metric: str
    n_train: int
    dim: int
    n_queries: int
    k: int
    backend: str
    n_probe: int = 0  # HRM2 clusters probed (0 for linear/cuda)
    n_clusters: int = 0  # total HRM2 clusters
    build_time_s: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    qps: float = 0.0
    recall_at_10: float = 0.0


def load_dataset(name: str, max_train: int = 0) -> dict:
    """Load an ANN-Benchmarks dataset."""
    info = DATASETS[name]
    path = BENCH_DIR / info["file"]

    if not path.exists():
        print(f"ERROR: Dataset not found: {path}")
        print(f"Download from: http://ann-benchmarks.com/{info['file']}")
        sys.exit(1)

    print(f"Loading {name} from {path}...", flush=True)
    t0 = time.perf_counter()
    with h5py.File(path, "r") as f:
        train = np.array(f["train"], dtype=np.float32)
        test = np.array(f["test"], dtype=np.float32)
        neighbors = np.array(f["neighbors"], dtype=np.int64)
        distances = np.array(f["distances"], dtype=np.float32)

    load_time = time.perf_counter() - t0

    # Optionally subsample training set
    if max_train > 0 and len(train) > max_train:
        train = train[:max_train]
        print(f"  WARNING: Subsampled to {max_train} train vectors")

    print(
        f"  train={train.shape}, test={test.shape}, "
        f"neighbors={neighbors.shape} ({load_time:.1f}s)",
        flush=True,
    )

    return {
        "name": name,
        "metric": info["metric"],
        "train": train,
        "test": test,
        "neighbors": neighbors[:len(train)],  # align GT with subsampled train
        "distances": distances,
    }


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2-normalize vectors (for angular/cosine datasets)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def compute_recall(approx_indices: np.ndarray, gt_neighbors: np.ndarray, k: int) -> float:
    """Compute recall@k."""
    correct = 0
    total = 0
    for i in range(len(approx_indices)):
        gt_set = set(gt_neighbors[i, :k])
        approx_set = set(int(x) for x in approx_indices[i, :k])
        correct += len(gt_set & approx_set)
        total += k
    return correct / total if total > 0 else 0.0


def bench_linear(
    train: np.ndarray,
    queries: np.ndarray,
    gt_neighbors: np.ndarray,
    k: int,
    metric: str,
    dataset_name: str,
) -> ANNResult:
    """Brute-force linear scan baseline."""
    n_q = len(queries)
    dim = train.shape[1]
    n = len(train)

    # For angular, normalize both sides
    if metric == "angular":
        db = l2_normalize(train)
        qs = l2_normalize(queries)
    else:
        db = train
        qs = queries

    # Note: zero-norm vectors have dot product 0 (rank last in cosine).
    # No filtering needed for cosine-based linear search.

    # Warmup
    for i in range(min(5, n_q)):
        _ = db @ qs[i]

    latencies = []
    all_indices = np.zeros((n_q, k), dtype=np.int64)
    for i in range(n_q):
        t0 = time.perf_counter()
        if metric == "angular":
            sims = db @ qs[i]
            idx = np.argpartition(-sims, k)[:k]
        else:
            dists = np.sum((db - qs[i]) ** 2, axis=1)
            idx = np.argpartition(dists, k)[:k]
        latencies.append((time.perf_counter() - t0) * 1000)
        all_indices[i] = idx

    lat_sorted = sorted(latencies)
    recall10 = compute_recall(all_indices, gt_neighbors, min(k, 10))

    return ANNResult(
        dataset=dataset_name,
        metric=metric,
        n_train=n,
        dim=dim,
        n_queries=n_q,
        k=k,
        backend="linear",
        p50_ms=np.percentile(lat_sorted, 50),
        p95_ms=np.percentile(lat_sorted, 95),
        p99_ms=np.percentile(lat_sorted, 99),
        qps=n_q / (sum(latencies) / 1000),
        recall_at_10=recall10,
    )


def bench_m2m(
    train: np.ndarray,
    queries: np.ndarray,
    gt_neighbors: np.ndarray,
    k: int,
    metric: str,
    dataset_name: str,
    n_probe: int,
    max_train: int,
) -> ANNResult:
    """M2M HRM2 engine benchmark with specified n_probe."""
    n_q = len(queries)
    dim = train.shape[1]

    # For angular, normalize before indexing
    if metric == "angular":
        data = l2_normalize(train)
        qs = l2_normalize(queries)
    else:
        data = train
        qs = queries

    # Filter out zero-norm vectors (all-zeros in source data).
    # These have L2²=1.0 to any query, corrupting L2 ranking on the unit sphere.
    if metric == "angular":
        norms = np.linalg.norm(data, axis=1)
        valid_mask = norms > 0.01
        n_zero = (~valid_mask).sum()
        if n_zero > 0:
            print(f"  [warn] Filtering {n_zero} zero-norm vectors")
            # Keep only valid vectors, but preserve original IDs
            valid_indices = np.where(valid_mask)[0]
            data = data[valid_mask]
            # Remap GT neighbors: any GT pointing to a zero vector is unmatchable
            valid_set = set(valid_indices.tolist())
        else:
            valid_indices = np.arange(len(train))
            valid_set = set(range(len(train)))
    else:
        valid_indices = np.arange(len(train))
        valid_set = set(range(len(train)))

    n = len(data)

    # Create DB with proper capacity and n_probe
    db = SimpleVectorDB(
        latent_dim=dim,
        enable_lsh_fallback=False,
        max_splats=n,
        n_probe=n_probe,
    )

    t0 = time.perf_counter()
    db.add(
        ids=[str(valid_indices[i]) for i in range(n)],
        vectors=data,
    )
    build_time = time.perf_counter() - t0

    # Get actual cluster count
    n_clusters = db.engine.m2m.splats.engine.n_coarse

    # Use pure L2 ranking for fair ANN comparison (matches FAISS IVFFlat)
    db.engine.m2m.splats.rank_by = "l2"

    # Warmup
    for i in range(min(10, n_q)):
        db.search(qs[i], k=k, include_metadata=True)

    latencies = []
    all_indices = np.zeros((n_q, k), dtype=np.int64)
    for i in range(n_q):
        t0 = time.perf_counter()
        result = db.search(qs[i], k=k, include_metadata=True)
        latencies.append((time.perf_counter() - t0) * 1000)
        for j, r in enumerate(result[:k]):
            all_indices[i, j] = int(r.id)

    lat_sorted = sorted(latencies)

    # Compute recall only against valid (non-zero) GT neighbors
    recall_correct = 0
    recall_total = 0
    for i in range(n_q):
        gt_set = set(int(x) for x in gt_neighbors[i, :k])
        approx_set = set(int(x) for x in all_indices[i, :k])
        # Only count GT neighbors that are in the valid set
        valid_gt = gt_set & valid_set
        recall_correct += len(valid_gt & approx_set)
        recall_total += len(valid_gt)
    recall10 = recall_correct / recall_total if recall_total > 0 else 0.0

    del db

    return ANNResult(
        dataset=dataset_name,
        metric=metric,
        n_train=n,
        dim=dim,
        n_queries=n_q,
        k=k,
        backend=f"m2m_ivf",
        n_probe=n_probe,
        n_clusters=n_clusters,
        build_time_s=build_time,
        p50_ms=np.percentile(lat_sorted, 50),
        p95_ms=np.percentile(lat_sorted, 95),
        p99_ms=np.percentile(lat_sorted, 99),
        qps=n_q / (sum(latencies) / 1000),
        recall_at_10=recall10,
    )


def bench_cuda(
    train: np.ndarray,
    queries: np.ndarray,
    gt_neighbors: np.ndarray,
    k: int,
    metric: str,
    dataset_name: str,
) -> ANNResult | None:
    """CUDA GPU brute-force benchmark."""
    try:
        import torch
    except ImportError:
        print("  [cuda] PyTorch not available, skipping")
        return None

    if not torch.cuda.is_available():
        print("  [cuda] CUDA not available, skipping")
        return None

    n_q = len(queries)
    dim = train.shape[1]
    n = len(train)

    if metric == "angular":
        data = l2_normalize(train)
        qs = l2_normalize(queries)
    else:
        data = train
        qs = queries

    # Note: zero-norm vectors have dot product 0 (rank last in cosine).
    # No filtering needed for cosine-based CUDA search.

    db_gpu = torch.from_numpy(data).cuda()
    qs_gpu = torch.from_numpy(qs).cuda()

    # Warmup
    for i in range(min(10, n_q)):
        if metric == "angular":
            sims = db_gpu @ qs_gpu[i]
            _ = torch.topk(sims, k)
        else:
            dists = torch.sum((db_gpu - qs_gpu[i]) ** 2, dim=1)
            _ = torch.topk(-dists, k)
    torch.cuda.synchronize()

    latencies = []
    all_indices = np.zeros((n_q, k), dtype=np.int64)
    for i in range(n_q):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if metric == "angular":
            sims = db_gpu @ qs_gpu[i]
            vals, idx = torch.topk(sims, k)
        else:
            dists = torch.sum((db_gpu - qs_gpu[i]) ** 2, dim=1)
            vals, idx = torch.topk(-dists, k)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000)
        all_indices[i] = idx.cpu().numpy()

    lat_sorted = sorted(latencies)
    recall10 = compute_recall(all_indices, gt_neighbors, min(k, 10))

    del db_gpu, qs_gpu
    torch.cuda.empty_cache()

    return ANNResult(
        dataset=dataset_name,
        metric=metric,
        n_train=n,
        dim=dim,
        n_queries=n_q,
        k=k,
        backend="cuda_gpu",
        p50_ms=np.percentile(lat_sorted, 50),
        p95_ms=np.percentile(lat_sorted, 95),
        p99_ms=np.percentile(lat_sorted, 99),
        qps=n_q / (sum(latencies) / 1000),
        recall_at_10=recall10,
    )


def run_benchmark(
    dataset_name: str, n_queries: int, k: int, n_probes: list, max_train: int
):
    """Run full benchmark on one dataset."""
    data = load_dataset(dataset_name, max_train=max_train)
    train = data["train"]
    queries = data["test"]
    gt = data["neighbors"]
    metric = data["metric"]

    # Limit number of queries
    if n_queries < len(queries):
        queries = queries[:n_queries]
        gt = gt[:n_queries]

    results = []

    # ── Linear ──
    print(f"\n{'─'*70}")
    print(f"  [{dataset_name.upper()}] LINEAR (brute-force)  N={len(train):,}")
    print(f"{'─'*70}")
    r = bench_linear(train, queries, gt, k, metric, dataset_name)
    results.append(r)
    print(f"  p50={r.p50_ms:.1f}ms  QPS={r.qps:.1f}  R@10={r.recall_at_10:.4f}",
          flush=True)

    # ── M2M IVF (sweep n_probe) ──
    for np_val in n_probes:
        print(f"\n  [{dataset_name.upper()}] M2M IVF  n_probe={np_val}  N={len(train):,}")
        r = bench_m2m(train, queries, gt, k, metric, dataset_name, np_val, max_train)
        results.append(r)
        print(f"  clusters={r.n_clusters}  build={r.build_time_s:.1f}s  "
              f"p50={r.p50_ms:.1f}ms  QPS={r.qps:.1f}  "
              f"R@10={r.recall_at_10:.4f}",
              flush=True)

    # ── CUDA ──
    print(f"\n  [{dataset_name.upper()}] CUDA GPU")
    r = bench_cuda(train, queries, gt, k, metric, dataset_name)
    if r is not None:
        results.append(r)
        print(f"  p50={r.p50_ms:.2f}ms  QPS={r.qps:.1f}  R@10={r.recall_at_10:.4f}",
              flush=True)

    return results


def main():
    parser = argparse.ArgumentParser(description="ANN-Benchmarks for M2M")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["nytimes"],
        choices=list(DATASETS.keys()),
        help="Datasets to benchmark",
    )
    parser.add_argument("--queries", type=int, default=1000,
                        help="Number of queries (max 10000)")
    parser.add_argument("--k", type=int, default=10,
                        help="k for recall@k")
    parser.add_argument("--n_probes", nargs="+", type=int,
                        default=[5, 10, 15, 20],
                        help="n_probe values to sweep for M2M IVF")
    parser.add_argument("--max_train", type=int, default=0,
                        help="Max training vectors (0 = use all)")
    parser.add_argument("--output", type=str,
                        default="benchmark_ann_results.json",
                        help="Output JSON file")
    args = parser.parse_args()

    import multiprocessing
    cpu = platform.processor() or "Unknown CPU"

    print("=" * 70)
    print("  M2M Vector Search — Standard ANN-Benchmarks")
    print("=" * 70)
    print(f"  CPU:      {cpu} ({multiprocessing.cpu_count()} threads)")
    print(f"  Python:   {platform.python_version()}")
    print(f"  NumPy:    {np.__version__}")

    gpu_name = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  GPU:      {gpu_name}")
            print(f"  PyTorch:  {torch.__version__}")
        else:
            print("  GPU:      Not available")
    except ImportError:
        print("  PyTorch:  Not available")

    print(f"  Queries:  {args.queries}")
    print(f"  K:        {args.k}")
    print(f"  n_probes: {args.n_probes}")
    print(f"  Datasets: {', '.join(args.datasets)}")
    print("=" * 70)

    all_results = []
    for ds in args.datasets:
        results = run_benchmark(ds, args.queries, args.k, args.n_probes, args.max_train)
        all_results.extend(results)

    # ── Summary table ──
    print("\n" + "=" * 100)
    print("  RESULTS SUMMARY — Standard ANN-Benchmarks")
    print("=" * 100)
    print(f"{'Dataset':<12} {'Backend':<14} {'N':>10} {'Dim':>5} {'nPrb':>5} "
          f"{'p50(ms)':>8} {'p99(ms)':>8} {'QPS':>8} {'R@10':>7}")
    print("─" * 100)

    for r in all_results:
        np_str = str(r.n_probe) if r.n_probe > 0 else "-"
        print(f"{r.dataset:<12} {r.backend:<14} {r.n_train:>10,} {r.dim:>5} "
              f"{np_str:>5} {r.p50_ms:>8.2f} {r.p99_ms:>8.2f} {r.qps:>8.1f} "
              f"{r.recall_at_10:>7.4f}")

    # Save JSON
    output = {
        "hardware": {
            "cpu": cpu,
            "gpu": gpu_name,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "params": {
            "queries": args.queries,
            "k": args.k,
            "n_probes": args.n_probes,
        },
        "results": [asdict(r) for r in all_results],
    }
    output_path = Path(__file__).parent.parent / args.output
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
