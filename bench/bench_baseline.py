#!/usr/bin/env python3
"""Baseline benchmark m2m-vector-search — metodologia replicada de scripts/benchmark_ann_standard.py
Datasets: sintetico clustered in-distribution (como benchmark_final.json) + GloVe-like uniforme.
Mide: build time, QPS, recall@10, p50/p95 para m2m (l2 y gaussian) y linear scan referencia.
Tambien: costo del default add() con silhouette (LSH fallback check) y batch API.
Salida: JSON en bench/baseline.json
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "m2m-vector-search", "src"))

from m2m import SimpleVectorDB  # noqa: E402

SEED = 42
K = 10
N_QUERIES = 200
DIM = 384
OUT = os.path.join(os.path.dirname(__file__), "baseline.json")

# ---------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------

def gen_clustered(n, dim, n_clusters=50, seed=SEED):
    """Clustered in-distribution: replica la metodologia de benchmark_final.json."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, dim).astype(np.float32)
    assign = rng.randint(0, n_clusters, size=n)
    data = centers[assign] + 0.35 * rng.randn(n, dim).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True) + 1e-9
    queries = centers[rng.randint(0, n_clusters, size=N_QUERIES)] + 0.35 * rng.randn(
        N_QUERIES, dim
    ).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9
    return data.astype(np.float32), queries.astype(np.float32)


def gen_uniform(n, dim, seed=SEED):
    """Distribucion uniforme (peor caso para IVF) — como GloVe en parte."""
    rng = np.random.RandomState(seed + 1)
    data = rng.randn(n, dim).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True) + 1e-9
    queries = rng.randn(N_QUERIES, dim).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9
    return data, queries


def ground_truth(data, queries, k=K):
    """GT exacto por brute force (dist^2 = 2-2cos sobre esfera; menor = más cercano)."""
    d2 = queries @ data.T
    d2 = 2.0 - 2.0 * d2  # dist^2 sobre esfera unidad
    return np.argsort(d2, axis=1)[:, :k]  # k más cercanos


def bench_linear(data, queries, gt, k=K):
    lat = []
    idxs = np.zeros((len(queries), k), dtype=np.int64)
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        dots = data @ q
        idx = np.argpartition(dots, -k)[-k:]
        lat.append((time.perf_counter() - t0) * 1000)
        idxs[i] = idx
    hits = sum(
        len(set(idxs[i].tolist()) & set(gt[i].tolist())) for i in range(len(queries))
    )
    recall = hits / (len(queries) * k)
    lat = np.array(lat)
    return {
        "backend": "linear_numpy",
        "qps": float(len(queries) / (lat.sum() / 1000)),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "recall": float(recall),
    }


def bench_m2m(data, queries, gt, n_probe, rank_by, k=K, use_lsh_fallback=False):
    n, dim = data.shape
    db = SimpleVectorDB(
        latent_dim=dim,
        enable_lsh_fallback=use_lsh_fallback,
        max_splats=n,
        n_probe=n_probe,
    )
    ids = [str(i) for i in range(n)]
    t0 = time.perf_counter()
    db.add(ids=ids, vectors=data)
    build = time.perf_counter() - t0

    n_clusters = db.engine.m2m.splats.engine.n_coarse
    db.engine.m2m.splats.rank_by = rank_by

    for i in range(10):
        db.search(queries[i], k=k, include_metadata=True)

    lat = []
    idxs = np.zeros((len(queries), k), dtype=np.int64)
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        res = db.search(q, k=k, include_metadata=True)
        lat.append((time.perf_counter() - t0) * 1000)
        for j, r in enumerate(res[:k]):
            idxs[i, j] = int(r.id)
    hits = sum(
        len(set(idxs[i].tolist()) & set(gt[i].tolist())) for i in range(len(queries))
    )
    recall = hits / (len(queries) * k)
    lat = np.array(lat)
    out = {
        "backend": f"m2m_{rank_by}",
        "n": n,
        "n_probe": n_probe,
        "n_clusters": int(n_clusters),
        "build_s": float(build),
        "qps": float(len(queries) / (lat.sum() / 1000)),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "recall": float(recall),
    }
    del db
    return out


def bench_batch(data, queries, gt, n_probe=5, k=K):
    """API batch: batch_find_neighbors (prometida para throughput)."""
    n, dim = data.shape
    db = SimpleVectorDB(
        latent_dim=dim, enable_lsh_fallback=False, max_splats=n, n_probe=n_probe
    )
    db.add(ids=[str(i) for i in range(n)], vectors=data)
    db.engine.m2m.splats.rank_by = "l2"
    for i in range(10):
        db.engine.m2m.splats.batch_find_neighbors(queries[i : i + 1], k=k)
    t0 = time.perf_counter()
    out = db.engine.m2m.splats.batch_find_neighbors(queries, k=k)
    dt = time.perf_counter() - t0
    mu, alpha, idx = out[0], out[1], out[2]
    qps = len(queries) / dt
    del db
    return {"backend": "m2m_batch", "qps": float(qps), "note": "verificar idx shape"}


def bench_default_add_cost(data):
    """Costo del add() DEFAULT (enable_lsh_fallback=True): silhouette + posible LSH."""
    n, dim = data.shape
    t0 = time.perf_counter()
    db = SimpleVectorDB(latent_dim=dim, enable_lsh_fallback=True, max_splats=n)
    t_init = time.perf_counter() - t0
    t0 = time.perf_counter()
    db.add(ids=[str(i) for i in range(n)], vectors=data)
    t_add = time.perf_counter() - t0
    used_lsh = db._use_lsh
    out = {
        "init_s": float(t_init),
        "add_s_default": float(t_add),
        "lsh_activated": bool(used_lsh),
    }
    del db
    return out


def main():
    results = {"meta": {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "dim": DIM, "k": K, "n_queries": N_QUERIES,
        "seed": SEED,
    }, "results": []}

    for n in [10000, 50000]:
        data, queries = gen_clustered(n, DIM)
        gt = ground_truth(data, queries)
        results["results"].append({"dataset": f"clustered_{n}", **bench_linear(data, queries, gt)})
        for n_probe in [5, 10]:
            r = bench_m2m(data, queries, gt, n_probe, "l2")
            results["results"].append({"dataset": f"clustered_{n}", **r})
        r = bench_m2m(data, queries, gt, 5, "gaussian")
        results["results"].append({"dataset": f"clustered_{n}", **r})

    # Uniforme 10k (peor caso IVF)
    data, queries = gen_uniform(10000, DIM)
    gt = ground_truth(data, queries)
    results["results"].append({"dataset": "uniform_10000", **bench_linear(data, queries, gt)})
    results["results"].append({"dataset": "uniform_10000", **bench_m2m(data, queries, gt, 5, "l2")})
    results["results"].append({"dataset": "uniform_10000", **bench_m2m(data, queries, gt, 10, "l2")})

    # Batch API (solo 10k para tiempo)
    data, queries = gen_clustered(10000, DIM)
    gt = ground_truth(data, queries)
    try:
        r = bench_batch(data, queries, gt)
        results["results"].append({"dataset": "clustered_10000", **r})
    except Exception as e:
        results["results"].append({"dataset": "clustered_10000", "backend": "m2m_batch",
                                   "error": repr(e)})

    # Costo default add con LSH fallback (10k)
    data, _ = gen_clustered(10000, DIM)
    results["default_add_10000"] = bench_default_add_cost(data)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
