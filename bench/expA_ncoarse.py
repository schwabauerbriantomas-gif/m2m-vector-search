#!/usr/bin/env python3
"""Experimento A: n_coarse correcto (desde n_active, no max_splats) sin tocar codigo.
Monkeypatch del engine post-construccion + rebuild. Mide QPS/recall en el grid.
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
NQ = 200
DIM = 384


def gen_clustered(n, dim, n_clusters=50, seed=SEED):
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, dim).astype(np.float32)
    assign = rng.randint(0, n_clusters, size=n)
    data = centers[assign] + 0.35 * rng.randn(n, dim).astype(np.float32)
    data /= np.linalg.norm(data, axis=1, keepdims=True) + 1e-9
    queries = centers[rng.randint(0, n_clusters, size=NQ)] + 0.35 * rng.randn(NQ, dim).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True) + 1e-9
    return data.astype(np.float32), queries.astype(np.float32)


def gt_exact(data, queries, k=K):
    d2 = 2.0 - 2.0 * (queries @ data.T)
    return np.argsort(d2, axis=1)[:, :k]


def run(data, queries, gt, n_coarse, n_probe, rank_by="l2"):
    n, dim = data.shape
    db = SimpleVectorDB(latent_dim=dim, enable_lsh_fallback=False, max_splats=n, n_probe=n_probe)
    db.add(ids=[str(i) for i in range(n)], vectors=data)
    eng = db.engine.m2m.splats.engine
    eng.n_coarse = n_coarse  # monkeypatch: n_coarse real, no el degenerado
    t0 = time.perf_counter()
    db.engine.m2m.splats.build_index()
    build = time.perf_counter() - t0
    actual_clusters = eng.n_coarse
    db.engine.m2m.splats.rank_by = rank_by
    for i in range(10):
        db.search(queries[i], k=K, include_metadata=True)
    lat = []
    idxs = np.zeros((NQ, K), dtype=np.int64)
    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        res = db.search(q, k=K, include_metadata=True)
        lat.append((time.perf_counter() - t0) * 1000)
        for j, r in enumerate(res[:K]):
            idxs[i, j] = int(r.id)
    hits = sum(len(set(idxs[i].tolist()) & set(gt[i].tolist())) for i in range(NQ))
    recall = hits / (NQ * K)
    lat = np.array(lat)
    out = {
        "n_coarse_cfg": n_coarse, "n_clusters": int(actual_clusters), "n_probe": n_probe,
        "build_s": round(build, 3), "qps": round(NQ / (lat.sum() / 1000), 1),
        "p50_ms": round(float(np.percentile(lat, 50)), 3), "recall": round(recall, 4),
    }
    del db
    return out


def main():
    results = []
    for n in [10000, 50000]:
        data, queries = gen_clustered(n, DIM)
        gt = gt_exact(data, queries)
        sqrt_n = int(np.sqrt(n))
        for nc in [sqrt_n // 2, sqrt_n, sqrt_n * 2]:
            for npr in [max(1, nc // 20), max(1, nc // 10), max(2, nc // 5)]:
                r = run(data, queries, gt, nc, npr)
                r["n"] = n
                results.append(r)
                print(f"n={n:>6d} n_coarse={nc:>4d} (cl={r['n_clusters']:>4d}) n_probe={npr:>3d} "
                      f"build={r['build_s']:>6.2f}s QPS={r['qps']:>8.1f} p50={r['p50_ms']:>7.3f}ms recall={r['recall']:.4f}",
                      flush=True)
    out_path = os.path.join(os.path.dirname(__file__), "expA_ncoarse.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print("saved", out_path)


if __name__ == "__main__":
    main()
