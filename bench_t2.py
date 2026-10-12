"""Bench T2 BUG 2: default enable_lsh_fallback sobre rag_test (659x384).

Corre el mismo protocolo que bench_baseline.py pero contra workB (con fix):
recall@10 y QPS de single y batch, con LSH (default) y sin LSH (HRM2 puro).
Escribe /tmp/m2mvs/workB/bench_t2_results.json
"""
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/tmp/m2mvs/workB/src")

from m2m import SimpleVectorDB  # noqa: E402

RNG = np.random.default_rng(42)
K = 10
N_QUERIES = 100
WARMUP = 10


def main():
    emb = np.load("/tmp/m2mvs/embeddings.npy").astype(np.float32)
    n, d = emb.shape
    queries = emb[RNG.choice(n, size=N_QUERIES, replace=False)]
    gt = np.argsort(-(queries @ emb.T), axis=1)[:, :K]
    gt_sets = [set(f"doc_{i}" for i in g.tolist()) for g in gt]

    def recall(rows):
        sets = [set(r.id for r in row) for row in rows]
        return sum(len(s & g) for s, g in zip(sets, gt_sets)) / (N_QUERIES * K)

    results = {"dataset": {"n": int(n), "dim": int(d), "k": K, "n_queries": N_QUERIES}}

    def bench(label, **kwargs):
        db = SimpleVectorDB(latent_dim=d, storage_path=None, **kwargs)
        t0 = time.perf_counter()
        db.add(ids=[f"doc_{i}" for i in range(n)], vectors=emb)
        t_build = time.perf_counter() - t0

        for i in range(WARMUP):
            db.search(queries[i], k=K, include_metadata=True)
        t0 = time.perf_counter()
        res = [db.search(q, k=K, include_metadata=True) for q in queries]
        t_single = time.perf_counter() - t0
        rec_single = recall(res)

        for i in range(WARMUP):
            db.search_batch(queries[:WARMUP], k=K, include_metadata=True)
        t0 = time.perf_counter()
        resb = db.search_batch(queries, k=K, include_metadata=True)
        t_batch = time.perf_counter() - t0
        rec_batch = recall(resb)

        out = {
            "mode": "LSH" if db._use_lsh else "HRM2",
            "build_s": round(t_build, 3),
            "single": {
                "qps": round(N_QUERIES / t_single, 1),
                "latency_ms_p50": round(t_single / N_QUERIES * 1000, 3),
                "recall_at_10": round(rec_single, 4),
            },
            "batch": {
                "qps": round(N_QUERIES / t_batch, 1),
                "recall_at_10": round(rec_batch, 4),
            },
            "config": kwargs,
        }
        results[label] = out
        print(f"{label}: mode={out['mode']} build {t_build:.2f}s | "
              f"single {out['single']['qps']} QPS R@10={rec_single:.4f} | "
              f"batch {out['batch']['qps']} QPS R@10={rec_batch:.4f}")
        return out

    bench("lsh_default_true", enable_lsh_fallback=True)
    bench("hrm2_lsh_false", enable_lsh_fallback=False)

    with open("/tmp/m2mvs/workB/bench_t2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("-> /tmp/m2mvs/workB/bench_t2_results.json")


if __name__ == "__main__":
    main()
