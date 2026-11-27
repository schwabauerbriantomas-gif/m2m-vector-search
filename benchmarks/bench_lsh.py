"""Medición de latencia LSH vs scan exacto (NO es un test: no aserta nada).

Salió de ``tests/test_lsh.py::test_lsh_speedup``, que asertaba ``speedup >= 0.3`` con
``time.time`` y fallaba bajo carga del host. La CI no fija umbrales por tiempos (los
runners son ruidosos): ``.github/workflows/benchmark.yml`` corre este script y sube el
JSON como artifact para comparar contra runs anteriores.

Uso (desde la raíz del repo, con o sin el paquete instalado)::

    python benchmarks/bench_lsh.py                      # JSON por stdout
    python benchmarks/bench_lsh.py --out benchmark_lsh.json
    python benchmarks/bench_lsh.py --n-vectors 20000 --dim 640 --repeats 5

Fijá ``OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`` para que los números sean comparables
entre máquinas: con los hilos por defecto de BLAS las ops chicas hacen spin-wait.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np


def _import_m2m():
    """Permite correr desde un checkout sin instalar (PYTHONPATH o fallback a src/)."""
    try:
        import m2m  # noqa: F401
    except ImportError:  # pragma: no cover - depende del entorno
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from m2m.lsh_index import CrossPolytopeLSH, LSHConfig

    return CrossPolytopeLSH, LSHConfig


def _unit_vectors(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def _linear_scan(vectors: np.ndarray, query: np.ndarray, k: int):
    distances = np.linalg.norm(vectors - query, axis=1)
    return np.argsort(distances)[:k], np.sort(distances)[:k]


def _best_of(fn, repeats: int) -> float:
    """Menor tiempo total entre `repeats` corridas (min = menos ruido del scheduler)."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def run(n_vectors: int, dim: int, k: int, n_queries: int, repeats: int, seed: int) -> dict:
    CrossPolytopeLSH, LSHConfig = _import_m2m()
    rng = np.random.default_rng(seed)

    vectors = _unit_vectors(rng, n_vectors, dim)
    base = rng.choice(n_vectors, size=n_queries, replace=False)
    noise = rng.standard_normal((n_queries, dim)).astype(np.float32) * 0.05
    queries = vectors[base] + noise
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    config = LSHConfig(dim=dim, n_tables=15, n_bits=18, n_probes=50, n_candidates=500, seed=seed)
    index = CrossPolytopeLSH(config)
    index_time = time.perf_counter()
    index.index(vectors)
    index_time = time.perf_counter() - index_time

    # Warm-up (una query por camino) para no cargar JIT/caches en la primera medición.
    index.query(queries[0], k=k)
    _linear_scan(vectors, queries[0], k)

    def scan_all():
        for query in queries:
            _linear_scan(vectors, query, k)

    def lsh_all():
        for query in queries:
            index.query(query, k=k)

    linear_time = _best_of(scan_all, repeats) / n_queries * 1000
    lsh_time = _best_of(lsh_all, repeats) / n_queries * 1000

    # Recall@k de LSH contra ground truth exacto: la velocidad sin calidad no dice nada.
    hits = expected = 0
    for query in queries:
        truth, _ = _linear_scan(vectors, query, k)
        found, _ = index.query(query, k=k)
        hits += len(set(found.tolist()) & set(truth.tolist()))
        expected += k

    return {
        "params": {
            "n_vectors": n_vectors,
            "dim": dim,
            "k": k,
            "n_queries": n_queries,
            "repeats": repeats,
            "seed": seed,
            "lsh_config": {
                "n_tables": config.n_tables,
                "n_bits": config.n_bits,
                "n_probes": config.n_probes,
                "n_candidates": config.n_candidates,
            },
        },
        "linear_scan_ms_per_query": round(linear_time, 4),
        "lsh_ms_per_query": round(lsh_time, 4),
        "speedup": round(linear_time / max(lsh_time, 1e-10), 3),
        "lsh_index_build_s": round(index_time, 4),
        "lsh_recall_at_k": round(hits / expected, 4),
        "env": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--n-vectors", type=int, default=20000)
    parser.add_argument("--dim", type=int, default=640)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=None, help="escribir el JSON a este archivo")
    args = parser.parse_args()

    result = run(args.n_vectors, args.dim, args.k, args.n_queries, args.repeats, args.seed)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"escrito: {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
