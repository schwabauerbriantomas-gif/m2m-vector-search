#!/usr/bin/env python
"""Tabla 1 — throughput y recall de búsqueda (audit r4-06, r4-10, r4-11, r5-01, r5-08).

Qué mide (el MISMO script para la rama y para el commit base):
  * baselines de escaneo lineal en NumPy puro: single (``X @ q`` por query, con top-10) y
    batched (``X @ Q.T``) con B = 200 y B = 1000;
  * ``db.search(q, k)`` en modo legacy (tupla) y con ``include_metadata=True``;
  * ``db.search_batch(Q, k)`` con B = 1 / 8 / 200 / 1000 (B = 1000 = las 200 queries
    repetidas 5 veces: la mediana de QPS no cambia, el recall se calcula sobre esas filas).
Datasets: clustered y uniform × 10k / 50k × d = 384, más clustered 10k × d = 64.
Todo contra GT exacto float64 (recall@10). Protocolo: ``bench/_protocol.py``.

Uso (desde la raíz del árbol a medir, con OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1):
    PYTHONPATH=$PWD/src python bench/bench_search.py [--out-dir DIR] [--etiqueta TEXTO]
Para la columna "before" se corre este MISMO archivo con PYTHONPATH apuntando al ``src``
del commit base: el protocolo anota el commit que sale de ``m2m.__file__``.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _protocol as P  # noqa: E402

CONFIGS = (
    ("clustered", 10000, 384),
    ("uniform", 10000, 384),
    ("clustered", 50000, 384),
    ("uniform", 50000, 384),
    ("clustered", 10000, 64),
)
TABLA = "bench_search"


def _single(nombre, call, ids_de, Q, gt, filas, meta, dataset, build_s, **extra):
    with P.RssSampler() as rs:
        m = P.medir_single(call, Q)
    rec = P.recall_at_k([ids_de(call(q)) for q in Q], gt)
    filas.append(P.fila(TABLA, meta, dataset, nombre, m, rec, rs.pico, build_s, **extra))


def _batch(nombre, call, ids_de, Qb, gt_b, filas, meta, dataset, build_s, **extra):
    def rep():
        call(Qb)

    with P.RssSampler() as rs:
        m = P.medir(rep, len(Qb))
    rec = P.recall_at_k(ids_de(call(Qb)), gt_b)
    filas.append(P.fila(TABLA, meta, dataset, nombre, m, rec, rs.pico, build_s, **extra))


def correr_dataset(kind, n, d, filas, datasets, meta):
    dataset = P.dataset_fila(kind, n, d)
    print(f"[{kind} {n} d={d}] generando...", flush=True)
    ds = P.generar(kind, n, d)
    X = ds["X"]
    Q = P.generar_queries(kind, ds)
    t0 = time.perf_counter()
    gt = P.ground_truth(X, Q)
    gt_s = time.perf_counter() - t0
    rep = -(-P.BATCHES[-1] // len(Q))
    Qb1000 = np.tile(Q, (rep, 1))
    gt1000 = np.tile(gt, (rep, 1))
    metadata = [{"cat": int(l)} for l in ds["labels"]]

    # ---- baseline lineal (NumPy puro) --------------------------------------
    _single("linear_single (X @ q + top-10)",
            lambda q: P.topk((X @ q)[np.newaxis, :]),
            lambda r: r[0].tolist(), Q, gt, filas, meta, dataset, None, camino="baseline")
    _batch("linear_batched B=200 (Q @ X.T)",
           lambda Qb: P.topk(Qb @ X.T),
           lambda r: r.tolist(), Q, gt, filas, meta, dataset, None, camino="baseline")
    _batch(f"linear_batched B=1000 ({len(Q)} queries x{rep})",
           lambda Qb: P.topk(Qb @ X.T),
           lambda r: r.tolist(), Qb1000, gt1000, filas, meta, dataset, None, camino="baseline")

    # ---- índice M2M ---------------------------------------------------------
    rss0 = P.rss_tras_gc()
    with P.RssSampler() as rs:
        db, build_s = P.construir_db(X, metadata=metadata)
    info = {"kind": kind, "n": n, "d": d, "build_s": round(build_s, 3),
            "rss_antes_mb": round(rss0, 1), "rss_pico_build_mb": round(rs.pico, 1),
            "rss_tras_build_mb": round(P.rss_tras_gc(), 1), "gt_s": round(gt_s, 3)}
    info.update(P.info_indice(db))
    datasets.append(info)
    print(f"  build={build_s:.2f}s {info}", flush=True)

    _single("search(q,k) legacy (tupla)",
            lambda q: db.search(q, k=P.K),
            lambda r: P.ids_desde_posiciones(db, r[3])[0], Q, gt, filas, meta, dataset, build_s,
            camino="m2m")
    _single("search(q,k) include_metadata=True",
            lambda q: db.search(q, k=P.K, include_metadata=True),
            P.ids_docresult, Q, gt, filas, meta, dataset, build_s, camino="m2m")
    for B in P.BATCHES:
        Qb, gt_b = (Qb1000, gt1000) if B == P.BATCHES[-1] else (Q[:B], gt[:B])
        _batch(f"search_batch B={B}" + (f" ({len(Q)} queries x{rep})" if B == P.BATCHES[-1] else ""),
               lambda Qb: db.search_batch(Qb, k=P.K),
               lambda r: P.ids_desde_posiciones(db, r), Qb, gt_b, filas, meta, dataset, build_s,
               camino="m2m", B=B)
    del db


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=None, help="directorio de salida (default benchmarks/results/)")
    ap.add_argument("--etiqueta", default="", help="etiqueta libre de la corrida (p. ej. 'corrida 2')")
    ap.add_argument("--solo", default="", help="sólo datasets cuyo 'kind:n:d' contiene este texto")
    args = ap.parse_args()
    P.requerir_hilos_1()
    import m2m  # noqa: E402  (del árbol indicado por PYTHONPATH)

    meta = P.meta(m2m)
    print(f"m2m={meta['m2m_file']} arbol={meta['arbol']} commit={meta['commit_corto']} "
          f"src_sucio={meta['src_sucio']} hilos={meta['hilos']}", flush=True)
    filas, datasets = [], []
    for kind, n, d in CONFIGS:
        if args.solo and args.solo not in f"{kind}:{n}:{d}":
            continue
        try:
            correr_dataset(kind, n, d, filas, datasets, meta)
        except Exception as exc:  # un dataset que falla no tira el resto
            import traceback
            traceback.print_exc()
            filas.append(P.nd(TABLA, meta, P.dataset_fila(kind, n, d), "dataset completo",
                              f"{type(exc).__name__}: {exc}"[:300]))
    destino = P.escribir(TABLA, filas, meta, out_dir=args.out_dir, datasets=datasets,
                         etiqueta=args.etiqueta, configs=[list(c) for c in CONFIGS])
    print(f"commit={meta['commit_corto']} etiqueta={args.etiqueta!r}")
    P.resumen(filas)
    print("->", destino)


if __name__ == "__main__":
    main()
