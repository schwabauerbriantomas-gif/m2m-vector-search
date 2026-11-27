#!/usr/bin/env python
"""Tabla 1b — opciones que devuelven el top-10 EXACTO en datos uniformes (uniform 10k × 384, k=10).

No hay ``exact=True`` ni un switch de escaneo lineal en el camino sin filtro. Qué se mide, todo
contra GT exacto en float64 (recall@10) y con el protocolo de ``bench/_protocol.py``:

  * ``default``: ``SimpleVectorDB()`` con su ``n_probe`` de fábrica (referencia aproximada);
  * ``n_probe=<n_coarse>``: ``SimpleVectorDB(n_probe=...)`` con el número REAL de clústeres
    gruesos del índice, leído de ``engine.coarse_cluster_embeddings`` tras el ``add()`` (no
    asumido: se imprime y queda en ``variantes``); valores mayores se recortan a ese número
    dentro de ``search_batch``;
  * filtro que acepta todo (medido sobre el índice default): ``search(q, k, filter=acepta)`` y
    ``search_batch_filtered(Q, k, filter=acepta)`` toman el camino exacto de P17 (un clúster
    sólo se salta si el bound ``(dist(q, c) - r_c)**2 >= d2_k`` prueba que no tiene el top-k);
  * escaneo lineal en NumPy: single (``X @ q`` + top-10) y batched B=200 (``Q @ X.T``), la referencia.

Todas las variantes indexan la MISMA matriz X (semilla del protocolo); queries y GT son los de
``generar_queries`` / ``ground_truth`` (200 queries).

Uso (desde la raíz del árbol, con OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1):
    PYTHONPATH=$PWD/src python bench/bench_exact_uniform.py [--out-dir DIR]
Salida: benchmarks/results/bench_exact_uniform.json — la tabla «What gives exact results» de
docs/PERFORMANCE.md se lee de ahí.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _protocol as P  # noqa: E402

KIND, N, D = "uniform", 10000, 384
TABLA = "bench_exact_uniform"
ACEPTA_TODO = "filter=accept-all"


def introspect(db):
    """Número real de clústeres gruesos del índice y ``n_probe`` efectivo."""
    eng = db.engine.m2m.splats.engine
    cce = getattr(eng, "coarse_cluster_embeddings", None)  # Dict[coarse_id -> embeddings]
    return {
        "n_coarse": int(len(cce)) if cce is not None else P.ND,
        "n_probe_engine": int(getattr(eng, "n_probe", 0) or 0),
        "n_docs": int(len(db._splat_id_order)),
    }


def acepta_todo(_metadata):
    return True


def medir_indice(tag, db, Q, gt, filas, meta, dataset, build_s, con_filtro):
    info = introspect(db)
    comunes = {"variante": tag, "camino": "m2m", "n_probe": info["n_probe_engine"], "n_coarse": info["n_coarse"]}

    def single(q):
        return db.search(q, k=P.K)

    with P.RssSampler() as rs:
        m = P.medir_single(single, Q)
    rec = P.recall_at_k([P.ids_desde_posiciones(db, single(q)[3])[0] for q in Q], gt)
    filas.append(P.fila(TABLA, meta, dataset, "search(q,k) legacy (tupla)", m, rec, rs.pico, build_s, **comunes))

    def batch():
        return db.search_batch(Q, k=P.K)

    with P.RssSampler() as rs:
        m = P.medir(batch, len(Q))
    rec = P.recall_at_k(P.ids_desde_posiciones(db, batch()), gt)
    filas.append(P.fila(TABLA, meta, dataset, "search_batch B=200", m, rec, rs.pico, build_s, B=len(Q), **comunes))

    if not con_filtro:
        return

    def single_f(q):
        return db.search(q, k=P.K, filter=acepta_todo, include_metadata=True)

    with P.RssSampler() as rs:
        m = P.medir_single(single_f, Q)
    rec = P.recall_at_k([P.ids_docresult(single_f(q)) for q in Q], gt)
    filas.append(P.fila(TABLA, meta, dataset, f"search(q,k) {ACEPTA_TODO} (exact path, P17)", m, rec, rs.pico,
                        build_s, **dict(comunes, camino="m2m_filtro")))

    def batch_f():
        return db.search_batch_filtered(Q, k=P.K, filter=acepta_todo)

    with P.RssSampler() as rs:
        m = P.medir(batch_f, len(Q))
    rec = P.recall_at_k(P.ids_desde_posiciones(db, batch_f()), gt)
    filas.append(P.fila(TABLA, meta, dataset, f"search_batch_filtered B=200 {ACEPTA_TODO}", m, rec, rs.pico,
                        build_s, B=len(Q), **dict(comunes, camino="m2m_filtro")))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=None, help="directorio de salida (default benchmarks/results/)")
    args = ap.parse_args()
    P.requerir_hilos_1()
    import m2m  # noqa: E402  (del árbol indicado por PYTHONPATH)

    meta = P.meta(m2m)
    print(f"m2m={meta['m2m_file']} arbol={meta['arbol']} commit={meta['commit_corto']} "
          f"src_sucio={meta['src_sucio']} hilos={meta['hilos']}", flush=True)
    dataset = P.dataset_fila(KIND, N, D)
    ds = P.generar(KIND, N, D)
    X = ds["X"]
    Q = P.generar_queries(KIND, ds)
    gt = P.ground_truth(X, Q, P.K)
    filas, variantes = [], {}

    # ---- referencia: escaneo lineal en NumPy --------------------------------
    def lin_single(q):
        return P.topk((X @ q)[np.newaxis, :])

    with P.RssSampler() as rs:
        m = P.medir_single(lin_single, Q)
    rec = P.recall_at_k([lin_single(q)[0].tolist() for q in Q], gt)
    filas.append(P.fila(TABLA, meta, dataset, "linear_single (X @ q + top-10)", m, rec, rs.pico, None,
                        variante="linear", camino="baseline"))

    def lin_batch():
        return P.topk(Q @ X.T, P.K)

    with P.RssSampler() as rs:
        m = P.medir(lin_batch, len(Q))
    rec = P.recall_at_k(lin_batch().tolist(), gt)
    filas.append(P.fila(TABLA, meta, dataset, "linear_batched B=200 (Q @ X.T)", m, rec, rs.pico, None,
                        variante="linear", camino="baseline", B=len(Q)))

    # ---- índice default -----------------------------------------------------
    with P.RssSampler() as rs:
        db, build_s = P.construir_db(X)
    variantes["default"] = dict(introspect(db), n_probe_configurado=None, build_s=round(build_s, 3),
                                rss_pico_build_mb=round(rs.pico, 1))
    print(f"  default: {variantes['default']}", flush=True)
    medir_indice("default", db, Q, gt, filas, meta, dataset, build_s, con_filtro=True)
    del db

    # ---- n_probe = número real de clústeres gruesos -------------------------
    n_coarse = variantes["default"]["n_coarse"]
    if not isinstance(n_coarse, int):
        raise SystemExit(f"no pude leer el número de clústeres gruesos: {n_coarse!r}")
    tag = f"n_probe={n_coarse}"
    with P.RssSampler() as rs:
        dbx, build_s = P.construir_db(X, n_probe=n_coarse)
    variantes[tag] = dict(introspect(dbx), n_probe_configurado=n_coarse, build_s=round(build_s, 3),
                          rss_pico_build_mb=round(rs.pico, 1))
    print(f"  {tag}: {variantes[tag]}", flush=True)
    medir_indice(tag, dbx, Q, gt, filas, meta, dataset, build_s, con_filtro=False)
    del dbx

    destino = P.escribir(TABLA, filas, meta, out_dir=args.out_dir, variantes=variantes,
                         config=[KIND, N, D], filtro_acepta_todo="callable que devuelve True para todo metadata",
                         tabla_docs="docs/PERFORMANCE.md «What gives exact results»")
    P.resumen(filas)
    print("->", destino)


if __name__ == "__main__":
    main()
