#!/usr/bin/env python
"""Tabla 2 — filtros de metadata (audit r4-04, r4-14, r5-15). Dataset: clustered 10k × d=384.

Qué mide (el MISMO script para la rama y para el commit base), para 4 filtros:
  * ``nocorr_25`` / ``nocorr_5``: NO correlacionados con el clúster (``i % 4 == 0``,
    ``i % 20 == 0``; selectividad exacta 25 % / 5 %);
  * ``corr_25`` / ``corr_5``: CORRELACIONADOS con el clúster verdadero (pasan sólo los
    docs de 1/4 y 1/20 de los clústeres del dataset; la selectividad real se anota).
Métodos:
  * ``search(q, k, filter=dict)`` repetida (máscara cacheada);
  * ``search(q, k, filter=dict)`` 1ª evaluación: antes de cada consulta una escritura
    pública (``update()`` de metadata) invalida la caché; se cronometra SÓLO la consulta;
  * ``search_batch_filtered(Q, k, filter=dict)`` B=200 (API nueva, P13);
  * ``search_batch_filtered(Q, k, filter_fn=callable)`` B=200 (API vieja);
  * ``search_batch(Q, k)`` B=200 + post-filtro del top-10 (el "antes" del README).
GT = fuerza bruta float64 restringido a los docs que pasan el filtro; recall@10 con
denominador min(10, docs que pasan). Protocolo: ``bench/_protocol.py``.

Uso (desde la raíz del árbol a medir, con OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1):
    PYTHONPATH=$PWD/src python bench/bench_filtered.py [--out-dir DIR] [--etiqueta TEXTO]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _protocol as P  # noqa: E402

TABLA = "bench_filtered"
CONFIG = ("clustered", 10000, 384)
B_BATCH = 200
N_FRIO = 20


def filtros(labels: np.ndarray):
    n = len(labels)
    i = np.arange(n)
    unicos = np.unique(labels)
    corr25 = np.isin(labels, unicos[::4])
    corr5 = np.isin(labels, unicos[::20])
    b25 = (i % 4) == 0
    b5 = (i % 20) == 0
    return [
        ("nocorr_25", {"bucket": 0}, b25, "no correlacionado: i % 4 == 0"),
        ("nocorr_5", {"bucket20": 0}, b5, "no correlacionado: i % 20 == 0"),
        ("corr_25", {"cl25": 1}, corr25, "correlacionado: 1/4 de los clusters verdaderos"),
        ("corr_5", {"cl5": 1}, corr5, "correlacionado: 1/20 de los clusters verdaderos"),
    ]


def _version_meta(db):
    try:
        return int(db._meta_version)
    except AttributeError:
        return None


def correr_filtro(nombre, filtro, mask, desc, db, Q, gt, X, filas, meta, dataset, metadata, build_s):
    selectividad = float(mask.mean())
    extra = {"filtro": filtro, "selectividad": selectividad, "descripcion": desc}
    gt_f = P.ground_truth(X, Q, permitidos=mask)
    consulta = lambda q, f=filtro: db.search(q, k=P.K, filter=f)

    # 1) single, repetida (máscara cacheada)
    with P.RssSampler() as rs:
        m = P.medir_single(consulta, Q)
    rec = P.recall_at_k([P.ids_docresult(consulta(q)) for q in Q], gt_f)
    filas.append(P.fila(TABLA, meta, dataset, f"search(filter=dict) repetida [{nombre}]", m, rec,
                        rs.pico, build_s, camino="single_filtro", filtro_nombre=nombre, **extra))

    # 2) single, 1ª evaluación: una escritura pública (no cronometada) invalida la caché
    #    antes de CADA consulta cronometrada (medir_frio prepara() -> consulta(q))
    v0 = _version_meta(db)

    def invalidar():
        db.update("doc0", metadata=dict(metadata[0]))

    with P.RssSampler() as rs:
        m = P.medir_frio(invalidar, consulta, Q, n=N_FRIO)
    rec = P.recall_at_k([P.ids_docresult(consulta(Q[j % len(Q)])) for j in range(N_FRIO)], gt_f)
    v1 = _version_meta(db)
    filas.append(P.fila(TABLA, meta, dataset, f"search(filter=dict) 1ra evaluacion [{nombre}]", m, rec,
                        rs.pico, build_s, camino="single_filtro_frio", filtro_nombre=nombre,
                        invalidacion_verificada=(None if v0 is None else bool(v1 > v0)),
                        invalidacion=("escritura publica update(doc0, metadata) antes de cada consulta"
                                      if v0 is not None else "sin cache de mascaras en este commit"),
                        **extra))

    # 3) batch filtrado con filter=dict (API nueva)
    try:
        with P.RssSampler() as rs:
            m = P.medir(lambda: db.search_batch_filtered(Q, k=P.K, filter=filtro), B_BATCH)
        r = db.search_batch_filtered(Q, k=P.K, filter=filtro)
        rec = P.recall_at_k(P.ids_desde_posiciones(db, r), gt_f)
        filas.append(P.fila(TABLA, meta, dataset, f"search_batch_filtered B=200 filter=dict [{nombre}]",
                            m, rec, rs.pico, build_s, camino="batch_filtro_dict", B=B_BATCH,
                            filtro_nombre=nombre, **extra))
    except Exception as exc:
        filas.append(P.nd(TABLA, meta, dataset,
                          f"search_batch_filtered B=200 filter=dict [{nombre}]",
                          f"{type(exc).__name__}: {exc}"[:300]))

    # 4) batch filtrado con filter_fn=callable (API vieja)
    try:
        fn = lambda p, mask=mask: bool(mask[int(p)])
        with P.RssSampler() as rs:
            m = P.medir(lambda: db.search_batch_filtered(Q, k=P.K, filter_fn=fn), B_BATCH)
        r = db.search_batch_filtered(Q, k=P.K, filter_fn=fn)
        rec = P.recall_at_k(P.ids_desde_posiciones(db, r), gt_f)
        filas.append(P.fila(TABLA, meta, dataset, f"search_batch_filtered B=200 filter_fn=callable [{nombre}]",
                            m, rec, rs.pico, build_s, camino="batch_filtro_fn", B=B_BATCH,
                            filtro_nombre=nombre, **extra))
    except Exception as exc:
        filas.append(P.nd(TABLA, meta, dataset,
                          f"search_batch_filtered B=200 filter_fn=callable [{nombre}]",
                          f"{type(exc).__name__}: {exc}"[:300]))

    # 5) batch + post-filtro del top-10 (legacy)
    def post_filtro(Qb):
        pos = db.search_batch(Qb, k=P.K)
        ids = P.ids_desde_posiciones(db, pos)
        return [[i for i in fila if i >= 0 and mask[i]] for fila in ids]

    def rep():
        post_filtro(Q[:B_BATCH])

    with P.RssSampler() as rs:
        m = P.medir(rep, B_BATCH)
    rec = P.recall_at_k(post_filtro(Q[:B_BATCH]), gt_f)
    filas.append(P.fila(TABLA, meta, dataset, f"search_batch B=200 + post-filtro top-10 [{nombre}]",
                        m, rec, rs.pico, build_s, camino="batch_postfiltro", B=B_BATCH,
                        filtro_nombre=nombre, **extra))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--etiqueta", default="")
    args = ap.parse_args()
    P.requerir_hilos_1()
    import m2m  # noqa: E402

    meta = P.meta(m2m)
    kind, n, d = CONFIG
    dataset = P.dataset_fila(kind, n, d)
    print(f"m2m={meta['m2m_file']} arbol={meta['arbol']} commit={meta['commit_corto']} "
          f"src_sucio={meta['src_sucio']} hilos={meta['hilos']}", flush=True)
    ds = P.generar(kind, n, d)
    X, labels = ds["X"], ds["labels"]
    Q = P.generar_queries(kind, ds)
    defs = filtros(labels)
    metadata = [
        {"cat": int(labels[i]), "bucket": int(i % 4), "bucket20": int(i % 20),
         "cl25": int(defs[2][2][i]), "cl5": int(defs[3][2][i])}
        for i in range(n)
    ]
    db, build_s = P.construir_db(X, metadata=metadata)
    print(f"build={build_s:.2f}s", flush=True)
    filas = []
    for nombre, filtro, mask, desc in defs:
        try:
            correr_filtro(nombre, filtro, mask, desc, db, Q, P.ground_truth(X, Q), X, filas, meta,
                          dataset, metadata, build_s)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            filas.append(P.nd(TABLA, meta, dataset, f"filtro {nombre}", f"{type(exc).__name__}: {exc}"[:300]))
    destino = P.escribir(TABLA, filas, meta, out_dir=args.out_dir, etiqueta=args.etiqueta,
                         dataset_info={"build_s": round(build_s, 3), **P.info_indice(db)})
    print(f"commit={meta['commit_corto']} etiqueta={args.etiqueta!r}")
    P.resumen(filas)
    print("->", destino)


if __name__ == "__main__":
    main()
