#!/usr/bin/env python
"""Tabla 4 — storage: arrays vs RSS pico, snapshot en disco, reopen y escrituras (audit r4-02, r4-15).

Qué mide (el MISMO script para la rama y para el commit base), clustered × d=384:
  * ``build``:  ``add(n)`` en memoria para n = 10k / 50k: tiempo, RSS antes / pico / después y
                tamaño de los arrays de splats (μ, α, κ float32): analítico n·(d+2)·4, reservado
                (lo que el store tiene alocado) y activo (filas vivas). El RSS pico medido es
                el del PROCESO (incluye copias temporales del build): no se afirma que los
                arrays expliquen el RSS.
  * ``snapshot``: ``save(path)`` → tamaño en disco del directorio, tiempo de reopen
                (``SimpleVectorDB(storage_path=path)``, reconstruye el índice) y n° de docs.
  * ``escritura``: ``add(1k)`` y ``add(10k)`` con y sin ``storage_path`` (1 transacción por
                ``add()``; no se llama ``save()``): docs/s.
  * ``add_unitario``: 100 ``add()`` de 1 doc con y sin ``storage_path``: ms por add.
Latencia = tiempo de la operación completa (ms). Protocolo: ``bench/_protocol.py``.

Uso (desde la raíz del árbol a medir, con OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1):
    PYTHONPATH=$PWD/src python bench/bench_storage.py [--out-dir DIR] [--etiqueta TEXTO]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _protocol as P  # noqa: E402

TABLA = "bench_storage"
D = 384
NS = (10000, 50000)
REOPEN_REPS = 3
ESCRITURAS = (1000, 10000)
ESCRITURA_REPS = 2
ADDS_UNITARIOS = 100


def _fresh(path):
    """Garantiza un directorio inexistente (save() rechaza paths existentes no vacíos)."""
    import shutil
    if os.path.isdir(path):
        shutil.rmtree(path)
    return path


def bloque_build_snapshot(m2m, filas, meta, tmp, etiqueta):
    for n in NS:
        kind = "clustered"
        dataset = P.dataset_fila(kind, n, D)
        print(f"[build n={n}] generando...", flush=True)
        ds = P.generar(kind, n, D)
        X = ds["X"]
        metadata = [{"cat": int(l)} for l in ds["labels"]]
        try:
            rss0 = P.rss_tras_gc()
            with P.RssSampler() as rs:
                db, build_s = P.construir_db(X, metadata=metadata)
            rss1 = P.rss_tras_gc()
            sp = db.engine.m2m.splats
            d_eff = int(sp.mu.shape[1])
            reservados = (sp.mu.nbytes + sp.alpha.nbytes + sp.kappa.nbytes) / 2 ** 20
            activos = n * (d_eff + 2) * 4 / 2 ** 20
            analitico = n * (D + 2) * 4 / 2 ** 20
            m = P.resumen_de_tiempos([build_s], n, qps_definicion="docs / tiempo de add(n)")
            filas.append(P.fila(TABLA, meta, dataset, f"add({n} docs) en memoria (build)", m, None,
                                rs.pico, build_s, camino="build", unidad="ms por add(n)",
                                rss_antes_mb=round(rss0, 1), rss_tras_build_mb=round(rss1, 1),
                                rss_delta_mb=round(rss1 - rss0, 1),
                                arrays_analitico_mb=round(analitico, 1),
                                arrays_reservados_mb=round(reservados, 1),
                                arrays_activos_mb=round(activos, 1),
                                docs_por_s=round(n / build_s, 1) if build_s > 0 else None))
        except Exception as exc:
            filas.append(P.nd(TABLA, meta, dataset, f"add({n} docs) en memoria (build)",
                              f"{type(exc).__name__}: {exc}"[:300]))
            continue
        try:
            path = _fresh(os.path.join(tmp, f"p9_snap_{n}_{etiqueta}"))
            t0 = time.perf_counter()
            db.save(path)
            save_s = time.perf_counter() - t0
            disco = P.dir_mb(path)
            reopen, n_docs = [], None
            for _ in range(REOPEN_REPS):
                t0 = time.perf_counter()
                db2 = m2m.SimpleVectorDB(latent_dim=D, storage_path=path)
                reopen.append(time.perf_counter() - t0)
                n_docs = P.info_indice(db2).get("n_docs")
                try:
                    db2.storage.close()
                except Exception:
                    pass
                del db2
            m = P.resumen_de_tiempos(reopen, None,
                                     qps_definicion="no aplica (latencia = tiempo de reopen)")
            filas.append(P.fila(TABLA, meta, dataset,
                                f"save(path) + reopen SimpleVectorDB(storage_path=path) n={n}", m,
                                None, None, None, camino="snapshot", unidad="ms por reopen",
                                save_s=round(save_s, 3), disco_mb=round(disco, 1),
                                bytes_por_doc=round(disco * 2 ** 20 / n, 1), n_docs_reopen=n_docs,
                                reopen_s=[round(t, 3) for t in reopen]))
        except Exception as exc:
            filas.append(P.nd(TABLA, meta, dataset, f"save(path) + reopen n={n}",
                              f"{type(exc).__name__}: {exc}"[:300]))
        try:
            db.storage.close()
        except Exception:
            pass
        del db


def bloque_escrituras(m2m, filas, meta, tmp, etiqueta):
    print(f"[escritura] generando {max(ESCRITURAS)}...", flush=True)
    X = P.generar("clustered", max(ESCRITURAS), D, seed=P.SEED_DATOS + 1)["X"]
    metadata = [{"cat": int(i % 16)} for i in range(len(X))]
    for size in ESCRITURAS:
        dataset = P.dataset_fila("clustered", size, D)
        for con_storage in (False, True):
            nombre = f"add({size} docs) {'con' if con_storage else 'sin'} storage_path"
            tiempos, disco = [], None
            try:
                for r in range(ESCRITURA_REPS):
                    sp = _fresh(os.path.join(tmp, f"p9_write_{size}_{int(con_storage)}_{etiqueta}_{r}")) \
                        if con_storage else None
                    db = m2m.SimpleVectorDB(latent_dim=D, storage_path=sp)
                    t0 = time.perf_counter()
                    db.add(ids=[f"{P.PREFIJO_ID}{i}" for i in range(size)],
                           vectors=X[:size], metadata=metadata[:size])
                    tiempos.append(time.perf_counter() - t0)
                    if sp:
                        disco = P.dir_mb(sp)
                    try:
                        db.storage.close() if db.storage else None
                    except Exception:
                        pass
                    del db
                m = P.resumen_de_tiempos(tiempos, size,
                                         qps_definicion="docs / tiempo de add() (mediana entre repeticiones)")
                filas.append(P.fila(TABLA, meta, dataset, nombre, m, None, None, None,
                                    camino="escritura", unidad="ms por add() completo",
                                    docs=size, con_storage_path=con_storage,
                                    disco_mb=None if disco is None else round(disco, 1),
                                    tiempos_s=[round(t, 3) for t in tiempos]))
            except Exception as exc:
                filas.append(P.nd(TABLA, meta, dataset, nombre, f"{type(exc).__name__}: {exc}"[:300]))


def bloque_adds_unitarios(m2m, filas, meta, tmp, etiqueta):
    X = P.generar("clustered", ADDS_UNITARIOS, D, seed=P.SEED_DATOS + 2)["X"]
    dataset = P.dataset_fila("clustered", ADDS_UNITARIOS, D)
    for con_storage in (False, True):
        nombre = f"add(1 doc) x{ADDS_UNITARIOS} {'con' if con_storage else 'sin'} storage_path"
        try:
            sp = _fresh(os.path.join(tmp, f"p9_add1_{int(con_storage)}_{etiqueta}")) if con_storage else None
            db = m2m.SimpleVectorDB(latent_dim=D, storage_path=sp)
            lat = []
            for j in range(ADDS_UNITARIOS):
                t0 = time.perf_counter()
                db.add(ids=[f"u{j}"], vectors=X[j:j + 1], metadata=[{"cat": j % 16}])
                lat.append(time.perf_counter() - t0)
            disco = P.dir_mb(sp) if sp else None
            try:
                if db.storage:
                    db.storage.close()
            except Exception:
                pass
            m = P.resumen_de_tiempos(lat, 1, qps_definicion="adds / s (1 doc por add; 1/mediana)")
            filas.append(P.fila(TABLA, meta, dataset, nombre, m, None, None, None,
                                camino="add_unitario", unidad="ms por add(1 doc)",
                                con_storage_path=con_storage,
                                disco_mb=None if disco is None else round(disco, 1)))
        except Exception as exc:
            filas.append(P.nd(TABLA, meta, dataset, nombre, f"{type(exc).__name__}: {exc}"[:300]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--etiqueta", default="")
    args = ap.parse_args()
    P.requerir_hilos_1()
    import m2m  # noqa: E402

    meta = P.meta(m2m)
    print(f"m2m={meta['m2m_file']} arbol={meta['arbol']} commit={meta['commit_corto']} "
          f"src_sucio={meta['src_sucio']} hilos={meta['hilos']}", flush=True)
    tmp = os.environ.get("TMPDIR") or "/tmp"
    os.makedirs(tmp, exist_ok=True)
    etiqueta = f"{meta['arbol']}-{meta['commit_corto']}"
    filas = []
    bloque_build_snapshot(m2m, filas, meta, tmp, etiqueta)
    bloque_escrituras(m2m, filas, meta, tmp, etiqueta)
    bloque_adds_unitarios(m2m, filas, meta, tmp, etiqueta)
    destino = P.escribir(TABLA, filas, meta, out_dir=args.out_dir, etiqueta=args.etiqueta,
                         reopen_reps=REOPEN_REPS, escritura_reps=ESCRITURA_REPS,
                         adds_unitarios=ADDS_UNITARIOS)
    print(f"commit={meta['commit_corto']} etiqueta={args.etiqueta!r}")
    for f in filas:
        if f.get("estado") == P.ND:
            print(f"  [n/d] {f['parametro']}: {f['motivo']}")
        else:
            qps = "n/a" if f["qps"] is None else f"{f['qps']:.1f}"
            print(f"  {f['parametro']:<58} med={f['mediana']:>10.2f}ms qps={qps} n={f['n_repeticiones']}")
    print("->", destino)


if __name__ == "__main__":
    main()
