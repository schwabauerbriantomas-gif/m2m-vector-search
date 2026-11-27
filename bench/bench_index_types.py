#!/usr/bin/env python
"""Tabla 3 — tipos de índice y paths experimentales (audit r4-02, r4-15; packs p12b/p12c).

Bloques (todos contra GT exacto float64, protocolo ``bench/_protocol.py``):
  * ``hrm2_lod``:  ``db.engine.m2m.retrieve(q, k, lod=0|1|2)`` en clustered 10k y 50k × d=384
                   (lod 2 es el camino exacto del router; 0/1 escanean menos y devuelven los
                   k más cercanos de lo sondeado).
  * ``sq8``:       ``SQ8DiskStore.build_from_float_npy`` + ``search_batch(Q, k, oversample=2)``
                   en modo completo (``spectral_r=None``) y en fase espectral opt-in
                   (``spectral_r="auto"``), clustered 10k y 50k × d=384. Anota RAM de arrays,
                   disco y tiempo de ``load()``.
  * ``hnsw_lsh``:  HNSW en Python puro (``m2m.hnsw_index.HNSWIndex``) y LSH cross-polytope
                   (``m2m.lsh_index.CrossPolytopeLSH``) con sus defaults, en N=2500 × d=384
                   clustered, contra el scan batched en el mismo N. N=2500 porque su build es
                   muy caro (p12b/p12c): NO extrapolar a 10k/50k.
  * ``gate``:      ``m2m.gate_modulus.gate_select_k(X)`` en N=2500 (clustered y uniform):
                   veredicto + tiempo, en un subproceso con timeout.
El MISMO script corre contra el commit base (lo que la API vieja no tiene sale "n/d").

Uso (desde la raíz del árbol a medir, con OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1):
    PYTHONPATH=$PWD/src python bench/bench_index_types.py [--out-dir DIR] [--bloques hrm2_lod,sq8,...]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _protocol as P  # noqa: E402

TABLA = "bench_index_types"
D = 384
N_HNSW = 2500
N_GATE = 2500
GATE_TIMEOUT_S = 90


def _na(**extra):
    return {"n_repeticiones": 1, "mediana": 0.0, "p5": 0.0, "p95": 0.0, "qps": None,
            "qps_definicion": "no aplica", "carga_cpu_sistema_pct": None, "cpu_propio_pct": None,
            "uptime_antes": None, "uptime_despues": None, **extra}


def bloque_hrm2_lod(filas, meta):
    for n in (10000, 50000):
        kind = "clustered"
        dataset = P.dataset_fila(kind, n, D)
        print(f"[hrm2_lod n={n}] generando + build...", flush=True)
        ds = P.generar(kind, n, D)
        X = ds["X"]
        Q = P.generar_queries(kind, ds)
        gt = P.ground_truth(X, Q)
        db, build_s = P.construir_db(X, metadata=[{"cat": int(l)} for l in ds["labels"]])
        retrieve = db.engine.m2m.retrieve
        for lod in (0, 1, 2):
            nombre = f"hrm2 retrieve(q, k=10, lod={lod})"
            call = lambda q, lod=lod: retrieve(q, k=P.K, lod=lod)
            try:
                with P.RssSampler() as rs:
                    m = P.medir_single(call, Q)
                rec = P.recall_at_k([P.ids_desde_posiciones(db, call(q)[3])[0] for q in Q], gt)
                filas.append(P.fila(TABLA, meta, dataset, nombre, m, rec, rs.pico, build_s,
                                    camino="hrm2_lod", lod=lod))
            except Exception as exc:
                filas.append(P.nd(TABLA, meta, dataset, nombre, f"{type(exc).__name__}: {exc}"[:300]))
        del db


def bloque_sq8(filas, meta, tmp, etiqueta):
    from m2m.sq8_disk import SQ8DiskStore
    for n in (10000, 50000):
        kind = "clustered"
        dataset = P.dataset_fila(kind, n, D)
        print(f"[sq8 n={n}] generando...", flush=True)
        ds = P.generar(kind, n, D)
        X = ds["X"]
        Q = P.generar_queries(kind, ds)
        gt = P.ground_truth(X, Q)
        xp = os.path.join(tmp, f"p9_sq8_{n}_{etiqueta}.npy")
        np.save(xp, X)
        for modo, r in (("completo", None), ("espectral_auto", "auto")):
            nombre = f"sq8_disk modo={modo} search_batch(k=10, oversample=2)"
            dump = os.path.join(tmp, f"p9_sq8_{n}_{modo}_{etiqueta}")
            try:
                os.makedirs(dump, exist_ok=True)  # el store exige que el directorio exista
                with P.RssSampler() as rs:
                    t0 = time.perf_counter()
                    st = SQ8DiskStore.build_from_float_npy(xp, dump_dir=dump, spectral_r=r)
                    build_s = time.perf_counter() - t0
                    ram = st.ram_bytes() / 2 ** 20
                    disco = P.dir_mb(dump)
                    m = P.medir(lambda: st.search_batch(Q, k=P.K, oversample=2), len(Q))
                rec = P.recall_at_k(st.search_batch(Q, k=P.K, oversample=2).tolist(), gt)
                t0 = time.perf_counter()
                st2 = SQ8DiskStore.load(dump)
                load_s = time.perf_counter() - t0
                filas.append(P.fila(TABLA, meta, dataset, nombre, m, rec, rs.pico, build_s,
                                    camino="sq8", modo=modo, spectral_r=r,
                                    ram_arrays_mb=round(ram, 1), disco_mb=round(disco, 1),
                                    load_s=round(load_s, 3), n_items=len(st2)))
                del st, st2
            except Exception as exc:
                filas.append(P.nd(TABLA, meta, dataset, nombre, f"{type(exc).__name__}: {exc}"[:300]))


def _topk_lineal(X, Q):
    """Scan lineal batched: similitudes [B, n] = Q @ X.T, top-10 por fila."""
    return P.topk(Q @ X.T)


def bloque_hnsw_lsh(filas, meta):
    from m2m.hnsw_index import HNSWIndex
    from m2m.lsh_index import CrossPolytopeLSH, LSHConfig
    kind, n = "clustered", N_HNSW
    dataset = P.dataset_fila(kind, n, D)
    print(f"[hnsw_lsh n={n}] generando...", flush=True)
    ds = P.generar(kind, n, D)
    X = ds["X"]
    Q = P.generar_queries(kind, ds)
    gt = P.ground_truth(X, Q)

    with P.RssSampler() as rs:
        def rep():
            _topk_lineal(X, Q)

        m = P.medir(rep, len(Q))
    rec = P.recall_at_k(_topk_lineal(X, Q).tolist(), gt)
    filas.append(P.fila(TABLA, meta, dataset, "linear_batched B=200 (referencia)", m, rec, rs.pico,
                        None, camino="baseline"))

    try:
        with P.RssSampler() as rs:
            h = HNSWIndex(dim=D, seed=P.SEED_DATOS)
            t0 = time.perf_counter()
            h.build(X)
            build_s = time.perf_counter() - t0
            m = P.medir_single(lambda q: h.search(q, k=P.K), Q)
        rec = P.recall_at_k([h.search(q, k=P.K).indices.tolist() for q in Q], gt)
        filas.append(P.fila(TABLA, meta, dataset,
                            f"hnsw python (M={h.M}, ef_construction={h.ef_construction}, ef_search={h.ef_search})",
                            m, rec, rs.pico, build_s, camino="hnsw"))
        print(f"  hnsw build={build_s:.1f}s qps={m['qps']:.1f} rec={rec:.4f}", flush=True)
    except Exception as exc:
        filas.append(P.nd(TABLA, meta, dataset, "hnsw python", f"{type(exc).__name__}: {exc}"[:300]))

    try:
        with P.RssSampler() as rs:
            cfg = LSHConfig(dim=D)
            lsh = CrossPolytopeLSH(cfg)
            t0 = time.perf_counter()
            lsh.index(X)
            build_s = time.perf_counter() - t0
            m = P.medir_single(lambda q: lsh.query(q, k=P.K), Q)
        rec = P.recall_at_k([lsh.query(q, k=P.K)[0].tolist() for q in Q], gt)
        filas.append(P.fila(TABLA, meta, dataset,
                            f"lsh cross-polytope (tables={cfg.n_tables}, bits={cfg.n_bits}, probes={cfg.n_probes}, cand={cfg.n_candidates})",
                            m, rec, rs.pico, build_s, camino="lsh"))
        print(f"  lsh build={build_s:.1f}s qps={m['qps']:.1f} rec={rec:.4f}", flush=True)
    except Exception as exc:
        filas.append(P.nd(TABLA, meta, dataset, "lsh cross-polytope", f"{type(exc).__name__}: {exc}"[:300]))


GATE_CODE = r'''
import dataclasses, json, sys, time
import numpy as np
sys.path.insert(0, {bench!r})
import _protocol as P
from m2m.gate_modulus import gate_select_k
import m2m
X = P.generar({kind!r}, {n}, {d})["X"]
t0 = time.perf_counter()
sel = gate_select_k(X)
t = time.perf_counter() - t0
d = dataclasses.asdict(sel) if dataclasses.is_dataclass(sel) else dict(vars(sel))
def corto(v):
    if isinstance(v, dict):
        return {{k: corto(x) for k, x in list(v.items())[:40]}}
    if isinstance(v, (list, tuple)):
        return [corto(x) for x in v[:12]] if len(v) <= 12 else "<lista len=%d>" % len(v)
    if isinstance(v, np.ndarray):
        return "<array %s>" % (v.shape,)
    if isinstance(v, float):
        return None if v != v else round(v, 6)
    if isinstance(v, (int, str, bool)) or v is None:
        return v
    return str(v)[:120]
print("P9JSON " + json.dumps({{"t_s": t, "m2m": m2m.__file__, "seleccion": corto(d)}}, default=str))
'''


def bloque_gate(filas, meta, kinds=("clustered", "uniform")):
    for kind in kinds:
        dataset = P.dataset_fila(kind, N_GATE, D)
        nombre = "gate_select_k(X) target_docs_per_cluster=25"
        print(f"[gate kind={kind} n={N_GATE}]...", flush=True)
        code = GATE_CODE.format(bench=os.path.dirname(os.path.abspath(__file__)), kind=kind,
                                n=N_GATE, d=D)
        try:
            t0 = time.perf_counter()
            proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                  timeout=GATE_TIMEOUT_S, env=os.environ.copy())
            linea = next((l for l in proc.stdout.splitlines() if l.startswith("P9JSON ")), None)
            if linea is None:
                raise RuntimeError(f"rc={proc.returncode} stderr={proc.stderr.strip()[-300:]}")
            out = json.loads(linea[len("P9JSON "):])
            if meta["m2m_file"] and not out["m2m"].endswith(meta["m2m_file"]):
                raise RuntimeError(f"el subproceso importó otro árbol: {out['m2m']}")
            m = P.resumen_de_tiempos([out["t_s"]], None,
                                     qps_definicion="no aplica (tiempo total del veredicto)")
            filas.append(P.fila(TABLA, meta, dataset, nombre, m, None, None, None, camino="gate",
                                veredicto=out["seleccion"], timeout_s=GATE_TIMEOUT_S,
                                carga_cpu_sistema_pct=None))
            sel = out["seleccion"]
            print(f"  t={out['t_s']:.2f}s k={sel.get('k')} jerarquico={sel.get('jerarquico')}", flush=True)
        except subprocess.TimeoutExpired:
            filas.append(P.nd(TABLA, meta, dataset, nombre,
                              f"timeout de {GATE_TIMEOUT_S}s (una sola llamada; fuera del presupuesto de cómputo)"))
        except Exception as exc:
            filas.append(P.nd(TABLA, meta, dataset, nombre, f"{type(exc).__name__}: {exc}"[:300]))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--etiqueta", default="")
    ap.add_argument("--bloques", default="hrm2_lod,sq8,hnsw_lsh,gate",
                    help="subset separado por comas: hrm2_lod,sq8,hnsw_lsh,gate")
    ap.add_argument("--gate-kinds", default="clustered,uniform")
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
    for bloque in [b.strip() for b in args.bloques.split(",") if b.strip()]:
        if bloque == "hrm2_lod":
            bloque_hrm2_lod(filas, meta)
        elif bloque == "sq8":
            bloque_sq8(filas, meta, tmp, etiqueta)
        elif bloque == "hnsw_lsh":
            bloque_hnsw_lsh(filas, meta)
        elif bloque == "gate":
            bloque_gate(filas, meta, tuple(k for k in args.gate_kinds.split(",") if k))
        else:
            raise SystemExit(f"bloque desconocido: {bloque}")
    destino = P.escribir(TABLA, filas, meta, out_dir=args.out_dir, etiqueta=args.etiqueta,
                         bloques=args.bloques, gate_timeout_s=GATE_TIMEOUT_S,
                         n_hnsw_lsh=N_HNSW, n_gate=N_GATE)
    print(f"commit={meta['commit_corto']} etiqueta={args.etiqueta!r}")
    P.resumen(filas)
    print("->", destino)


if __name__ == "__main__":
    main()
