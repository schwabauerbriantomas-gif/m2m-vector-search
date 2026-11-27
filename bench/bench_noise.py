#!/usr/bin/env python
"""Ruido entre corridas de la tabla de búsqueda — hace verificable la sección «Noise between runs».

Compara fila por fila —misma clave (kind, n, d) + ``parametro``— dos corridas de
``bench/bench_search.py`` con el mismo comportamiento de código:

  * corrida 1: ``benchmarks/results/bench_search_run1_2109459.json`` (= ``git show
    bfdca29:benchmarks/results/bench_search.json``, commit de medición 2109459);
  * corrida 2: ``benchmarks/results/bench_search.json`` (commit de medición 626e4ff).

Por fila: ``qps1``, ``qps2``, ``delta_pct = 100 · (qps2 − qps1) / qps1``, ``dentro_umbral`` y si el
recall@10 es idéntico. Resumen: filas comparadas, filas dentro de ±10 %, delta máximo absoluto y su
fila. NO mide nada: sólo lee los dos JSON (ambos con el protocolo de ``bench/_protocol.py``).

Uso (desde la raíz):  python bench/bench_noise.py [--run1 ARCHIVO] [--run2 ARCHIVO] [--out-dir DIR]
Salida: benchmarks/results/bench_noise.json — la sección «Noise between runs» de docs/PERFORMANCE.md y
la nota «Noise» de docs/BENCHMARKS.md se leen de ahí.
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _protocol as P  # noqa: E402

TABLA = "bench_noise"
RUN1 = "bench_search_run1_2109459.json"
RUN2 = "bench_search.json"
UMBRAL_PCT = 10.0


def cargar(ruta):
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def clave(f):
    d = f["dataset"]
    return (d["kind"], int(d["n"]), int(d["d"]), str(f["parametro"]))


def fila_clave(c):
    return {"kind": c[0], "n": c[1], "d": c[2], "parametro": c[3]}


def info_corrida(doc, ruta):
    m = doc["meta"]
    return {
        "archivo": os.path.relpath(ruta, P.RAIZ),
        "sha256": sha256(ruta),
        "tabla_origen": doc.get("tabla"),
        "commit": m.get("commit"),
        "commit_corto": m.get("commit_corto"),
        "hilos": m.get("hilos"),
        "uptime_antes": m.get("uptime_antes"),
        "uptime_despues": m.get("uptime_despues"),
        "loadavg_antes": m.get("loadavg_antes"),
        "loadavg_despues": m.get("loadavg_despues"),
        "n_filas": len(doc["resultados"]),
    }


def comparar(run1, run2):
    idx = []
    for doc in (run1, run2):
        d = {}
        for f in doc["resultados"]:
            if f.get("estado") == P.ND or f.get("qps") is None:
                continue
            c = clave(f)
            if c in d:
                raise SystemExit(f"bench_noise: clave duplicada {c}")
            d[c] = f
        idx.append(d)
    comunes = [c for c in idx[1] if c in idx[0]]
    solo = {"solo_run1": [fila_clave(c) for c in idx[0] if c not in idx[1]],
            "solo_run2": [fila_clave(c) for c in idx[1] if c not in idx[0]]}
    filas = []
    for c in comunes:
        f1, f2 = idx[0][c], idx[1][c]
        q1, q2 = float(f1["qps"]), float(f2["qps"])
        delta = 100.0 * (q2 - q1) / q1
        r1, r2 = f1.get("recall@10"), f2.get("recall@10")
        filas.append({
            "tabla": TABLA,
            "dataset": {"kind": c[0], "n": c[1], "d": c[2]},
            "parametro": c[3],
            "camino": f2.get("camino"),
            "qps1": q1,
            "qps2": q2,
            "delta_pct": round(delta, 2),
            "delta_abs_pct": round(abs(delta), 2),
            "dentro_umbral": bool(abs(delta) <= UMBRAL_PCT),
            "recall1": r1,
            "recall2": r2,
            "recall_identico": bool(r1 == r2),
            "n_repeticiones_1": f1.get("n_repeticiones"),
            "n_repeticiones_2": f2.get("n_repeticiones"),
        })
    return filas, solo


def resumen_de(filas, solo):
    if not filas:
        raise SystemExit("bench_noise: ninguna fila común entre las dos corridas")
    deltas = np.asarray([f["delta_abs_pct"] for f in filas], dtype=np.float64)
    maxima = max(filas, key=lambda f: f["delta_abs_pct"])
    fuera = [f for f in filas if not f["dentro_umbral"]]
    rec_dist = [fila_clave(clave(f)) if False else {"dataset": f["dataset"], "parametro": f["parametro"],
                                                    "recall1": f["recall1"], "recall2": f["recall2"]}
                for f in filas if not f["recall_identico"]]
    return {
        "umbral_pct": UMBRAL_PCT,
        "delta_definicion": "delta_pct = 100 * (qps2 - qps1) / qps1; dentro_umbral = |delta_pct| <= umbral_pct",
        "filas_comparadas": len(filas),
        "filas_dentro_umbral": int(sum(1 for f in filas if f["dentro_umbral"])),
        "filas_fuera_umbral": len(fuera),
        "delta_abs_mediana_pct": round(float(np.median(deltas)), 2),
        "delta_abs_p90_pct": round(float(np.percentile(deltas, 90)), 2),
        "delta_max_abs_pct": maxima["delta_abs_pct"],
        "fila_delta_max": {"dataset": maxima["dataset"], "parametro": maxima["parametro"],
                           "qps1": maxima["qps1"], "qps2": maxima["qps2"], "delta_pct": maxima["delta_pct"]},
        "filas_fuera_umbral_detalle": [{"dataset": f["dataset"], "parametro": f["parametro"],
                                        "qps1": f["qps1"], "qps2": f["qps2"], "delta_pct": f["delta_pct"]}
                                       for f in sorted(fuera, key=lambda f: -f["delta_abs_pct"])],
        "recall_identico_en_todas": not rec_dist,
        "filas_recall_distinto": rec_dist,
        "filas_solo_en_una_corrida": solo,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run1", default=os.path.join(P.RESULTS, RUN1))
    ap.add_argument("--run2", default=os.path.join(P.RESULTS, RUN2))
    ap.add_argument("--out-dir", default=None, help="directorio de salida (default benchmarks/results/)")
    args = ap.parse_args()
    run1, run2 = cargar(args.run1), cargar(args.run2)
    for nombre, doc in (("run1", run1), ("run2", run2)):
        if doc.get("tabla") != "bench_search":
            raise SystemExit(f"bench_noise: {nombre} no es una tabla bench_search ({doc.get('tabla')!r})")
    filas, solo = comparar(run1, run2)
    resumen = resumen_de(filas, solo)
    meta = P.meta()  # este script no mide: el commit es el del árbol que contiene los JSON
    destino = P.escribir(
        TABLA, filas, meta, out_dir=args.out_dir, resumen=resumen,
        corridas={"run1": info_corrida(run1, args.run1), "run2": info_corrida(run2, args.run2)},
        protocolo_igual=bool(run1.get("protocolo") == run2.get("protocolo")),
        nota=("este archivo no contiene una medición propia: compara dos corridas ya archivadas de "
              "bench/bench_search.py (misma clave dataset+parametro); los dos commits de medición "
              "tienen el mismo comportamiento de código en los caminos medidos"),
    )
    print(f"run1 commit={run2 and run1['meta'].get('commit_corto')}  run2 commit={run2['meta'].get('commit_corto')}")
    print(f"filas comparadas={resumen['filas_comparadas']}  dentro de ±{UMBRAL_PCT:g} %={resumen['filas_dentro_umbral']}"
          f"  |Δ| mediana={resumen['delta_abs_mediana_pct']} %  p90={resumen['delta_abs_p90_pct']} %")
    print(f"Δ máximo absoluto={resumen['delta_max_abs_pct']} % en {resumen['fila_delta_max']}")
    print(f"recall idéntico en todas: {resumen['recall_identico_en_todas']}  solo_en_una_corrida={solo}")
    for f in sorted(filas, key=lambda f: -f["delta_abs_pct"])[:5]:
        print(f"  {f['delta_pct']:+7.2f} %  {f['dataset']['kind']} {f['dataset']['n']}x{f['dataset']['d']}  "
              f"{f['parametro']}  {f['qps1']:.1f} -> {f['qps2']:.1f}")
    print("->", destino)


if __name__ == "__main__":
    main()
