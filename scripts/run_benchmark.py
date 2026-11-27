#!/usr/bin/env python
"""Benchmark nocturno (artifact de CI, SIN gates) — audit r4-05.

Mide con el protocolo común (``bench/_protocol.py``): datasets clustered y uniforme de
``n_vectors`` x ``--dim`` (default 384), ``search(q, k)`` (legacy y con metadata),
``search_batch`` B=200 y un scan lineal de referencia; calentamiento + repeticiones,
recall@10 contra fuerza bruta float64 y meta (commit del árbol de ``m2m``, hilos, uptime).
Los umbrales son INFORMATIVOS: si midió, sale 0; compara contra la corrida previa si existe.

Uso:  python scripts/run_benchmark.py [n_vectors] [--dim D] [--out FILE]
Si OPENBLAS_NUM_THREADS / OMP_NUM_THREADS no son 1, se re-ejecuta con 1 hilo (protocolo).
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "bench"))
import _protocol as P  # noqa: E402


def rehilos():
    """El protocolo exige BLAS en 1 hilo: si no, se re-ejecuta con las variables puestas."""
    env = dict(os.environ)
    cambiar = {v: "1" for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS") if env.get(v) != "1"}
    if cambiar:
        env.update(cambiar)
        print(f"[run_benchmark] re-ejecutando con {cambiar} (protocolo bench/_protocol.py)")
        os.execve(sys.executable, [sys.executable, os.path.abspath(__file__), *sys.argv[1:]], env)


def umbral(nombre, valor, limite):
    if valor is None:
        return f"  {nombre}: n/d"
    return (f"  {nombre}: {valor:.4f} (referencia >= {limite}) -> "
            f"{'ok' if valor >= limite else 'POR DEBAJO (informativo, no es un gate)'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("n_vectors", nargs="?", type=int, default=10000)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--out", default=os.path.join(ROOT, "benchmark_results.json"))
    a = ap.parse_args()
    rehilos()
    import m2m  # noqa: E402  (del árbol indicado por PYTHONPATH)
    import numpy as np  # noqa: E402

    meta = P.meta(m2m)
    filas, datasets = [], []
    for kind in ("clustered", "uniform"):
        ds = P.generar(kind, a.n_vectors, a.dim)
        X = ds["X"]
        Q = P.generar_queries(kind, ds)
        gt = P.ground_truth(X, Q)
        dataset = P.dataset_fila(kind, a.n_vectors, a.dim)
        db, build_s = P.construir_db(X, metadata=[{"cat": int(l)} for l in ds["labels"]])
        B = min(200, len(Q))
        Qb, gtb = Q[:B], gt[:B]

        def single(nombre, call, ids_de):
            m = P.medir_single(call, Q)
            filas.append(P.fila("run_benchmark", meta, dataset, nombre, m,
                                P.recall_at_k([ids_de(call(q)) for q in Q], gt), None, build_s))

        def batch(nombre, call, ids_de):
            def rep():
                call(Qb)

            m = P.medir(rep, B)
            filas.append(P.fila("run_benchmark", meta, dataset, nombre, m,
                                P.recall_at_k(ids_de(call(Qb)), gtb), None, build_s, B=B))

        single("linear_single (referencia)", lambda q: P.topk((X @ q)[np.newaxis, :]),
               lambda r: r[0].tolist())
        single("search(q,k) legacy", lambda q: db.search(q, k=P.K),
               lambda r: P.ids_desde_posiciones(db, r[3])[0])
        single("search(q,k) include_metadata=True", lambda q: db.search(q, k=P.K, include_metadata=True),
               P.ids_docresult)
        batch("search_batch B=200", lambda Q_: db.search_batch(Q_, k=P.K),
              lambda r: P.ids_desde_posiciones(db, r))
        datasets.append({**dataset, "build_s": round(build_s, 3), **P.info_indice(db)})
        del db

    previa = None
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as f:
            previa = json.load(f)
    doc = {"tabla": "run_benchmark", "protocolo": P.PROTOCOLO, "meta": P.cerrar_meta(meta),
           "resultados": filas, "datasets": datasets}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1, ensure_ascii=False, default=P._json_default)
    print(f"commit {meta['commit_corto']} hilos {meta['hilos']} -> {a.out}")
    P.resumen(filas)
    print("Umbrales informativos (no son gates):")
    for f in filas:
        if f.get("estado") == P.ND:
            continue
        if f["dataset"]["kind"] == "clustered" and "legacy" in f["parametro"]:
            print(umbral("recall@10 search() clustered", f["recall@10"], 0.9))
    if previa:
        for f in filas:
            for p in previa.get("resultados", []):
                if p["parametro"] == f["parametro"] and p["dataset"] == f["dataset"] \
                        and p.get("qps") and f.get("qps"):
                    print(f"  vs corrida previa ({p['commit']}): {f['parametro']} "
                          f"{f['dataset']['kind']}: {100 * (f['qps'] / p['qps'] - 1):+.1f}% QPS, "
                          f"recall {p['recall@10']} -> {f['recall@10']} (informativo)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
