#!/usr/bin/env python
"""Cifras de aprendizaje online y SOC del README, reproducidas desde el código ACTUAL.

Setups deterministas copiados de ``tests/test_audit_learning_soc.py`` (mismos tamaños, semillas y
llamadas). No mide tiempo: registra valores. Salida: ``secciones`` (bloques estructurados) y
``resultados`` (filas planas antes → después):

  (a) ``feedback``: 400 docs clustered (seed 6), 100 queries (rng seed 7); 300 × ``db.feedback(q,
      relevant_ids=[target])`` con target = 3er resultado de q. Rank y score del target antes /
      después y recall@10 de las otras 99 queries antes / después (GT = fuerza bruta sobre X).
  (b) ``mu``: ``OnlineUpdater(lr_mu=0.01)`` sobre 8 splats d=64 (seed 0); 500 feedbacks relevantes
      con queries de la cuenca del splat 0 (rng seed 1): ‖μ‖ antes / después. Además el camino
      público ``db.feedback`` 500 × sobre ``SimpleVectorDB(32, enable_ebm=True)`` con 20 docs.
  (c) ``criticality``: ``AdvancedVectorDB(32)``, 200 docs (seed 14), 60 × ``feedback(x[0],
      irrelevant_ids=ids[:150])`` → 75 % de splats de amplitud baja; ``check_criticality().index``
      y estado antes / después.
  (d) ``relax``: 60 docs random (seed 15): F antes / después de ``relax(iterations=10)``; ídem
      store sobre-afilado (κ = 100, seed 16).
  (e) ``maybe_consolidate``: 60 docs (seed 21), 60 × feedback irrelevante sobre 45: triggered,
      removed, índice y estado después, y el resultado de la 2ª llamada.

Los valores «antes del fix» (‖μ‖ 0.073, índice 0.5649) NO están: ese código ya no existe en este
árbol y no son reproducibles desde acá.

Uso (desde la raíz del árbol, con OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1):
    PYTHONPATH=$PWD/src python bench/bench_learning_soc.py [--out-dir DIR]
Salida: benchmarks/results/bench_learning.json
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _protocol as P  # noqa: E402

TABLA = "bench_learning"


# --------------------------------------------------------------------------- helpers (tests)
def _unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _random_unit(n, d, seed):
    rng = np.random.default_rng(seed)
    return _unit(rng.normal(size=(n, d)).astype(np.float32))


def _clustered(n, d, seed=0, k=8, noise=0.25):
    rng = np.random.default_rng(seed)
    centers = _unit(rng.normal(size=(k, d)).astype(np.float32))
    labels = rng.integers(0, k, size=n)
    x = centers[labels] + noise * rng.normal(size=(n, d)).astype(np.float32)
    return _unit(x), labels


def _norms(a):
    return np.linalg.norm(np.asarray(a, dtype=np.float64), axis=-1)


def _brute_topk(x, q, k):
    sims = x @ q
    return [int(i) for i in np.argsort(-sims)[:k]]


# --------------------------------------------------------------------------- salida
def fila(bloque, medicion, antes, despues, **extra):
    return {"tabla": TABLA, "bloque": bloque, "medicion": medicion, "antes": antes, "despues": despues, **extra}


def r(x, n=4):
    return round(float(x), n)


def reporte(rep):
    return {"state": rep.state.value, "index": float(rep.index), "n_clusters": int(rep.n_clusters),
            "decayed_fraction": float(rep.decayed_fraction), "kappa_dispersion": float(rep.kappa_dispersion),
            "components": {k: float(v) for k, v in rep.components.items()}}


# --------------------------------------------------------------------------- (a) feedback
def sec_feedback(filas):
    import m2m
    x, _ = _clustered(400, 64, seed=6)
    ids = [f"d{i}" for i in range(400)]
    db = m2m.SimpleVectorDB(latent_dim=64, enable_ebm=True)
    db.add(ids=ids, vectors=x)
    rng = np.random.default_rng(7)
    q_idx = rng.choice(400, 100, replace=False)
    queries = _unit(x[q_idx] + 0.15 * rng.normal(size=(100, 64)).astype(np.float32))
    q = queries[0]
    others = list(range(1, 100))

    def recall():
        hits = tot = 0
        for j in others:
            got = {res.id for res in db.search(queries[j], k=10, include_metadata=True)}
            truth = {ids[i] for i in _brute_topk(x, queries[j], 10)}
            hits += len(got & truth)
            tot += 10
        return hits / tot

    before = db.search(q, k=10, include_metadata=True)
    target = before[2].id  # el 3er resultado: el rank puede mejorar
    rank0 = 2
    score0 = before[2].score
    recall0 = recall()
    for _ in range(300):
        db.feedback(q, relevant_ids=[target])
    after = db.search(q, k=10, include_metadata=True)
    ids_after = [res.id for res in after]
    rank1 = ids_after.index(target) if target in ids_after else len(ids_after)
    score1 = next(res.score for res in after if res.id == target)
    recall1 = recall()
    st = db.engine.m2m.splats
    n = st.n_active
    normas = _norms(st.mu[:n])
    sec = {"n_docs": 400, "d": 64, "seed_datos": 6, "seed_queries": 7, "n_queries": 100,
           "n_feedbacks": 300, "relevant_ids": [target], "target": target,
           "score_definicion": "DocResult.score de search(q, k=10, include_metadata=True)",
           "rank": {"antes": rank0, "despues": rank1},
           "score": {"antes": float(score0), "despues": float(score1)},
           "recall@10_otras_99_queries": {"antes": recall0, "despues": recall1},
           "norma_mu_min_max_despues": [float(normas.min()), float(normas.max())]}
    filas.append(fila("feedback", "rank del documento target para su query", rank0, rank1, target=target))
    filas.append(fila("feedback", "score del target para su query", r(score0), r(score1), target=target))
    filas.append(fila("feedback", "recall@10 de las otras 99 queries (GT fuerza bruta)", r(recall0), r(recall1)))
    return sec


# --------------------------------------------------------------------------- (b) ‖μ‖
def sec_mu(filas):
    import m2m
    from m2m.online_updates import FeedbackEvent, OnlineUpdater
    rng = np.random.default_rng(0)
    mu = _unit(rng.normal(size=(8, 64)).astype(np.float32))
    alpha = np.ones(8, dtype=np.float32)
    kappa = np.full(8, 10.0, dtype=np.float32)
    mu0 = mu[0].copy()
    norma0 = float(_norms(mu[0]))
    rng = np.random.default_rng(1)
    upd = OnlineUpdater(lr_mu=0.01)
    for _ in range(500):
        # queries unitarias de la cuenca del splat (como las que lo encuentran relevante)
        q = _unit(mu0 + 0.3 * rng.normal(size=mu.shape[1]).astype(np.float32))
        upd.apply_feedback(FeedbackEvent(query=q, splat_index=0, relevant=True), mu, alpha, kappa)
    norma1 = float(_norms(mu[0]))
    standalone = {"n_splats": 8, "d": 64, "seed_splats": 0, "seed_queries": 1, "lr_mu": 0.01,
                  "n_feedbacks": 500, "norma_mu_antes": norma0, "norma_mu_despues": norma1,
                  "cos_mu_antes_despues": float(mu0 @ mu[0]), "alpha_despues": float(alpha[0]),
                  "kappa_despues": float(kappa[0]), "alpha_min": float(upd.alpha_min), "alpha_max": float(upd.alpha_max)}

    # camino público: SimpleVectorDB.feedback por ids de documento
    db = m2m.SimpleVectorDB(latent_dim=32, enable_ebm=True)
    x = _random_unit(20, 32, seed=3)
    db.add(ids=[f"d{i}" for i in range(20)], vectors=x)
    rng = np.random.default_rng(4)
    for _ in range(500):
        q = _unit(x[0] + 0.3 * rng.normal(size=32).astype(np.float32))
        db.feedback(q, relevant_ids=["d0", "d1"])
    st = db.engine.m2m.splats
    n = st.n_active
    normas = _norms(st.mu[:n])
    publico = {"n_docs": 20, "d": 32, "seed_datos": 3, "seed_queries": 4, "n_feedbacks": 500,
               "relevant_ids": ["d0", "d1"], "norma_mu_antes": 1.0,
               "norma_mu_min_despues": float(normas.min()), "norma_mu_max_despues": float(normas.max()),
               "alpha_max_despues": float(st.alpha[:n].max()), "kappa_max_despues": float(st.kappa[:n].max())}
    filas.append(fila("mu", "‖μ‖ del splat reforzado tras 500 feedbacks relevantes "
                           "(OnlineUpdater lr_mu=0.01, 8 splats d=64)", r(norma0, 6), r(norma1, 6)))
    filas.append(fila("mu", "‖μ‖ mín / máx de los 20 splats tras 500 db.feedback (d0, d1 relevantes)",
                      [1.0, 1.0], [r(normas.min(), 6), r(normas.max(), 6)]))
    return {"online_updater": standalone, "db_feedback": publico}


# --------------------------------------------------------------------------- (c) criticality
def sec_criticality(filas):
    import m2m
    db = m2m.AdvancedVectorDB(latent_dim=32)
    x, _ = _clustered(200, 32, seed=14)
    ids = [f"d{i}" for i in range(200)]
    db.add(ids=ids, vectors=x)
    rep0 = db.check_criticality()
    for _ in range(60):
        db.feedback(x[0], irrelevant_ids=ids[:150])
    rep = db.check_criticality()
    sec = {"n_docs": 200, "d": 32, "seed_datos": 14, "n_feedbacks": 60, "irrelevantes_por_feedback": 150,
           "fraccion_splats_amplitud_baja": float(rep.decayed_fraction),
           "antes": reporte(rep0), "despues": reporte(rep), "n_docs_activos_despues": len(db._vectors)}
    filas.append(fila("criticality", "check_criticality().index — store con 75 % de splats de α baja",
                      r(rep0.index), r(rep.index), estado_antes=rep0.state.value, estado_despues=rep.state.value,
                      fraccion_amplitud_baja=r(rep.decayed_fraction)))
    return sec


# --------------------------------------------------------------------------- (d) relax
def _relax(seed, sobrefilar):
    import m2m
    db = m2m.AdvancedVectorDB(latent_dim=32)
    x = _random_unit(60, 32, seed=seed)
    db.add(ids=[f"d{i}" for i in range(60)], vectors=x)
    if sobrefilar:
        st = db.engine.m2m.splats
        st.kappa[: st.n_active] = 100.0  # sobre-afilado (lo que satura el feedback relevante)
        db._update_ebm_splats()
    f0 = db.get_stats()["free_energy"]
    res = db.relax(iterations=10)
    f1 = db.get_stats()["free_energy"]
    return {"seed": seed, "kappa_inicial": 100.0 if sobrefilar else 10.0, "iterations": int(res.iterations),
            "improved": bool(res.improved), "F_antes": float(f0), "F_despues": float(f1),
            "energy_delta": float(res.energy_delta), "initial_energy": float(res.initial_energy),
            "final_energy": float(res.final_energy)}


def sec_relax(filas):
    fresco = _relax(15, sobrefilar=False)
    afilado = _relax(16, sobrefilar=True)
    filas.append(fila("relax", "free_energy (get_stats) tras relax(iterations=10) — store fresco, seed 15",
                      r(fresco["F_antes"], 3), r(fresco["F_despues"], 3), improved=fresco["improved"]))
    filas.append(fila("relax", "free_energy (get_stats) tras relax(iterations=10) — store sobre-afilado (κ=100), seed 16",
                      r(afilado["F_antes"], 3), r(afilado["F_despues"], 3), improved=afilado["improved"]))
    return {"store_fresco": fresco, "store_sobreafilado_kappa_100": afilado, "n_docs": 60, "d": 32}


# --------------------------------------------------------------------------- (e) maybe_consolidate
def sec_maybe_consolidate(filas):
    import m2m
    db = m2m.AdvancedVectorDB(latent_dim=32)
    x, _ = _clustered(60, 32, seed=21)
    ids = [f"d{i}" for i in range(60)]
    db.add(ids=ids, vectors=x)
    for _ in range(60):
        db.feedback(x[0], irrelevant_ids=ids[:45])
    n_antes = len(db._vectors)
    rep0 = db.check_criticality()

    def decision(dec, n_docs):
        return {"triggered": bool(dec.triggered), "removed": int(dec.removed), "state": dec.state.value,
                "index": float(dec.index), "threshold": float(dec.threshold), "n_docs": int(n_docs)}

    dec = db.maybe_consolidate()
    primera = decision(dec, len(db._vectors))
    dec2 = db.maybe_consolidate()
    segunda = decision(dec2, len(db._vectors))
    sec = {"n_docs": n_antes, "d": 32, "seed_datos": 21, "n_feedbacks": 60, "irrelevantes_por_feedback": 45,
           "index_antes": float(rep0.index), "state_antes": rep0.state.value,
           "llamada_1": primera, "llamada_2": segunda}
    filas.append(fila("maybe_consolidate", "check_criticality().index antes → después de la 1ª llamada",
                      r(rep0.index), r(primera["index"]), estado_antes=rep0.state.value, estado_despues=primera["state"]))
    filas.append(fila("maybe_consolidate", "documentos activos antes → después de la 1ª llamada", n_antes, primera["n_docs"]))
    filas.append(fila("maybe_consolidate", "triggered: 1ª llamada → 2ª llamada", primera["triggered"], segunda["triggered"]))
    filas.append(fila("maybe_consolidate", "removed: 1ª llamada → 2ª llamada", primera["removed"], segunda["removed"]))
    filas.append(fila("maybe_consolidate", "index después: 1ª llamada → 2ª llamada", r(primera["index"]), r(segunda["index"]),
                      estado_despues_1=primera["state"], estado_despues_2=segunda["state"]))
    return sec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=None, help="directorio de salida (default benchmarks/results/)")
    args = ap.parse_args()
    P.requerir_hilos_1()
    import m2m  # noqa: E402  (del árbol indicado por PYTHONPATH)

    meta = P.meta(m2m)
    print(f"m2m={meta['m2m_file']} arbol={meta['arbol']} commit={meta['commit_corto']} "
          f"src_sucio={meta['src_sucio']} hilos={meta['hilos']}", flush=True)
    filas = []
    secciones = {}
    for clave, fn in (("a_feedback", sec_feedback), ("b_norma_mu", sec_mu), ("c_criticality", sec_criticality),
                      ("d_relax", sec_relax), ("e_maybe_consolidate", sec_maybe_consolidate)):
        print(f"[{clave}] ...", flush=True)
        secciones[clave] = fn(filas)
    for f in filas:
        print(f"  [{f['bloque']}] {f['medicion']}: {f['antes']} -> {f['despues']}")
    destino = P.escribir(
        TABLA, filas, meta, out_dir=args.out_dir, secciones=secciones,
        nota=("valores del código ACTUAL con setups deterministas (tests/test_audit_learning_soc.py); "
              "no hay timings. Los valores «antes del fix» (‖μ‖ 0.073, índice 0.5649) no son "
              "reproducibles desde este árbol y no están en este archivo"),
        filas_definicion="antes / despues = valor antes y después de la operación de la sección; ver secciones",
    )
    print("->", destino)


if __name__ == "__main__":
    main()
