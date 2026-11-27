"""P17 — búsqueda con filtro EXACTA en el camino IVF/HRM2 (cota de poda por cluster).

Antes: `_batch_topk_per_query` sondeaba sólo n_probe clusters y expandía sólo si el
filtro dejaba < 1.5*k candidatos; con filtros correlacionados con los clusters el
recall@10 caía (0.458 / 0.701 en clustered 10k x 384). Ahora: los clusters se
recorren en orden de cercanía y sólo se saltea uno si
``(dist(q, c) - r_c)^2 >= d2_k`` (cota exacta), con ``r_c`` mantenido en cada mutación.
"""

import numpy as np
import pytest

import m2m

N, D, K = 3500, 32, 10
C = 59  # clusters verdaderos del dataset (= round(sqrt(N)))
SEED = 1234


def _norm(X):
    X = np.asarray(X, dtype=np.float64)
    return (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)


def _puntos(rng, centros, labels):
    return _norm(centros[labels] + rng.standard_normal((len(labels), D)))


def _meta(labels, offset, pasan25, pasan5):
    return [
        {
            "cat": int(labels[j]),
            "bucket": int((offset + j) % 4),
            "bucket20": int((offset + j) % 20),
            "cl25": int(labels[j] in pasan25),
            "cl5": int(labels[j] in pasan5),
        }
        for j in range(len(labels))
    ]


FILTROS = [
    ("nocorr_25", {"bucket": 0}, lambda m: m["bucket"] == 0),
    ("nocorr_5", {"bucket20": 0}, lambda m: m["bucket20"] == 0),
    ("corr_25", {"cl25": 1}, lambda m: m["cl25"] == 1),
    ("corr_5", {"cl5": 1}, lambda m: m["cl5"] == 1),
]


def _gt(X_cur, alive, Q, mask, k=K):
    """Fuerza bruta float64 sobre los docs vivos que pasan `mask`; ids = índice del doc."""
    Xv = np.asarray([x for x, a in zip(X_cur, alive) if a], dtype=np.float32)
    idx_v = np.array([i for i, a in enumerate(alive) if a])
    S = Q.astype(np.float64) @ Xv.T
    S[:, ~mask[idx_v]] = -np.inf
    out = []
    for fila in S:
        validos = np.flatnonzero(np.isfinite(fila))
        orden = validos[np.argsort(-fila[validos], kind="stable")][:k]
        out.append([int(idx_v[i]) for i in orden])
    return out


def _ids_single(db, q, filtro):
    return [int(str(x.id)[len("doc") :]) for x in db.search(q, k=K, filter=filtro)]


def _ids_batch(db, Q, **kw):
    orden = db._splat_id_order
    filas = np.atleast_2d(np.asarray(db.search_batch_filtered(Q, k=K, **kw)))
    return [[int(str(orden[int(p)])[len("doc") :]) for p in fila if int(p) >= 0] for fila in filas]


def _check(db, X_cur, alive, meta_cur, Q, etiqueta):
    for nombre, filtro, pred in FILTROS:
        mask = np.array([a and pred(m) for a, m in zip(alive, meta_cur)], dtype=bool)
        assert mask.any()
        gt = _gt(X_cur, alive, Q, mask)
        s = [_ids_single(db, q, filtro) for q in Q]
        assert s == gt, f"[{etiqueta}] search(filter=dict) {nombre}: ids/orden != fuerza bruta"
        d = _ids_batch(db, Q, filter=filtro)
        assert d == gt, f"[{etiqueta}] search_batch_filtered(filter=dict) {nombre}: != fuerza bruta"
        pasan = {i for i in range(len(alive)) if mask[i]}
        ids_por_pos = [str(x)[len("doc") :] for x in db._splat_id_order]

        def fn(p, pasan=pasan, ids_por_pos=ids_por_pos):
            # filter_fn recibe POSICIONES de splat, no índices de doc
            p = int(p)
            return 0 <= p < len(ids_por_pos) and int(ids_por_pos[p]) in pasan

        f = _ids_batch(db, Q, filter_fn=fn)
        assert (
            f == gt
        ), f"[{etiqueta}] search_batch_filtered(filter_fn=callable) {nombre}: != fuerza bruta"
    _check_radii(db, etiqueta)


def _check_radii(db, etiqueta):
    """`coarse_cluster_radii` debe existir tras una búsqueda con filtro y ser COTA
    SUPERIOR de la distancia centroide->fila para TODAS las filas asignadas."""
    store = db.engine.m2m.splats
    eng = store.engine
    radii = getattr(eng, "coarse_cluster_radii", None)
    assert radii is not None, f"[{etiqueta}] cache de radios sin calcular"
    cent = np.asarray(eng.coarse_model.centroids_, dtype=np.float64)
    mu = store.mu[: store.n_active].astype(np.float64)
    for cid, rows in eng.coarse_cluster_indices.items():
        if len(rows) == 0:
            continue
        real = float(np.sqrt(((mu[rows] - cent[cid]) ** 2).sum(axis=1)).max())
        assert (
            radii[cid] >= real - 1e-6
        ), f"[{etiqueta}] r[{cid}]={radii[cid]:.6f} < distancia real máxima {real:.6f} (no es cota)"


class Estado:
    def __init__(self, X, meta):
        self.X_cur = [x for x in X]
        self.meta_cur = list(meta)
        self.alive = [True] * len(X)


@pytest.fixture(scope="module")
def datos():
    rng = np.random.default_rng(SEED)
    centros = rng.standard_normal((C, D)) * 2.0
    labels = rng.integers(0, C, size=N)
    X = _puntos(rng, centros, labels)
    unicos = np.unique(labels)
    pasan25, pasan5 = set(unicos[::4].tolist()), set(unicos[::20].tolist())
    meta = _meta(labels, 0, pasan25, pasan5)
    qlabels = rng.integers(0, C, size=40)
    Q = _puntos(rng, centros, qlabels)
    return dict(
        rng=rng, centros=centros, X=X, labels=labels, meta=meta, Q=Q, pasan25=pasan25, pasan5=pasan5
    )


def _nuevo_estado(rng, centros, labels_nuevas, offset, pasan25, pasan5):
    xn = _puntos(rng, centros, labels_nuevas)
    return xn, _meta(labels_nuevas, offset, pasan25, pasan5)


def test_filtro_exacto_y_tras_cada_mutacion(datos, tmp_path, monkeypatch):
    rng, centros, Q = datos["rng"], datos["centros"], datos["Q"]
    pasan25, pasan5 = datos["pasan25"], datos["pasan5"]
    est = Estado(datos["X"], datos["meta"])
    db = m2m.SimpleVectorDB(latent_dim=D)
    db.add(ids=[f"doc{i}" for i in range(N)], vectors=datos["X"], metadata=datos["meta"])
    eng = db.engine.m2m.splats.engine
    store = db.engine.m2m.splats
    assert eng._is_indexed and len(eng.coarse_cluster_indices) == C
    _check(db, est.X_cur, est.alive, est.meta_cur, Q, "inicial")
    radii0 = eng.coarse_cluster_radii

    # (ii) add() incremental SIN rebuild (attach: n <= 1.25 * last_n)
    n_new = 400
    assert N + n_new <= 1.25 * N
    ln = rng.integers(0, C, size=n_new)
    xn, metan = _nuevo_estado(rng, centros, ln, N, pasan25, pasan5)
    db.add(ids=[f"doc{N + j}" for j in range(n_new)], vectors=xn, metadata=metan)
    assert eng._is_indexed and store._last_trained_n == N + n_new, "se esperaba attach"
    assert eng.coarse_cluster_radii is radii0, "(ii) attach debe mantener el cache de radios"
    est.X_cur += [x for x in xn]
    est.meta_cur += list(metan)
    est.alive += [True] * n_new
    _check(db, est.X_cur, est.alive, est.meta_cur, Q, "add incremental (attach)")

    # (iii) update_vector: la fila se reescribe (misma o distinta cluster)
    for _ in range(80):
        i = int(rng.integers(0, len(est.alive)))
        v = _puntos(rng, centros, np.array([int(rng.integers(0, C))]))[0]
        db.update(f"doc{i}", vector=v)
        est.X_cur[i] = v
    assert (
        eng.coarse_cluster_radii is radii0
    ), "(iii) update_vector debe mantener el cache de radios"
    _check(db, est.X_cur, est.alive, est.meta_cur, Q, "update_vector")

    # delete (soft) + re-add de ids borrados, con vector NUEVO
    borrar = [int(i) for i in rng.choice(len(est.alive), size=150, replace=False)]
    db.delete(ids=[f"doc{i}" for i in borrar])
    for i in borrar:
        est.alive[i] = False
    _check(db, est.X_cur, est.alive, est.meta_cur, Q, "delete soft")
    readd = borrar[:60]
    lr = rng.integers(0, C, size=len(readd))
    xr, _ = _nuevo_estado(rng, centros, lr, 0, pasan25, pasan5)
    metar = []
    for j, i in enumerate(readd):
        m = dict(est.meta_cur[i])  # bucket/bucket20 no cambian (dependen del índice del doc)
        m["cl25"], m["cl5"] = int(lr[j] in pasan25), int(lr[j] in pasan5)
        metar.append(m)
        est.meta_cur[i] = m
        est.X_cur[i] = xr[j]
    db.add(ids=[f"doc{i}" for i in readd], vectors=xr, metadata=metar)
    for i in readd:
        est.alive[i] = True
    _check(db, est.X_cur, est.alive, est.meta_cur, Q, "delete + re-add")

    # hard delete: compact + reconstrucción
    duros = [
        int(i)
        for i in rng.choice([i for i, a in enumerate(est.alive) if a], size=40, replace=False)
    ]
    db.delete(ids=[f"doc{i}" for i in duros], hard=True)
    for i in duros:
        est.alive[i] = False
    _check(db, est.X_cur, est.alive, est.meta_cur, Q, "hard delete")

    # save() + instancia nueva (reopen)
    monkeypatch.setenv("M2M_HMAC_SECRET", "p17-secret")
    p = tmp_path / "store"
    db.save(str(p))
    db2 = m2m.SimpleVectorDB(latent_dim=D)
    db2.load(str(p))
    assert db2 is not None
    _check(db2, est.X_cur, est.alive, est.meta_cur, Q, "save + reopen")


def test_consolidate_mantiene_exactitud(datos):
    rng, centros, Q = datos["rng"], datos["centros"], datos["Q"]
    est = Estado(datos["X"], datos["meta"])
    db = m2m.AdvancedVectorDB(latent_dim=D)
    db.add(ids=[f"doc{i}" for i in range(N)], vectors=datos["X"], metadata=datos["meta"])
    store = db.engine.m2m.splats
    _check(db, est.X_cur, est.alive, est.meta_cur, Q, "consolidate: antes")
    # podar ~100 splats: alpha por debajo del umbral
    orden = list(db._splat_id_order)
    podar = {int(i) for i in rng.choice(len(orden), size=100, replace=False)}
    with db._rw_lock:
        for i in podar:
            store.alpha[i] = 0.01
    n_removed = db.consolidate(0.5)
    assert n_removed >= 1
    vivos = {str(x) for x in db._splat_id_order}
    for i, doc in enumerate(orden):
        if i in podar:
            est.alive[int(doc[len("doc") :])] = doc in vivos
    _check(db, est.X_cur, est.alive, est.meta_cur, Q, "consolidate")
