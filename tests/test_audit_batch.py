"""Tests de regresion del pack P2 (fix/p2): search_batch / search_batch_filtered.

Un test por criterio de aceptacion de P2 (BACKLOG). Items de auditoria:
r4-07 (isin aplana probe / expansion con union del lote), r1-12 (union de
candidatos), r1-22 (sentinel BIG), r1-07 (LSH en search_batch_filtered), r1-05
(soft-delete sin backfill, parte batch), r1-27 (k float / DB vacia, parte batch).

Convenciones: las posiciones que devuelven search_batch / search_batch_filtered
en modo HRM2 indexan db._splat_id_order (posicion de splat -> id).
"""

import numpy as np
import pytest

from m2m import SimpleVectorDB

K = 10


# --------------------------------------------------------------------------- helpers
def _unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _clustered(n, dim, n_clusters=50, seed=42, noise=0.35, nq=200):
    """Mismo generador que bench/bench_baseline.py (clustered in-distribution)."""
    rng = np.random.RandomState(seed)
    centers = rng.randn(n_clusters, dim).astype(np.float32)
    assign = rng.randint(0, n_clusters, size=n)
    data = centers[assign] + noise * rng.randn(n, dim).astype(np.float32)
    queries = centers[rng.randint(0, n_clusters, size=nq)] + noise * rng.randn(nq, dim).astype(
        np.float32
    )
    return _unit(data).astype(np.float32), _unit(queries).astype(np.float32)


def _uniform(n, dim, seed=42, nq=200):
    rng = np.random.RandomState(seed + 1)
    data = rng.randn(n, dim).astype(np.float32)
    queries = rng.randn(nq, dim).astype(np.float32)
    return _unit(data).astype(np.float32), _unit(queries).astype(np.float32)


def _db(data, **kw):
    db = SimpleVectorDB(
        latent_dim=data.shape[1], enable_lsh_fallback=False, max_splats=len(data), n_probe=5, **kw
    )
    db.add(ids=[str(i) for i in range(len(data))], vectors=data)
    return db


def _brute(data, q, k=K, allowed=None):
    """Top-k exacto por coseno (datos unitarios); `allowed` = mascara bool de posiciones."""
    sims = data @ q
    if allowed is not None:
        sims = np.where(allowed, sims, -np.inf)
    return np.argsort(-sims)[:k]


def _routing(db, q):
    """Replica el routing IVF de la query: (orden completo de clusters, set de probes)."""
    eng = db.engine.m2m.splats.engine
    centroids = np.asarray(eng.coarse_model.centroids_, dtype=np.float32)
    n_probe = int(eng.n_probe)
    cdist = ((centroids - q) ** 2).sum(axis=1)
    return np.argsort(cdist), set(np.argpartition(cdist, n_probe - 1)[:n_probe].tolist())


def _assert_hrm2_path(db, n_min=3001):
    store = db.engine.m2m.splats
    assert (
        store.n_active >= n_min and store.engine.coarse_model is not None
    ), "el test necesita el camino HRM2 (n > 3000): es el unico donde vive el bug de la union"


# --------------------------------------------------------------------------- criterio 1 + r4-07 / r1-12
@pytest.mark.parametrize("dataset", ["clustered", "uniform"])
def test_candidates_are_per_query_and_batch_equals_single_batch(dataset):
    """Cada query solo se puntua contra SUS clusters sondeados:
    search_batch(Q)[i] == search_batch(Q[i:i+1])[0] para todas las filas, y ningun
    resultado cae fuera de los candidatos propios (hoy: np.isin aplana probe y la
    union del lote entero se puntua para cada query)."""
    data, q = _clustered(5000, 64) if dataset == "clustered" else _uniform(5000, 64)
    db = _db(data)
    _assert_hrm2_path(db)
    n = db.engine.m2m.splats.n_active

    out = db.search_batch(q, k=K)
    single = np.vstack([db.search_batch(q[i : i + 1], k=K)[0] for i in range(len(q))])
    assert out.shape == (len(q), K)
    assert np.array_equal(out, single), (
        f"search_batch(Q)[i] != search_batch(Q[i:i+1])[0] en "
        f"{int((out != single).any(axis=1).sum())} filas: el resultado depende de las companeras de lote"
    )
    # ids via include_metadata == ids via posiciones
    rows = db.search_batch(q, k=K, include_metadata=True)
    for i in range(len(q)):
        assert [r.id for r in rows[i]] == [db._splat_id_order[int(j)] for j in out[i]]

    # filas puntuadas por query == filas de SUS probes (no la union del lote)
    outside = 0
    rows_per_query = []
    for i in range(len(q)):
        _, probes = _routing(db, q[i])
        eng = db.engine.m2m.splats.engine
        own = set().union(*(set(eng.coarse_cluster_indices[c].tolist()) for c in probes))
        rows_per_query.append(len(own))
        outside += sum(1 for j in out[i] if int(j) not in own)
    assert outside == 0, f"{outside} resultados fuera de los candidatos propios de su query"
    assert 0 < np.mean(rows_per_query) < 0.5 * n, "el routing deberia podar (IVF), no escanear todo"


# --------------------------------------------------------------------------- criterio 2
@pytest.mark.parametrize("dataset", ["clustered", "uniform"])
def test_recall_batch_equals_recall_single_search(dataset):
    """recall@10 de search_batch == recall@10 de search() individual (misma DB, mismas
    queries). El batch rankea por L2 puro; se fija rank_by='l2' en el store (mismo orden
    que find_neighbors) para comparar la misma metrica — el rerank gaussiano (default)
    es otra semantica de score, fuera del alcance de P2 (P7a)."""
    data, q = _clustered(5000, 64) if dataset == "clustered" else _uniform(5000, 64)
    db = _db(data)
    _assert_hrm2_path(db)
    db.engine.m2m.splats.rank_by = "l2"

    batch = db.search_batch(q, k=K)
    batch_sets = [set(db._splat_id_order[int(j)] for j in row) for row in batch]
    single_sets = [set(r.id for r in db.search(x, k=K, include_metadata=True)) for x in q]
    gt = [set(str(j) for j in _brute(data, x)) for x in q]
    rec_batch = np.mean([len(b & g) / K for b, g in zip(batch_sets, gt)])
    rec_single = np.mean([len(s & g) / K for s, g in zip(single_sets, gt)])
    assert (
        rec_batch == rec_single
    ), f"recall batch {rec_batch:.4f} != recall single {rec_single:.4f}"
    assert all(len(s) == K for s in single_sets)
    n_diff = sum(1 for b, s in zip(batch_sets, single_sets) if b != s)
    assert n_diff <= 2, f"top-10 batch vs single difieren en {n_diff}/200 queries"


# --------------------------------------------------------------------------- criterio 3 (r4-07)
def test_filtered_expansion_is_decided_per_query():
    """ACORN-lite por query: la expansion de probes se decide con la cobertura de CADA
    query y sigue SU orden de clusters (hoy: cobertura de la union del lote y orden de
    la query 0). Escenario: la query A solo tiene docs permitidos en el cluster X (su
    primer cluster NO sondeado); la query B (primera del lote) tiene docs permitidos en
    W (sondeado por B, ajeno a A). Correcto: A recibe X; con la union: W."""
    rng = np.random.RandomState(0)
    d, n_blobs, per = 16, 16, 250
    centers = rng.randn(n_blobs, d).astype(np.float32)
    data = centers[rng.randint(0, n_blobs, per * n_blobs)] + 0.05 * rng.randn(per * n_blobs, d)
    data = _unit(data).astype(np.float32)
    db = _db(data)
    _assert_hrm2_path(db)
    eng = db.engine.m2m.splats.engine
    assignments = np.asarray(eng.coarse_assignments)
    centroids = np.asarray(eng.coarse_model.centroids_, dtype=np.float32)
    n_probe = int(eng.n_probe)
    need = max(1, int(K * 1.5))

    # query A y su primer cluster no sondeado X
    qA = data[0]
    order_A, probes_A = _routing(db, qA)
    # X: su primer cluster NO sondeado con >= need docs (la expansion debe parar ahi)
    X = int(next(c for c in order_A[n_probe:] if int((assignments == c).sum()) >= need))
    assert X not in probes_A
    # query B: la mas lejana a X (para que X no entre en sus probes)
    far = int(np.argmax(((centroids - centroids[X]) ** 2).sum(axis=1)))
    qB = _unit(centroids[far : far + 1])[0]
    order_B, probes_B = _routing(db, qB)
    assert X not in probes_B
    # W: un cluster sondeado por B, no sondeado por A, con >= need docs permitidos
    W = next(
        c
        for c in order_B
        if c in probes_B and c not in probes_A and int((assignments == c).sum()) >= need
    )
    assert int((assignments == X).sum()) >= K
    allowed_pos = np.isin(assignments, [X, W])
    assert not allowed_pos[
        np.concatenate([eng.coarse_cluster_indices[c] for c in probes_A])
    ].any(), "precondicion: A no tiene ningun doc permitido en sus probes -> necesita expansion"

    for ia in (0, 1):  # posicion de A en el lote: el orden no debe importar
        batch = np.vstack([qA, qB]) if ia == 0 else np.vstack([qB, qA])
        out = db.search_batch_filtered(batch, k=K, filter_fn=lambda i: bool(allowed_pos[i]))
        row_A, row_B = out[ia], out[1 - ia]
        assert (row_A >= 0).all() and (row_B >= 0).all()
        assert allowed_pos[row_A].all() and allowed_pos[row_B].all()
        assert (
            assignments[row_A] == X
        ).all(), (
            "la query A recibio docs de un cluster ajeno (W) en vez de expandir SUS probes hasta X"
        )
        assert (assignments[row_B] == W).all()
        # A recibio los K mas cercanos de X
        assert set(row_A.tolist()) == set(_brute(data, qA, allowed=(assignments == X)).tolist())


# --------------------------------------------------------------------------- criterio 4 (r1-22)
def test_filtered_sentinel_survives_huge_norm_query():
    """Sentinel: con una query de norma 1e5 y un filtro que admite un solo doc, el
    resultado contiene solo ese doc (hoy BIG=1e9 < ||q||^2 y ganan los no permitidos)."""
    rng = np.random.RandomState(1)
    X = rng.randn(500, 16).astype(np.float32)
    db = _db(X)
    pos_zero = db._splat_id_order.index("0")
    for scale in (1e5, 1e6):
        q = X[1] * scale
        out = db.search_batch_filtered(q[None, :], k=3, filter_fn=lambda i: i == pos_zero)
        assert out.shape == (1, 3)
        assert out[0, 0] == pos_zero, f"norma {scale:.0e}: {out[0].tolist()}"
        assert (out[0, 1:] == -1).all()


def test_filtered_results_respect_filter_uniform():
    """Sanidad del camino filtrado con selectividad 25% sobre datos uniformes: todo
    resultado pasa el filtro y cada fila trae K resultados."""
    data, q = _uniform(5000, 64)
    db = _db(data)
    _assert_hrm2_path(db)
    n = db.engine.m2m.splats.n_active
    allowed = np.zeros(n, dtype=bool)
    allowed[::4] = True
    out = db.search_batch_filtered(q, k=K, filter_fn=lambda i: bool(allowed[i]))
    assert out.shape == (len(q), K)
    assert (out >= 0).all()
    assert allowed[out].all()


# --------------------------------------------------------------------------- criterio 5 (r1-07)
def test_lsh_mode_filtered_batch_returns_valid_positions():
    """Modo LSH: search_batch_filtered devolvia todo -1 (sin branch LSH). Debe devolver
    posiciones validas (indexan _lsh_id_map) con los mismos ids que search() single, y
    respetar el filtro. search_batch en modo LSH sigue mapeando a ids correctos."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((2000, 32)).astype(np.float32)
    db = SimpleVectorDB(latent_dim=32, enable_lsh_fallback=True)
    db.add(ids=[f"u{i}" for i in range(2000)], vectors=X)
    assert db._use_lsh and db.engine.m2m.splats.n_active == 0
    n_map = len(db._lsh_id_map)

    out = db.search_batch_filtered(X[:3], k=5, filter_fn=lambda i: True)
    assert out.shape == (3, 5)
    assert (out >= 0).all() and (out < n_map).all(), f"posiciones invalidas: {out}"
    for i in range(3):
        single = [r.id for r in db.search(X[i], k=5, include_metadata=True)]
        assert [db._lsh_id_map[int(j)] for j in out[i]] == single

    sb = db.search_batch(X[:3], k=5)
    assert (sb >= 0).all() and (sb < n_map).all()
    for i in range(3):
        single = [r.id for r in db.search(X[i], k=5, include_metadata=True)]
        assert [db._lsh_id_map[int(j)] for j in sb[i]] == single
        rows = db.search_batch(X[i : i + 1], k=5, include_metadata=True)[0]
        assert [r.id for r in rows] == single

    # filtro restrictivo: solo las primeras 100 posiciones -> todo resultado dentro del
    # filtro; -1 cuando el sondeo del LSH no junta k candidatos (no hay expansion en LSH)
    out = db.search_batch_filtered(X[:4], k=5, filter_fn=lambda i: i < 100)
    assert (out >= 0).any(axis=1).all() and (out < 100)[out >= 0].all()
    # filtro que no admite nada -> padding -1
    out = db.search_batch_filtered(X[:2], k=5, filter_fn=lambda i: False)
    assert (out == -1).all()


# --------------------------------------------------------------------------- criterio 6 (r1-05)
def test_batch_excludes_soft_deleted_and_backfills():
    """Tras un soft-delete del 50%, search_batch nunca devuelve posiciones de docs en
    _deleted (rellenando desde mas candidatos propios) mientras haya activos."""
    data, q = _clustered(5000, 64)
    db = _db(data)
    _assert_hrm2_path(db)
    n = db.engine.m2m.splats.n_active
    order = db._splat_id_order[:n]
    deleted_ids = [order[i] for i in range(0, n, 2)]
    assert db.delete(ids=deleted_ids).deleted == len(deleted_ids)

    out = db.search_batch(q, k=K)
    assert out.shape == (len(q), K)
    assert (out >= 0).all()
    bad = sum(1 for row in out for j in row if order[int(j)] in db._deleted)
    assert bad == 0, f"{bad} posiciones apuntan a docs soft-deleted"
    rows = db.search_batch(q, k=K, include_metadata=True)
    assert all(len(r) == K for r in rows)
    assert all(r_.id not in db._deleted for r in rows for r_ in r)

    # idem en search_batch_filtered
    out = db.search_batch_filtered(q, k=K, filter_fn=lambda i: i % 3 != 0)
    bad = sum(1 for row in out for j in row if order[int(j)] in db._deleted)
    assert bad == 0


# --------------------------------------------------------------------------- criterio 7 (r1-27)
def test_empty_db_and_k_validation_batch_apis():
    db = SimpleVectorDB(latent_dim=8)
    q2 = np.zeros((2, 8), dtype=np.float32)
    a = db.search_batch(q2, k=3)
    assert a.shape == (2, 0)
    b = db.search_batch_filtered(q2, k=3, filter_fn=lambda i: True)
    assert (
        b.shape == a.shape
    ), f"DB vacia: search_batch {a.shape} vs search_batch_filtered {b.shape}"

    rng = np.random.RandomState(3)
    db = _db(rng.randn(300, 8).astype(np.float32))
    q2 = rng.randn(2, 8).astype(np.float32)
    for call in (
        lambda: db.search_batch(q2, k=2.5),
        lambda: db.search_batch_filtered(q2, k=2.5, filter_fn=lambda i: True),
    ):
        with pytest.raises(TypeError, match="k must be an integer"):
            call()
    with pytest.raises(ValueError, match="k must be >= 1"):
        db.search_batch_filtered(q2, k=0, filter_fn=lambda i: True)
    assert db.search_batch(q2, k=np.int64(3)).shape == (2, 3)  # numpy int aceptado
