"""P13 (medicion del orquestador, probe_filter_latency.py): latencia de search() con filtro.

Antes: el camino con filtro armaba la mascara ``allowed`` con un bucle O(n) EN PYTHON
por llamada (``position_ok(i) and passes_filter(order[i])``, dos funciones Python por
posicion) y ``search_batch_filtered`` rehacia otro O(n) con ``np.fromiter(filter_fn)``.
20k docs: 36.8 ms/query (27 QPS) con selectividad 25% vs 0.45 ms sin filtro.

Ahora: cache de mascaras por filtro dict (clave = forma canonica + ``_meta_version`` +
``n_splats``; LRU <= 32) + evaluacion vectorizada por columnas de metadata para la
primera evaluacion, con paridad EXACTA contra ``match_metadata_filter`` (helper unico de
semantica de filtros). Los tests de cache son por CONTEO de evaluaciones, no por tiempo.
"""

import json
import threading

import numpy as np
import pytest

import m2m
from m2m import SimpleVectorDB, match_metadata_filter, validate_metadata_filter

K = 10
IN_OPS = ("$in", "$nin")
CONTAINERS = (list, tuple, set, frozenset)


def _make_db(n=500, d=16, seed=0):
    """500 docs con: 'cat' int 0..3, 'kind' constante, 'score' AUSENTE en 1/7 de los
    docs, 'tag' con tipos mezclados (str / int / ausente) y 'vec' (lista) en 1/5."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    meta = []
    for i in range(n):
        m = {"cat": int(i % 4), "kind": "base"}
        if i % 7:
            m["score"] = float(rng.standard_normal())
        if i % 3 == 0:
            m["tag"] = ("a", "b", "c")[(i // 3) % 3]
        elif i % 3 == 1:
            m["tag"] = int(i % 5)
        # i % 3 == 2: sin "tag"
        if i % 5 == 0:
            m["vec"] = [i % 5, 1.0]
        meta.append(m)
    meta[1]["score"] = 1.5  # doc con tag int Y score > 0 (para el filtro que levanta ValueError)
    db = SimpleVectorDB(latent_dim=d)
    db.add(ids=[f"d{i}" for i in range(n)], vectors=X, metadata=meta)
    return db, X


def _n_active_splats(db):
    return int(db.engine.m2m.splats.n_active)


def _python_mask(db, flt, n):
    """Referencia: el camino Python original de search() (position_ok AND filtro)."""
    order = db._splat_id_order
    out = np.zeros(n, dtype=bool)
    for i in range(min(n, len(order))):
        doc_id = order[i]
        if (
            doc_id in db._vectors
            and doc_id not in db._deleted
            and match_metadata_filter(db._metadata.get(doc_id, {}), flt)
        ):
            out[i] = True
    return out


def _outcome(fn):
    try:
        return ("ok", fn())
    except ValueError as e:
        return ("ValueError", str(e))


@pytest.fixture
def count_evals(monkeypatch):
    """Cuenta las evaluaciones de match_metadata_filter (helper unico de semantica)."""
    calls = {"n": 0}
    real = m2m.match_metadata_filter

    def counting(meta, flt):
        calls["n"] += 1
        return real(meta, flt)

    monkeypatch.setattr(m2m, "match_metadata_filter", counting)
    return calls


# ---------------------------------------------------------------------------
# 2. Paridad EXACTA entre el camino vectorizado y el Python (>= 8 filtros)
# ---------------------------------------------------------------------------
FILTERS = [
    {"cat": {"$eq": 1}},
    {"cat": {"$ne": 1}},
    {"cat": {"$in": [1, 3]}},
    {"cat": {"$nin": [0, 2]}},
    {"score": {"$gt": 0.0}},  # clave ausente en algunos docs -> no cumple
    {"score": {"$lte": 0.5}, "cat": {"$gte": 2}},  # AND implicito por clave
    {"tag": "a"},  # tipos mezclados (str e int) en la misma columna
    {"tag": {"$in": [0, "b"]}},
    {"tag": {"$nin": ["a"]}},
    {"cat": 2, "score": {"$ne": 0.0}},  # valor directo + dict de operador
    {"cat": {"$in": {1, 2}}},  # set
    {"cat": {"$lt": 2.5}},  # int vs float
]


@pytest.mark.parametrize(
    "flt", FILTERS, ids=[json.dumps(f, sort_keys=True, default=str) for f in FILTERS]
)
def test_vectorized_mask_matches_python_path(count_evals, flt):
    db, X = _make_db()
    n = _n_active_splats(db)
    ref = _python_mask(db, flt, n)
    assert ref.any() and not ref.all()
    count_evals["n"] = 0
    got = db._filter_position_mask(flt, db._splat_id_order, n)
    assert count_evals["n"] == 0, "filtro simple: debio evaluarse vectorizado (0 calls Python)"
    np.testing.assert_array_equal(got, ref)
    # End-to-end: search devuelve exactamente los docs de la mascara (k >= #matches)
    res = db.search(X[0], k=10**6, filter=flt, include_metadata=True)
    assert {r.id for r in res} == {db._splat_id_order[i] for i in np.flatnonzero(ref)}


def test_vectorized_and_python_paths_raise_together():
    db, _ = _make_db()
    n = _n_active_splats(db)
    # tag int vs str: TypeError -> _invalid_filter (ValueError) en ambos caminos
    flt = {"score": {"$gt": 0.0}, "tag": {"$lt": "b"}}
    ref = _outcome(lambda: _python_mask(db, flt, n))
    assert ref[0] == "ValueError"
    got = _outcome(lambda: db._filter_position_mask(flt, db._splat_id_order, n))
    assert got[0] == "ValueError"
    # Y a traves de la API publica (fail closed, sin cache de un resultado parcial)
    with pytest.raises(ValueError):
        db.search(db._vectors["d1"], k=K, filter=flt)


def test_vectorized_mask_random_filters_parity(count_evals):
    """Propiedad: 200 filtros aleatorios (ops, tipos mezclados, claves ausentes,
    $in/$nin, columnas no vectorizables) -> misma mascara o misma excepcion."""
    rng = np.random.default_rng(1234)
    db, _ = _make_db()
    n = _n_active_splats(db)
    keys = ["cat", "score", "tag", "vec", "absent"]
    ops = ["$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"]
    comps = [
        0,
        1,
        2,
        3,
        4,
        0.5,
        -1.0,
        2.5,
        "a",
        "b",
        "c",
        True,
        None,
        [0, 1],
        ["a", "c"],
        (2, 4),
        {1, 2, 3},
    ]
    n_vectorized = 0
    n_checked = 0
    for _ in range(200):
        flt = {}
        for key in rng.choice(keys, size=int(rng.integers(1, 3)), replace=False):
            op = str(rng.choice(ops))
            comp = comps[int(rng.integers(len(comps)))]
            if op in IN_OPS and not isinstance(comp, CONTAINERS):
                comp = [comp]
            direct = rng.random() < 0.15 and not isinstance(comp, CONTAINERS)
            flt[str(key)] = comp if direct else {op: comp}
        validate_metadata_filter(flt)
        ref = _outcome(lambda: _python_mask(db, flt, n))
        count_evals["n"] = 0
        got = _outcome(lambda: db._filter_position_mask(flt, db._splat_id_order, n))
        n_vectorized += count_evals["n"] == 0
        n_checked += 1
        assert got[0] == ref[0], (flt, ref, got)
        if ref[0] == "ok":
            np.testing.assert_array_equal(got[1], ref[1], err_msg=str(flt))
    assert n_checked == 200
    assert n_vectorized >= 80, f"solo {n_vectorized}/200 filtros fueron vectorizados"


# ---------------------------------------------------------------------------
# 1. Cache de mascaras: 2ª query con el mismo filtro -> 0 evaluaciones
# ---------------------------------------------------------------------------
def test_second_query_same_filter_zero_evaluations(count_evals):
    db, X = _make_db()
    # 'vec' guarda listas: NO vectorizable -> primera evaluacion por el camino Python
    flt = {"vec": {"$eq": [0, 1.0]}, "cat": {"$gte": 1}}
    r1 = db.search(X[0], k=K, filter=flt, include_metadata=True)
    first = count_evals["n"]
    assert first > 0, "la primera evaluacion no vectorizable debio pasar por Python"
    assert r1 and all(r.metadata["vec"] == [0, 1.0] for r in r1)
    r2 = db.search(X[0], k=K, filter=flt, include_metadata=True)
    assert count_evals["n"] == first, "2ª query con el mismo filtro: 0 evaluaciones"
    assert [r.id for r in r1] == [r.id for r in r2]
    # Filtro vectorizable: tampoco re-evalua, y la mascara es el mismo objeto (cache hit)
    flt2 = {"cat": {"$eq": 1}}
    db.search(X[1], k=K, filter=flt2, include_metadata=True)
    n = _n_active_splats(db)
    m1 = db._filter_position_mask(flt2, db._splat_id_order, n)
    assert db._filter_position_mask(flt2, db._splat_id_order, n) is m1
    assert count_evals["n"] == first


def test_callable_filter_is_never_cached():
    db, X = _make_db()
    calls = {"n": 0}

    def flt(meta):
        calls["n"] += 1
        return meta["cat"] == 1

    r1 = db.search(X[0], k=K, filter=flt, include_metadata=True)
    r2 = db.search(X[0], k=K, filter=flt, include_metadata=True)
    assert r1 and r2 and all(r.metadata["cat"] == 1 for r in r1 + r2)
    assert calls["n"] >= 2 * _n_active_splats(db), "un callable se evalua por query, no se cachea"


def test_filter_mask_cache_is_bounded_lru():
    db, X = _make_db()
    for s in range(40):  # 40 filtros distintos -> mas que el maximo del cache
        db.search(X[0], k=K, filter={"score": {"$gt": float(s) - 20.0}}, include_metadata=True)
    cache = db._filter_mask_cache
    assert len(cache) <= 32
    # La entrada mas reciente sigue y la mas vieja salio
    assert list(cache)[-1][0] == m2m._canonical_filter({"score": {"$gt": 19.0}})


def test_cache_invalidated_by_every_writer(count_evals):
    db, X = _make_db()
    flt = {"cat": {"$eq": 1}}
    db.search(X[0], k=K, filter=flt, include_metadata=True)
    v0 = db._meta_version

    db.update("d1", metadata={"cat": 9})  # d1 era cat=1: deja de cumplir
    assert db._meta_version > v0
    res = db.search(X[1], k=K, filter=flt, include_metadata=True)
    assert "d1" not in {r.id for r in res}

    db.add(ids=["new1"], vectors=X[1:2], metadata=[{"cat": 1}])  # nuevo doc que cumple
    assert db._meta_version > v0
    res = db.search(X[1], k=K, filter=flt, include_metadata=True)
    assert "new1" in {r.id for r in res}

    db.delete("d5")  # soft delete de un doc cat=1
    res = db.search(X[5], k=K, filter=flt, include_metadata=True)
    assert "d5" not in {r.id for r in res}

    db.delete("d2", hard=True)  # compacta y reordena posiciones
    res = db.search(X[5], k=K, filter=flt, include_metadata=True)
    assert res and all(r.metadata["cat"] == 1 for r in res)

    db.update("d3", metadata={"cat": 1})  # d3 era cat=3: pasa a cumplir
    res = db.search(X[3], k=K, filter=flt, include_metadata=True)
    assert "d3" in {r.id for r in res}
    assert db._meta_version > v0


def test_cache_rebuilt_after_restore_from_storage(tmp_path):
    d = 16
    rng = np.random.default_rng(3)
    n = 200
    X = rng.standard_normal((n, d)).astype(np.float32)
    path = str(tmp_path / "db")
    db = SimpleVectorDB(latent_dim=d, storage_path=path)
    db.add(
        ids=[f"d{i}" for i in range(n)], vectors=X, metadata=[{"cat": int(i % 4)} for i in range(n)]
    )
    flt = {"cat": {"$eq": 1}}
    assert len(db.search(X[0], k=1000, filter=flt, include_metadata=True)) == 50
    db.save()
    db2 = SimpleVectorDB(latent_dim=d, storage_path=path)
    res = db2.search(X[0], k=1000, filter=flt, include_metadata=True)
    assert len(res) == 50 and all(r.metadata["cat"] == 1 for r in res)
    db2.update("d1", metadata={"cat": 7})
    res = db2.search(X[0], k=1000, filter=flt, include_metadata=True)
    assert len(res) == 49 and "d1" not in {r.id for r in res}


# ---------------------------------------------------------------------------
# 3. search_batch_filtered(Q, k, filter={...})
# ---------------------------------------------------------------------------
def test_search_batch_filtered_accepts_dict_filter(count_evals):
    db, X = _make_db()
    Q = X[:8]
    flt = {"cat": {"$in": [1, 3]}}
    got = db.search_batch_filtered(Q, k=K, filter=flt)
    assert got.shape == (8, K)
    order = db._splat_id_order
    ref = db.search_batch_filtered(
        Q,
        k=K,
        filter_fn=lambda i: order[i] in db._vectors and db._metadata[order[i]]["cat"] in (1, 3),
    )
    np.testing.assert_array_equal(got, ref)
    for row in got:
        for pos in row:
            assert pos >= 0 and db._metadata[order[pos]]["cat"] in (1, 3)
    # 2ª llamada con el mismo filtro: 0 evaluaciones (usa el cache)
    n0 = count_evals["n"]
    got2 = db.search_batch_filtered(Q, k=K, filter=flt)
    assert count_evals["n"] == n0
    np.testing.assert_array_equal(got2, got)
    # Filtro invalido -> ValueError ANTES de tocar el indice (fail closed)
    with pytest.raises(ValueError):
        db.search_batch_filtered(Q, k=K, filter={"cat": {"$bogus": 1}})
    with pytest.raises(ValueError):
        db.search_batch_filtered(Q, k=K, filter={"cat": {"$eq": 1}}, filter_fn=lambda i: True)


def test_lsh_mode_dict_filter_matches_filter_fn():
    rng = np.random.default_rng(7)
    n, d = 400, 16
    X = rng.standard_normal((n, d)).astype(np.float32)  # uniforme -> silueta baja -> LSH
    db = SimpleVectorDB(latent_dim=d, enable_lsh_fallback=True)
    db.add(
        ids=[f"d{i}" for i in range(n)], vectors=X, metadata=[{"cat": int(i % 4)} for i in range(n)]
    )
    assert db._use_lsh
    flt = {"cat": {"$eq": 1}}
    res = db.search(X[0], k=20, filter=flt, include_metadata=True)
    assert len(res) == 20 and all(r.metadata["cat"] == 1 for r in res)
    Q = X[:6]
    a = db.search_batch_filtered(Q, k=20, filter=flt)
    b = db.search_batch_filtered(
        Q, k=20, filter_fn=lambda i: db._metadata[db._lsh_id_map[i]]["cat"] == 1
    )
    np.testing.assert_array_equal(a, b)
    db.update("d1", metadata={"cat": 0})
    res = db.search(X[1], k=1000, filter=flt, include_metadata=True)
    assert "d1" not in {r.id for r in res} and len(res) == 99


# ---------------------------------------------------------------------------
# 5. Concurrencia: 2 lectores con filtro + 1 writer (add/update/delete con metadata)
# ---------------------------------------------------------------------------
def test_filtered_search_under_concurrent_writes():
    n = 300
    db, X = _make_db(n=n)
    flt = {"kind": {"$eq": "base"}}
    errors = []
    stop = threading.Event()
    n_searches = [0, 0]

    def reader(r):
        q = X[r]
        try:
            while not stop.is_set():
                res = db.search(q, k=K, filter=flt, include_metadata=True)
                ids = [r_.id for r_ in res]
                assert len(ids) == len(set(ids)) <= K
                # Violacion medida sobre la metadata DEL resultado (snapshot al momento
                # de la busqueda): un cache stale dejaria pasar 'toggle' con kind=other.
                for r_ in res:
                    assert (
                        r_.metadata.get("kind") == "base"
                    ), f"filtro violado: {r_.id} {r_.metadata}"
                n_searches[r] += 1
        except Exception as e:  # pragma: no cover - se reporta en el assert de abajo
            errors.append(e)

    def writer():
        try:
            for t in range(40):
                db.update("dtoggle", metadata={"kind": "base" if t % 2 else "other", "cat": 1})
                db.add(ids=[f"new{t}"], vectors=X[t : t + 1], metadata=[{"kind": "new", "cat": 1}])
                if t % 3 == 0:
                    db.delete(f"d{100 + t}")  # soft
                if t % 5 == 0:
                    db.delete(f"d{200 + t}", hard=True)  # compacta: reordena posiciones
        except Exception as e:  # pragma: no cover
            errors.append(e)
        finally:
            stop.set()

    db.add(ids=["dtoggle"], vectors=X[0:1], metadata=[{"kind": "base", "cat": 1}])
    threads = [threading.Thread(target=reader, args=(r,)) for r in range(2)]
    threads.append(threading.Thread(target=writer))
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=120)
        assert not th.is_alive(), "deadlock: un hilo no termino"
    assert not errors, errors[0]
    assert min(n_searches) >= 5, n_searches
    # Estado final consistente: sin borrados ni docs que violen el filtro
    res = db.search(X[1], k=1000, filter=flt, include_metadata=True)
    ids = {r_.id for r_ in res}
    assert all(r_.metadata["kind"] == "base" for r_ in res)
    assert not any(i.startswith("new") for i in ids)
    assert all(f"d{100 + t}" not in ids for t in range(40) if t % 3 == 0)
    assert all(f"d{200 + t}" not in ids for t in range(40) if t % 5 == 0)
    # 'dtoggle' esta en el resultado sii su metadata actual cumple (writer termino en t=39 -> base)
    assert "dtoggle" in ids
