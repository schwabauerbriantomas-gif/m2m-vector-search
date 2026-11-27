"""Auditoria r1 - pack P3: ``search()`` y semantica de filtros de metadata.

Un test por criterio de aceptacion del BACKLOG (pack P3). Escritos ROJOS contra la
base del pack antes de cada fix (log: ``reports/p3_red.log``).

Ids de auditoria: r1-14 (filtros), r1-27 (validacion de k), r1-05 (backfill),
r1-06 (retorno legacy sin posiciones hard-deleted), r1-10 (energia + filtro),
r1-18 (aridad legacy), r1-16 (sin lista O(n) por query).
"""

import numpy as np
import pytest

from m2m import SimpleVectorDB

D = 8


@pytest.fixture(params=["mem"])
def mode(request, tmp_path):
    """Los casos corren en memoria; la MISMA tabla se verifica con storage_path en
    test_r1_14_storage_mode_mirrors_memory_semantics (un init de storage cuesta ~3 s)."""
    if request.param == "storage":
        return str(tmp_path / "store")
    return None


def _meta_db(storage_path):
    db = SimpleVectorDB(latent_dim=D, storage_path=storage_path)
    db.add(
        ids=["a", "b"],
        vectors=np.ones((2, D), dtype=np.float32),
        metadata=[{"c": 0, "t": "x"}, {"c": 3, "t": "y"}],
    )
    return db


def _ids(res):
    return [r.id for r in res]


# ---------------------------------------------------------------------------
# r1-14: operadores soportados, fail-closed y callable (criterios 1 y 2)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "flt,expected",
    [
        ({"c": {"$in": [3]}}, True),
        ({"c": {"$in": [0, 3]}}, True),
        ({"c": {"$in": (3,)}}, True),
        ({"c": {"$in": [0]}}, False),
        ({"c": {"$in": []}}, False),
        ({"c": {"$nin": [0]}}, True),
        ({"c": {"$nin": []}}, True),
        ({"c": {"$nin": [3]}}, False),
        ({"c": {"$eq": 3, "$in": [3]}}, True),
        ({"c": {"$eq": 3, "$in": [0]}}, False),
        ({"c": {"$gte": 3}, "t": {"$in": ["y"]}}, True),
        ({"c": {"$lte": 3}}, True),
        ({"c": 3}, True),
        ({"zz": {"$in": [1]}}, False),  # campo ausente -> no matchea (semantica previa)
        ({"zz": {"$nin": [1]}}, False),
    ],
)
def test_r1_14_match_filter_operators(mode, flt, expected):
    db = _meta_db(mode)
    assert db._match_filter({"c": 3, "t": "y"}, flt) is expected


@pytest.mark.parametrize(
    "flt",
    [
        {"c": {"$gtt": 3}},  # typo de operador
        {"c": {"$exists": True}},  # operador no soportado
        {"c": {}},  # condicion vacia
        {"c": [3]},  # tipo invalido
        {"c": {"$in": 3}},  # $in sin coleccion
        {},  # filtro vacio
        "c",  # ni dict ni callable
    ],
)
def test_r1_14_match_filter_fail_closed(mode, flt):
    db = _meta_db(mode)
    with pytest.raises(ValueError):
        db._match_filter({"c": 3}, flt)


def test_r1_14_match_filter_incomparable_types(mode):
    db = _meta_db(mode)
    with pytest.raises(ValueError):
        db._match_filter({"c": 3}, {"c": {"$gt": "x"}})


def test_r1_14_match_filter_callable(mode):
    db = _meta_db(mode)
    assert db._match_filter({"c": 3}, lambda m: m["c"] == 3) is True
    assert db._match_filter({"c": 3}, lambda m: m["c"] == 0) is False


def test_r1_14_search_in_returns_only_matching(mode):
    db = _meta_db(mode)
    q = np.ones(D, dtype=np.float32)
    assert _ids(db.search(q, k=2, filter={"c": {"$in": [3]}}, include_metadata=True)) == ["b"]
    assert _ids(db.search(q, k=2, filter={"c": {"$in": [0]}}, include_metadata=True)) == ["a"]
    assert _ids(db.search(q, k=2, filter={"c": {"$nin": [3]}}, include_metadata=True)) == ["a"]
    assert db.search(q, k=2, filter={"c": {"$in": [99]}}, include_metadata=True) == []
    # include_metadata=False sigue devolviendo DocResult filtrados
    assert _ids(db.search(q, k=2, filter={"c": {"$in": [3]}})) == ["b"]


def test_r1_14_search_unknown_op_raises_even_on_empty_db(mode):
    db = SimpleVectorDB(latent_dim=D, storage_path=mode)
    q = np.ones(D, dtype=np.float32)
    with pytest.raises(ValueError):
        db.search(q, k=3, filter={"c": {"$gtt": 1}}, include_metadata=True)


def test_r1_14_search_filter_callable(mode):
    db = _meta_db(mode)
    q = np.ones(D, dtype=np.float32)
    assert _ids(db.search(q, k=2, filter=lambda m: m["c"] == 3, include_metadata=True)) == ["b"]
    assert sorted(_ids(db.search(q, k=2, filter=lambda m: m["t"] in ("x", "y")))) == ["a", "b"]
    assert db.search(q, k=2, filter=lambda m: False, include_metadata=True) == []


def test_r1_14_delete_filter_in_only_deletes_matching(mode):
    db = _meta_db(mode)
    res = db.delete(filter={"c": {"$in": [0]}})
    assert res.deleted == 1
    assert db._deleted == {"a"}
    # Un typo de operador no puede borrar "de mas": levanta antes de tocar nada.
    with pytest.raises(ValueError):
        db.delete(filter={"c": {"$gtt": 3}})
    assert db._deleted == {"a"}


# ---------------------------------------------------------------------------
# r1-27: validacion de k en search() (criterio 7)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [{}, {"include_metadata": True}, {"filter": {"c": {"$in": [3]}}, "include_metadata": True}],
)
def test_r1_27_k_float_raises_clear_typeerror(kwargs):
    db = _meta_db(None)
    q = np.ones(D, dtype=np.float32)
    with pytest.raises(TypeError, match="k must be an integer"):
        db.search(q, k=2.5, **kwargs)


def test_r1_27_k_numpy_int_accepted_and_zero_rejected():
    db = _meta_db(None)
    q = np.ones(D, dtype=np.float32)
    assert len(db.search(q, k=np.int64(1), include_metadata=True)) == 1
    assert len(db.search(q, k=np.int32(2), include_metadata=True)) == 2
    assert db.search(q, k=np.int64(2))[0].shape[1] == 2
    with pytest.raises(ValueError, match="k must be >= 1"):
        db.search(q, k=0)


# ---------------------------------------------------------------------------
# Helpers (r1-05, r1-06, r1-10, r1-16, r1-18)
# ---------------------------------------------------------------------------
def _random_db(n, d=16, seed=0, metadata=None, **kw):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    db = SimpleVectorDB(latent_dim=d, **kw)
    db.add(ids=[f"d{i}" for i in range(n)], vectors=X, metadata=metadata)
    return db, X


def _legacy_positions(*args):
    """Posiciones de _splat_id_order que trae el retorno legacy (4to elemento)."""
    return [int(i) for i in np.asarray(args[-1][3]).reshape(-1)]


def _all_active_positions(db, legacy):
    order = db._splat_id_order
    return all(
        order[i] in db._vectors and order[i] not in db._deleted
        for i in _legacy_positions(db, legacy)
    )


def _rows_per_component(legacy):
    """Filas de resultado en cada componente: eje 1 si mu conserva el batch dim [1, m, D]."""
    axis = 1 if np.asarray(legacy[0]).ndim == 3 else 0
    return tuple(np.asarray(a).shape[axis] if np.asarray(a).ndim > axis else 0 for a in legacy)


def _satisfies(db, meta, flt):
    return bool(flt(meta)) if callable(flt) else db._match_filter(meta, flt)


# ---------------------------------------------------------------------------
# r1-05: backfill tras soft-delete (criterio 4)
# ---------------------------------------------------------------------------
def test_r1_05_backfill_after_soft_delete_50pct():
    db, X = _random_db(60, seed=1)
    victims = [f"d{i}" for i in range(0, 60, 2)]
    assert db.delete(ids=victims).deleted == 30
    rng = np.random.default_rng(7)
    for _ in range(20):
        q = X[rng.integers(60)]
        res = db.search(q, k=10, include_metadata=True)
        ids_ = _ids(res)
        assert len(res) == 10
        assert len(set(ids_)) == 10
        assert not (set(ids_) & set(victims))
        legacy = db.search(q, k=10)
        pos = _legacy_positions(db, legacy)
        assert len(pos) == 10 and len(set(pos)) == 10
        assert _all_active_positions(db, legacy)
        assert _rows_per_component(legacy) == (10, 10, 10, 10)


def test_r1_05_backfill_returns_every_active_when_fewer_than_k():
    db, X = _random_db(12, seed=2)
    db.delete(ids=[f"d{i}" for i in range(5)])  # quedan 7 activos
    assert len(db.search(X[6], k=10, include_metadata=True)) == 7
    assert len(_legacy_positions(db, db.search(X[6], k=10))) == 7


def test_r1_05_backfill_with_filter_after_soft_delete():
    n = 80
    meta = [{"c": i % 4} for i in range(n)]
    db, X = _random_db(n, seed=10, metadata=meta)
    # i%4==1 equivale a i%8 in (1,5): borrando i%8 in (1,2) se va la MITAD de los c==1 y
    # la mitad de los c==2 -> quedan 10 activos de cada uno (c==0 y c==3 conservan 20).
    victims = [f"d{i}" for i in range(n) if i % 8 in (1, 2)]
    assert db.delete(ids=victims).deleted == 20
    for flt in ({"c": 1}, {"c": {"$in": [1, 2]}}, lambda m: m["c"] == 3):
        for q in X[:10]:
            res = db.search(q, k=10, filter=flt, include_metadata=True)
            ids_ = _ids(res)
            assert len(res) == 10 and len(set(ids_)) == 10
            assert all(_satisfies(db, r.metadata, flt) for r in res)
            assert not (set(ids_) & set(victims))
    # Menos matching que k -> todos los matching, sin relleno ni violaciones
    res = db.search(X[0], k=50, filter={"c": 1}, include_metadata=True)
    expected = sorted(f"d{i}" for i in range(n) if i % 8 == 5)
    assert sorted(_ids(res)) == expected and len(expected) == 10


# ---------------------------------------------------------------------------
# r1-06: el retorno legacy jamas incluye posiciones de docs hard-deleted (criterio 5)
# ---------------------------------------------------------------------------
def test_r1_06_legacy_never_returns_hard_deleted_positions():
    db, X = _random_db(20, seed=3)
    victim = "d7"
    old_vec = db._vectors[victim].copy()
    assert db.delete(id=victim, hard=True).deleted == 1
    assert victim not in db._vectors and victim not in db._deleted
    # Desde el pack P6 el hard-delete compacta el indice: ya no queda huerfano.
    assert victim not in db._splat_id_order
    # Defensa del lector (en profundidad): se inyecta a mano un huerfano — una posicion
    # de splat con el vector viejo que apunta al id borrado y que no esta en _vectors —
    # como lo dejaria un indice cargado de un storage viejo o un escritor externo.
    assert db.engine.add_splats(old_vec[np.newaxis, :]) == 1
    orphan_pos = len(db._splat_id_order)
    db._splat_id_order.append(victim)
    assert orphan_pos < db.engine.m2m.splats.n_active
    for q in (old_vec, X[0], np.ones(16, dtype=np.float32)):
        legacy = db.search(q, k=20)
        pos = _legacy_positions(db, legacy)
        assert len(pos) == 19
        assert _all_active_positions(db, legacy)
        assert orphan_pos not in pos
        assert victim not in {db._splat_id_order[i] for i in pos}
        # k=10: el backfill cubre el hueco del huerfano
        assert len(_legacy_positions(db, db.search(q, k=10))) == 10
        res = db.search(q, k=10, include_metadata=True)
        assert len(res) == 10 and victim not in _ids(res)


def test_r1_06_no_duplicate_ids_after_hard_delete_and_readd():
    db, X = _random_db(20, seed=4)
    db.delete(id="d7", hard=True)
    db.add(ids=["d7"], vectors=X[7:8])
    # Tras el pack P6 el hard-delete compacta: el re-add deja UNA sola posicion para el id.
    assert db._splat_id_order.count("d7") == 1
    # Defensa del lector: si un indice externo tuviera DOS posiciones vivas para el mismo
    # id (huerfano + re-add), el resultado no lo duplica y sigue devolviendo k docs.
    assert db.engine.add_splats(X[7:8]) == 1
    db._splat_id_order.append("d7")
    assert db._splat_id_order.count("d7") == 2
    for q in (X[7], X[0]):
        res = db.search(q, k=20, include_metadata=True)
        ids_ = _ids(res)
        assert len(ids_) == 20 and len(set(ids_)) == 20


# ---------------------------------------------------------------------------
# r1-10: include_energy + filter (criterio 3) y coherencia ranking/score del filtro
# ---------------------------------------------------------------------------
def test_r1_10_include_energy_plus_filter_same_ids():
    n = 400
    rng = np.random.default_rng(5)
    X = rng.standard_normal((n, 16)).astype(np.float32)
    meta = [{"r": float(v)} for v in rng.random(n)]
    db = SimpleVectorDB(latent_dim=16, enable_ebm=True)
    db.add(ids=[f"d{i}" for i in range(n)], vectors=X, metadata=meta)
    assert db.ebm_enabled and db._ebm_energy is not None
    flt = {"r": {"$lt": 0.05}}
    assert sum(1 for m in meta if m["r"] < 0.05) >= 10
    for q in X[:10]:
        with_e = db.search(q, k=10, filter=flt, include_energy=True, include_metadata=True)
        without_e = db.search(q, k=10, filter=flt, include_energy=False, include_metadata=True)
        assert len(without_e) == 10
        assert _ids(with_e) == _ids(without_e)
        assert all(r.metadata["r"] < 0.05 for r in with_e)
        assert all(r.energy is not None and r.confidence is not None for r in with_e)
        assert all(r.energy is None for r in without_e)


def test_filtered_path_ranks_and_scores_with_the_same_query():
    db, X = _random_db(40, seed=11, metadata=[{"c": i % 3} for i in range(40)])
    q = X[3] * 7.0  # norma != 1: rankear con la query cruda y puntuar con la normalizada divergia
    qn = q / (np.linalg.norm(q) + 1e-8)
    res = db.search(q, k=10, filter={"c": {"$gte": 0}}, include_metadata=True)
    assert len(res) == 10
    # P7a [audit r1-13] (decisión del orquestador): `score` pasó de -dist a similitud
    # coseno; la distancia L2 a la query normalizada ahora es el campo `distance`.
    for r in res:
        assert abs(r.distance - float(np.linalg.norm(r.vector - qn))) < 1e-5
        assert abs(r.score - float(np.dot(qn, r.vector))) < 1e-5
    d = [r.distance for r in res]
    assert d == sorted(d)  # el orden del ranking es el de la MISMA distancia que se reporta


# ---------------------------------------------------------------------------
# r1-18: retorno legacy SIEMPRE 4-tupla (mu, alpha, kappa, idx) (criterio 5)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", ["brute_force", "empty"])
def test_r1_18_legacy_is_always_4_tuple(case):
    d = 16
    if case == "empty":
        db = SimpleVectorDB(latent_dim=d)
    else:
        db, _ = _random_db(30, d=d, seed=6)
    res = db.search(np.ones(d, dtype=np.float32), k=3)
    assert isinstance(res, tuple) and len(res) == 4
    counts = _rows_per_component(res)
    assert len(set(counts)) == 1 and counts[0] == (0 if case == "empty" else 3)
    if case != "empty":
        assert np.asarray(res[0]).shape[-1] == d
        assert np.issubdtype(np.asarray(res[3]).dtype, np.integer)


def test_r1_18_docstring_documents_4_tuple():
    doc = SimpleVectorDB.search.__doc__
    assert "(mu, alpha, kappa, idx)" in doc
    assert "vectors, alphas, kappas" not in doc


def _lsh_db():
    d, n = 640, 2000
    rng = np.random.default_rng(42)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    db = SimpleVectorDB(device="cpu", latent_dim=d, enable_lsh_fallback=True)
    db.add(ids=[f"d{i}" for i in range(n)], vectors=X, metadata=[{"c": i % 4} for i in range(n)])
    assert db._use_lsh and db.lsh is not None
    return db, X


@pytest.fixture(scope="module")
def lsh_db():
    return _lsh_db()


def test_r1_18_legacy_4_tuple_and_filter_in_lsh_mode(lsh_db):
    db, X = lsh_db
    q = X[1]
    res = db.search(q, k=10)
    assert isinstance(res, tuple) and len(res) == 4
    assert _rows_per_component(res) == (10, 10, 10, 10)
    assert np.asarray(res[0]).shape[-1] == 640
    # Posicion LSH == posicion en _splat_id_order (ambos se extienden igual en add())
    assert all(0 <= i < len(db._splat_id_order) for i in _legacy_positions(res))
    # Filtro (dict y callable) en modo LSH, con y sin include_metadata
    for flt in ({"c": 1}, lambda m: m["c"] == 2):
        for im in (True, False):
            out = db.search(q, k=10, filter=flt, include_metadata=im)
            assert len(out) == 10
            assert all(_satisfies(db, db._metadata[r.id], flt) for r in out)


def test_r1_05_lsh_backfill_and_no_soft_deleted(lsh_db):
    db, X = lsh_db
    victims = [f"d{i}" for i in range(0, 2000, 2)]
    assert db.delete(ids=victims).deleted == 1000
    for q in X[[1, 3, 5]]:
        legacy = db.search(q, k=10)
        assert len(_legacy_positions(legacy)) == 10
        assert _all_active_positions(db, legacy)
        res = db.search(q, k=10, include_metadata=True)
        assert len(res) == 10 and not (set(_ids(res)) & set(victims))


# ---------------------------------------------------------------------------
# r1-16: sin lista O(n) de activos por query (criterio 6)
# ---------------------------------------------------------------------------
class _NoIterDict(dict):
    """dict que explota si alguien itera todas las claves (lista O(n) por query)."""

    def __iter__(self):
        raise AssertionError("search() iteró _vectors: lista O(n) de activos por query [r1-16]")


def test_r1_16_search_does_not_build_active_list():
    db, X = _random_db(50, seed=8, metadata=[{"c": i % 4} for i in range(50)])
    db._vectors = _NoIterDict(db._vectors)
    for q in X[:5]:
        assert len(db.search(q, k=5, include_metadata=True)) == 5
        assert len(_legacy_positions(db, db.search(q, k=5))) == 5
    db.delete(ids=["d1", "d2", "d3"])  # con borrados (backfill activo) tampoco itera
    assert len(db.search(X[0], k=5, include_metadata=True)) == 5
    assert len(_legacy_positions(db, db.search(X[0], k=5))) == 5
    assert len(db.search(X[0], k=5, filter={"c": 1}, include_metadata=True)) == 5


# ---------------------------------------------------------------------------
# Concurrencia (nota del pack P6 para search): el cuerpo corre bajo _rw_lock
# ---------------------------------------------------------------------------
def test_search_consistent_under_concurrent_hard_delete():
    """Un delete(hard=True) concurrente no deja ids/posiciones inexistentes ni excepciones.

    En esta base el hard-delete todavia no compacta el indice (lado escritor: otro
    pack), asi que aca funciona como guard de regresion para el merge con P6.
    """
    import threading

    db, X = _random_db(200, seed=12)
    q = X[0]
    errors = []

    def writer():
        try:
            for i in range(1, 200):
                db.delete(id=f"d{i}", hard=True)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    def reader():
        try:
            while len(db._vectors) > 20:
                with db._rw_lock:
                    before = set(db._vectors)  # _vectors solo se achica: todo id valido esta aca
                for r in db.search(q, k=5, include_metadata=True):
                    if r.id not in before:
                        errors.append(AssertionError(f"id inexistente devuelto: {r.id}"))
                        return
                legacy = db.search(q, k=5)
                for pos in _legacy_positions(legacy):
                    if pos >= len(before) + 200 or db._splat_id_order[pos] not in before:
                        errors.append(AssertionError("posicion de doc inexistente en legacy"))
                        return
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, daemon=True)] + [
        threading.Thread(target=reader, daemon=True) for _ in range(2)
    ]
    for t_ in threads:
        t_.start()
    for t_ in threads:
        t_.join(timeout=60)
    assert not any(
        t_.is_alive() for t_ in threads
    ), "hang/deadlock en search bajo delete concurrente"
    assert not errors, errors[:3]


# --- r1-14: la MISMA tabla de semantica con storage_path (M2MPersistence._matches_filter) ---
def test_r1_14_storage_mode_mirrors_memory_semantics(tmp_path):
    """Con storage_path, _match_filter delega a M2MPersistence._matches_filter: no debe divergir.
    Un solo init de storage (cuesta ~3 s), por eso los casos van todos aca en vez de parametrizados.
    """
    mem = _meta_db(None)
    sto = _meta_db(str(tmp_path / "store"))
    assert sto.storage is not None and mem.storage is None
    cases = [
        ({"c": {"$in": [3]}}, True),
        ({"c": {"$in": [0]}}, False),
        ({"c": {"$in": []}}, False),
        ({"c": {"$nin": [3]}}, False),
        ({"c": {"$nin": [0]}}, True),
        ({"c": {"$nin": []}}, True),
        ({"c": {"$eq": 3, "$in": [0]}}, False),
        ({"c": {"$gte": 3}, "t": {"$in": ["y"]}}, True),
        ({"zz": {"$in": [1]}}, False),
        ({"c": 3}, True),
    ]
    for flt, expected in cases:
        assert mem._match_filter({"c": 3, "t": "y"}, flt) is expected
        assert sto._match_filter({"c": 3, "t": "y"}, flt) is expected
    for flt in (
        {"c": {"$gtt": 3}},
        {"c": {}},
        {"c": [3]},
        {"c": {"$in": 3}},
        {},
        "c",
        {"c": {"$gt": "x"}},
    ):
        for db in (mem, sto):
            with pytest.raises(ValueError):
                db._match_filter({"c": 3}, flt)
    q = np.ones(D, dtype=np.float32)
    for db in (mem, sto):
        assert db._match_filter({"c": 3}, lambda m: m["c"] == 3) is True
        assert _ids(db.search(q, k=2, filter={"c": {"$in": [3]}}, include_metadata=True)) == ["b"]
        assert db.delete(filter={"c": {"$in": [0]}}).deleted == 1
        assert db._deleted == {"a"}
        with pytest.raises(ValueError):
            db.delete(filter={"c": {"$gtt": 3}})
        assert db._deleted == {"a"}
