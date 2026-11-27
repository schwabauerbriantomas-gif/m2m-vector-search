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


@pytest.fixture(params=["mem", "storage"])
def mode(request, tmp_path):
    """Mismo test en memoria y con storage_path: la semantica no debe divergir."""
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


# --- END PART 1 (r1-14, r1-27) ---
