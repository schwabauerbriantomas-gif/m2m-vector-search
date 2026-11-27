"""
Ciclo de vida de escritura y concurrencia — pack P6 (audit r1-03, r1-23, r1-04,
r1-06, r1-21, r1-08 y r3b-02).

Un test por criterio de aceptación de la sección P6 del BACKLOG. El test de
concurrencia corre con timeout (hilos daemon + join acotado) para no colgar la
suite si un writer se cuelga con el lock tomado.
"""

import threading

import numpy as np
import pytest

from m2m import SimpleVectorDB, normalize_sphere

D = 16


def _vecs(n, seed=0, dim=D):
    rng = np.random.RandomState(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def _db(n=0, seed=0, dim=D, **kw):
    db = SimpleVectorDB(latent_dim=dim, **kw)
    if n:
        db.add(ids=[f"d{i}" for i in range(n)], vectors=_vecs(n, seed, dim))
    return db


def _ids(results):
    return [r.id for r in results]


def _snapshot(db):
    store = db.engine.m2m.splats
    return (
        store.n_active,
        len(db._splat_id_order),
        store.mu[: store.n_active].copy(),
        store.alpha[: store.n_active].copy(),
        dict(db._vectors),
        dict(db._metadata),
        dict(db._documents),
        set(db._deleted),
    )


def _assert_same(db, snap):
    n_active, n_order, mu, alpha, vecs, meta, docs, deleted = snap
    store = db.engine.m2m.splats
    assert store.n_active == n_active
    assert len(db._splat_id_order) == n_order
    np.testing.assert_array_equal(store.mu[:n_active], mu)
    np.testing.assert_array_equal(store.alpha[:n_active], alpha)
    assert db._vectors == vecs
    assert db._metadata == meta
    assert db._documents == docs
    assert db._deleted == deleted


def _rows_close(rows, vec):
    return [bool(np.allclose(r, vec)) for r in rows]


def _legacy_rows(db, q, k):
    """Filas de mu del retorno legacy de search() (sin metadata). El path splats
    devuelve (1, k, D) + 4-tupla y el LSH (k, D) + 3-tupla: se normaliza acá
    porque la forma del retorno legacy es del pack P3."""
    out = db.search(q, k)
    mu = np.asarray(out[0])
    return mu.reshape(-1, mu.shape[-1])


# ---------------------------------------------------------------------------
# r1-03: concurrencia de add() — una sola sección crítica para toda la mutación
# ---------------------------------------------------------------------------


def test_add_mutates_index_under_the_write_lock(monkeypatch):
    """engine.add_splats (add_splat + build_index) debe correr con el lock tomado."""
    db = _db(n=3)
    seen = []
    orig = db.engine.add_splats

    def spy(vectors, labels=None):
        seen.append(bool(db._rw_lock._is_owned()))
        return orig(vectors, labels)

    monkeypatch.setattr(db.engine, "add_splats", spy)
    assert db.add(ids=["x"], vectors=_vecs(1, 1)) == 1
    assert db.add(ids=["y", "z"], vectors=_vecs(2, 2)) == 2
    assert seen == [True, True]


def test_concurrent_adds_and_reads_are_consistent():
    """2 writers x 300 add() + 4 readers: 0 excepciones, estado final coherente."""
    db = _db(n=200)
    n_w, per_w, n_r = 2, 300, 4
    expected = {f"d{i}" for i in range(200)}
    expected |= {f"w{w}_{i}" for w in range(n_w) for i in range(per_w)}

    errors = []
    reads = []
    err_lock = threading.Lock()
    stop = threading.Event()

    def record(e):
        with err_lock:
            errors.append(f"{type(e).__name__}: {e}")

    def writer(w):
        rs = np.random.RandomState(100 + w)
        for i in range(per_w):
            try:
                n = db.add(ids=[f"w{w}_{i}"], vectors=rs.standard_normal((1, D)).astype(np.float32))
                if n != 1:
                    record(AssertionError(f"add devolvió {n}"))
            except Exception as e:  # noqa: BLE001
                record(e)

    def reader(r):
        rs = np.random.RandomState(200 + r)
        count = 0
        while not stop.is_set():
            try:
                q = rs.standard_normal(D).astype(np.float32)
                res = db.search(q, k=5, include_metadata=True)
                assert isinstance(res, list) and 0 < len(res) <= 5, res
                got = _ids(res)
                assert len(set(got)) == len(got), f"ids duplicados: {got}"
                assert all(i in expected for i in got), f"ids inválidos: {got}"
                assert all(np.isfinite(x.score) for x in res)
                out = db.search(q, 3)
                mu = np.asarray(out[0]).reshape(-1, D)
                assert mu.shape[0] <= 3 and np.isfinite(mu).all()
                assert np.isfinite(np.asarray(out[1])).all()
            except Exception as e:  # noqa: BLE001
                record(e)
            count += 1
        reads.append(count)

    threads = [threading.Thread(target=writer, args=(w,), daemon=True) for w in range(n_w)]
    threads += [threading.Thread(target=reader, args=(r,), daemon=True) for r in range(n_r)]
    for t in threads:
        t.start()
    for t in threads[:n_w]:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in threads[:n_w]), "writers no terminaron (posible deadlock)"
    stop.set()
    for t in threads[n_w:]:
        t.join(timeout=30)

    assert errors == []
    assert len(reads) == n_r and min(reads) >= 20, f"lecturas por reader: {reads}"
    store = db.engine.m2m.splats
    assert store.n_active == 800 == len(db._splat_id_order) == len(db._vectors)
    assert len(set(db._splat_id_order)) == 800
    # Todos los docs buscables por su propio vector.
    sample = sorted(expected)[::8]
    for doc_id in sample:
        res = db.search(db._vectors[doc_id], k=1, include_metadata=True)
        assert _ids(res) == [doc_id], (doc_id, _ids(res))


# ---------------------------------------------------------------------------
# r1-23: add() atómico — validar ids ANTES de mutar
# ---------------------------------------------------------------------------


def test_add_unhashable_id_rejected_before_mutation():
    db = _db(n=3)
    snap = _snapshot(db)
    with pytest.raises(ValueError):
        db.add(ids=[["x"]], vectors=_vecs(1, 1))
    _assert_same(db, snap)
    with pytest.raises(ValueError):
        db.add(ids=["ok", ["bad"]], vectors=_vecs(2, 2))
    _assert_same(db, snap)
    assert "ok" not in db._vectors
    # El estado queda sano: las búsquedas con filtro no rompen después.
    q = _vecs(1, 3)[0]
    assert len(db.search(q, k=3, include_metadata=True)) == 3
    assert db.search(q, k=3, include_metadata=True, filter={"zz": 1}) == []


def test_add_duplicate_ids_in_batch_rejected_before_mutation():
    db = _db(n=2)
    snap = _snapshot(db)
    with pytest.raises(ValueError, match="(?i)duplic"):
        db.add(ids=["dup", "dup"], vectors=_vecs(2, 1))
    _assert_same(db, snap)
    assert db.add(ids=["dup"], vectors=_vecs(1, 2)) == 1
    assert db._splat_id_order.count("dup") == 1


def test_add_existing_active_id_rewrites_in_place():
    """Re-add de un id activo: sin splat huérfano, una sola ocurrencia, gana el nuevo."""
    db = SimpleVectorDB(latent_dim=D)
    v1, v2 = _vecs(1, 1), _vecs(1, 2)
    db.add(ids=["dup"], vectors=v1, metadata=[{"v": 1}])
    assert db.add(ids=["dup"], vectors=v2, metadata=[{"v": 2}]) == 1
    assert db._splat_id_order.count("dup") == 1
    assert db.engine.m2m.splats.n_active == 1
    np.testing.assert_allclose(db._vectors["dup"], v2[0])
    assert db._metadata["dup"] == {"v": 2}
    res = db.search(v1[0], k=5, include_metadata=True)
    assert _ids(res) == ["dup"]
    np.testing.assert_allclose(res[0].vector, normalize_sphere(v2[0]))


# ---------------------------------------------------------------------------
# r1-04: delete(id) -> add(id, vector nuevo) revive el id
# ---------------------------------------------------------------------------


def test_readd_after_soft_delete_revives_id():
    db = _db(n=5)
    old = db._vectors["d1"].copy()
    v = _vecs(1, 7)
    assert db.delete("d1").deleted == 1
    assert db.add(ids=["d1"], vectors=v) == 1
    assert "d1" not in db._deleted
    assert db._splat_id_order.count("d1") == 1
    assert db.engine.m2m.splats.n_active == 5
    assert db.get_stats()["active_documents"] == 5
    assert _ids(db.search(v[0], k=1, include_metadata=True)) == ["d1"]
    mu = _legacy_rows(db, old, 5)
    assert not any(_rows_close(mu, normalize_sphere(old)))


def test_update_upsert_restores_deleted_without_duplicates():
    db = _db(n=4)
    db.delete("d2")
    v = _vecs(1, 8)[0]
    assert db.update("d2", vector=v, upsert=True).success
    assert "d2" not in db._deleted
    assert db._splat_id_order.count("d2") == 1
    assert db.engine.m2m.splats.n_active == 4
    assert _ids(db.search(v, k=1, include_metadata=True)) == ["d2"]


# ---------------------------------------------------------------------------
# r1-06: delete(hard=True) saca el vector del índice
# ---------------------------------------------------------------------------


def test_hard_delete_removes_vector_from_index():
    db = _db(n=6)
    old = db._vectors["d1"].copy()
    assert db.delete("d1", hard=True).deleted == 1
    assert "d1" not in db._vectors
    assert db.engine.m2m.splats.n_active == 5 == len(db._splat_id_order)
    mu = _legacy_rows(db, old, 10)
    assert mu.shape[0] == 5
    assert not any(_rows_close(mu, normalize_sphere(old)))
    assert "d1" not in _ids(db.search(old, k=10, include_metadata=True))
    for i in (0, 2, 3, 4, 5):
        assert _ids(db.search(db._vectors[f"d{i}"], k=1, include_metadata=True)) == [f"d{i}"]

    v = _vecs(1, 9)
    assert db.add(ids=["d1"], vectors=v) == 1
    assert db._splat_id_order.count("d1") == 1
    assert db.engine.m2m.splats.n_active == 6
    assert _ids(db.search(v[0], k=1, include_metadata=True)) == ["d1"]
    assert _ids(db.search(old, k=10, include_metadata=True)).count("d1") <= 1
    mu2 = _legacy_rows(db, old, 10)
    assert not any(_rows_close(mu2, normalize_sphere(old)))


def test_hard_delete_frees_capacity():
    db = SimpleVectorDB(latent_dim=D, max_splats=5)
    db.add(ids=[f"d{i}" for i in range(5)], vectors=_vecs(5))
    assert db.add(ids=["x"], vectors=_vecs(1, 1)) == 0
    assert db.delete("d0", hard=True).deleted == 1
    assert db.add(ids=["x"], vectors=_vecs(1, 1)) == 1
    assert db.engine.m2m.splats.n_active == 5
    assert db.get_stats()["active_documents"] == 5


def test_hard_delete_in_lsh_mode_removes_from_lsh_index():
    db = SimpleVectorDB(latent_dim=D, enable_lsh_fallback=True, lsh_threshold=0.99)
    vecs = _vecs(20, 3)
    db.add(ids=[f"d{i}" for i in range(20)], vectors=vecs)
    assert db._use_lsh
    assert db.delete("d3", hard=True).deleted == 1
    assert len(db._lsh_id_map) == 19 == len(db.lsh.vectors) == len(db._splat_id_order)
    mu = _legacy_rows(db, vecs[3], 20)
    assert mu.shape[0] == 19
    assert not any(_rows_close(mu, normalize_sphere(vecs[3])))


# ---------------------------------------------------------------------------
# r1-21: bordes de delete()
# ---------------------------------------------------------------------------


def test_delete_id_zero_and_empty_string():
    db = SimpleVectorDB(latent_dim=D)
    vecs = _vecs(3, 4)
    db.add(ids=[0, 1, ""], vectors=vecs)
    assert db.delete(id=0).deleted == 1
    assert db.get_stats()["active_documents"] == 2
    assert 0 not in _ids(db.search(vecs[0], k=3, include_metadata=True))
    assert db.delete(id="").deleted == 1
    assert db.get_stats()["active_documents"] == 1


def test_delete_ids_ndarray():
    db = _db(n=5)
    assert db.delete(ids=np.array(["d0", "d2"])).deleted == 2
    assert db._deleted == {"d0", "d2"}
    assert db.delete(ids=np.array(["d1"])).deleted == 1


def test_delete_twice_counts_once():
    db = _db(n=5)
    assert [db.delete("d1").deleted for _ in range(3)] == [1, 0, 0]
    assert db.delete(ids=["d2", "d2", "d1", "d3"]).deleted == 2
    assert db.get_stats()["deleted_documents"] == 3
    assert [db.delete("d0", hard=True).deleted, db.delete("d0", hard=True).deleted] == [1, 0]


# ---------------------------------------------------------------------------
# r1-08: activación tardía de LSH
# ---------------------------------------------------------------------------


def test_late_lsh_activation_indexes_previous_docs():
    dim = 32
    db = SimpleVectorDB(latent_dim=dim, enable_lsh_fallback=True)
    early = _vecs(2, 5, dim)
    assert db.add(ids=["a0", "a1"], vectors=early) == 2  # n < 3: aún sin decisión
    assert not db._use_lsh
    late = _vecs(600, 6, dim)
    db.add(ids=[f"b{i}" for i in range(600)], vectors=late)
    assert db._use_lsh and db.lsh is not None
    assert len(db.lsh.vectors) == 602 == len(db._lsh_id_map) == len(db._splat_id_order)
    assert db._lsh_id_map[:2] == ["a0", "a1"]
    assert db._lsh_id_map == db._splat_id_order
    idx = db.search_batch(early[:1], k=5)
    for s in idx[0]:
        assert s >= 0 and db._lsh_id_map[s] == db._splat_id_order[s]
    assert _ids(db.search(early[0], k=1, include_metadata=True)) == ["a0"]
    new = _vecs(1, 7, dim)[0]
    db.update("a0", vector=new)
    assert np.allclose(db.lsh.vectors[0], normalize_sphere(new))
    assert _ids(db.search(new, k=1, include_metadata=True)) == ["a0"]


# ---------------------------------------------------------------------------
# r3b-02: M2MVectorStore.delete() — el top-k no lo ocupan docs borrados
# ---------------------------------------------------------------------------


def test_langchain_delete_does_not_starve_topk():
    pytest.importorskip("langchain_core")
    from langchain_core.embeddings import Embeddings

    from m2m import M2MConfig
    from m2m.integrations.langchain import M2MVectorStore

    class KeywordEmbeddings(Embeddings):
        """Determinista: el primer carácter del texto elige la dimensión activa."""

        def __init__(self, dim=32):
            self.dim = dim

        def _embed(self, text):
            v = np.zeros(self.dim, dtype=np.float32)
            v[ord(text.strip()[0]) % self.dim] = 1.0
            return v.tolist()

        def embed_documents(self, texts):
            return [self._embed(t) for t in texts]

        def embed_query(self, text):
            return self._embed(text)

    emb = KeywordEmbeddings(32)
    vs = M2MVectorStore(embeddings=emb, config=M2MConfig(device="cpu", latent_dim=32))
    texts = ["apple fruit red", "banana yellow fruit", "car engine"]
    ids = vs.add_texts(texts)
    assert len(ids) == 3
    assert len(vs.similarity_search("apple fruit red", k=3)) == 3

    vs.delete([ids[0]])
    assert len(vs._store) == 2
    r1 = vs.similarity_search("apple fruit red", k=1)
    assert len(r1) == 1 and r1[0].page_content != texts[0]
    r3 = vs.similarity_search("apple fruit red", k=3)
    assert sorted(d.page_content for d in r3) == sorted(texts[1:])
    # borrar dos veces / ids inexistentes es no-op
    vs.delete([ids[0], "ghost"])
    assert len(vs.similarity_search("car", k=2)) == 2
