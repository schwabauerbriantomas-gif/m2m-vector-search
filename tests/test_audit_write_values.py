"""Regresión del pack P1 — valores que entran por add()/update() (fix/p1).

Un test por criterio de aceptación de BACKLOG.md §P1; el id de auditoría va en
el nombre (hallazgos r1.json). Base: a74d9e4.
"""

import logging

import numpy as np
import pytest

import m2m
from m2m.storage.persistence import M2MPersistence

D = 8
_rng = np.random.default_rng(5)
X = _rng.standard_normal((64, D)).astype(np.float32)


def _sphere(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _db(n=5, **kw):
    db = m2m.SimpleVectorDB(latent_dim=D, **kw)
    ids = [f"doc{i}" for i in range(n)]
    db.add(ids=ids, vectors=X[:n])
    return db, ids


def _mu(db):
    s = db.engine.m2m.splats
    return s.mu[: s.n_active]


def _snapshot(db):
    return {k: v.copy() for k, v in db._vectors.items()}


class TestR102UpdateNaNEInf:
    """Criterio 1: update(vector=NaN|Inf) → ValueError, estado intacto, retrain posterior OK."""

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
    def test_update_non_finite_vector_raises_value_error(self, bad_value):
        db, ids = _db()
        bad = X[1].copy()
        bad[0] = bad_value
        with pytest.raises(ValueError):
            db.update(ids[1], vector=bad)

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf])
    def test_update_non_finite_vector_leaves_state_intact(self, bad_value):
        db, ids = _db()
        before_vecs, before_mu = _snapshot(db), _mu(db).copy()
        bad = X[1].copy()
        bad[2] = bad_value
        with pytest.raises(ValueError):
            db.update(ids[1], vector=bad, metadata={"g": 99}, document="poisoned")
        assert "g" not in db._metadata[ids[1]]
        assert db._documents[ids[1]] is None
        for k, v in before_vecs.items():
            np.testing.assert_array_equal(db._vectors[k], v)
        np.testing.assert_array_equal(_mu(db), before_mu)
        assert np.isfinite(_mu(db)).all()

    @pytest.mark.parametrize("bad_value", [np.nan, np.inf])
    def test_add_after_rejected_update_triggers_full_retrain_and_works(self, bad_value):
        # 2 docs entrenan el índice (last_trained_n=2); el 3er add (3 > 1.25*2) fuerza retrain completo
        db = m2m.SimpleVectorDB(latent_dim=D)
        db.add(ids=["i0", "i1"], vectors=X[:2])
        assert db.engine.m2m.splats._last_trained_n == 2
        bad = np.full(D, bad_value, dtype=np.float32)
        with pytest.raises(ValueError):
            db.update("i0", vector=bad)
        assert db.add(ids=["post"], vectors=X[2:3]) == 1
        s = db.engine.m2m.splats
        assert s.n_active == 3 and s._last_trained_n == 3
        assert np.isfinite(s.mu[: s.n_active]).all()
        assert [r.id for r in db.search(X[0], k=1, include_metadata=True)] == ["i0"]


class TestR119ZeroVectors:
    """Criterio 2: norma < 1e-6 → ValueError en add/update; upsert sin vector → ValueError."""

    def test_add_zero_vector_raises_value_error(self):
        db = m2m.SimpleVectorDB(latent_dim=D)
        with pytest.raises(ValueError, match="zero"):
            db.add(ids=["z"], vectors=np.zeros((1, D), dtype=np.float32))
        assert db._vectors == {} and db.engine.m2m.splats.n_active == 0

    def test_add_batch_with_one_zero_row_rejected_atomically(self):
        db, ids = _db(3)
        batch = np.vstack([X[3:5], np.zeros((1, D), np.float32)])
        with pytest.raises(ValueError):
            db.add(ids=["n0", "n1", "z"], vectors=batch)
        assert list(db._vectors) == ids and db._splat_id_order == ids
        assert db.engine.m2m.splats.n_active == 3

    def test_update_zero_vector_raises_value_error_and_keeps_state(self):
        db, ids = _db()
        before_vecs, before_mu = _snapshot(db), _mu(db).copy()
        with pytest.raises(ValueError, match="zero"):
            db.update(ids[2], vector=np.zeros(D, dtype=np.float32), metadata={"g": 1})
        for k, v in before_vecs.items():
            np.testing.assert_array_equal(db._vectors[k], v)
        np.testing.assert_array_equal(_mu(db), before_mu)
        assert "g" not in db._metadata[ids[2]]

    def test_near_zero_rejected_but_small_nonzero_norm_accepted(self):
        db = m2m.SimpleVectorDB(latent_dim=D)
        with pytest.raises(ValueError):  # norma ≈ 2.8e-7
            db.add(ids=["tiny"], vectors=np.full((1, D), 1e-7, dtype=np.float32))
        small = np.full(D, 1e-4, dtype=np.float32)  # norma ≈ 2.8e-4
        assert db.add(ids=["small"], vectors=small[np.newaxis, :]) == 1
        assert db.update("small", vector=small).success

    def test_upsert_new_id_without_vector_raises_value_error(self):
        db, _ = _db(3)
        with pytest.raises(ValueError, match="vector"):
            db.update("brand_new", metadata={"x": 1}, upsert=True)
        assert "brand_new" not in db._vectors
        assert db.engine.m2m.splats.n_active == 3

    def test_upsert_deleted_id_without_vector_raises_and_stays_deleted(self):
        db, ids = _db(3)
        db.delete(id=ids[0])
        with pytest.raises(ValueError, match="vector"):
            db.update(ids[0], metadata={"x": 1}, upsert=True)
        assert ids[0] in db._deleted
        assert db.engine.m2m.splats.n_active == 3

    def test_upsert_with_vector_still_creates_document(self):
        db, _ = _db(3)
        assert db.update("brand_new", vector=X[7], metadata={"x": 1}, upsert=True).success
        np.testing.assert_allclose(db._vectors["brand_new"], X[7])
        assert [r.id for r in db.search(X[7], k=1, include_metadata=True)] == ["brand_new"]

    def test_upsert_without_vector_no_longer_inserts_hub_doc(self):
        db, ids = _db(6)
        q = _rng.standard_normal(D).astype(np.float32)
        with pytest.raises(ValueError):
            db.update("hub", metadata={"x": 1}, upsert=True)
        res = [r.id for r in db.search(q, k=3, include_metadata=True)]
        assert "hub" not in res and set(res) <= set(ids)


class TestR120IndependentCopy:
    """Criterio 3: db._vectors[id] es copia independiente del array del llamador."""

    def test_add_stores_independent_copy_of_caller_array(self):
        db = m2m.SimpleVectorDB(latent_dim=D)
        Xa = _rng.standard_normal((3, D)).astype(np.float32)
        orig = Xa.copy()
        db.add(ids=["a", "b", "c"], vectors=Xa)
        assert not np.shares_memory(db._vectors["a"], Xa)
        Xa[0] = 999.0
        for i, k in enumerate("abc"):
            np.testing.assert_array_equal(db._vectors[k], orig[i])

    def test_add_1d_vector_stores_independent_copy(self):
        db = m2m.SimpleVectorDB(latent_dim=D)
        v = np.ones(D, dtype=np.float32)
        db.add(ids=["one"], vectors=v)
        v[:] = 999.0
        np.testing.assert_array_equal(db._vectors["one"], np.ones(D, dtype=np.float32))

    def test_add_legacy_positional_stores_independent_copy(self):
        db = m2m.SimpleVectorDB(latent_dim=D)
        Xa = _rng.standard_normal((2, D)).astype(np.float32)
        orig = Xa.copy()
        assert db.add(Xa) == 2
        Xa[1] = 999.0
        for vec in db._vectors.values():
            assert any(np.array_equal(vec, o) for o in orig)

    def test_update_stores_independent_copy(self):
        db, ids = _db()
        v = X[9].copy()
        db.update(ids[1], vector=v)
        assert not np.shares_memory(db._vectors[ids[1]], v)
        v[:] = 999.0
        np.testing.assert_array_equal(db._vectors[ids[1]], X[9])

    def test_upsert_stores_independent_copy(self):
        db, _ = _db(3)
        v = X[9].copy()
        db.update("new", vector=v, upsert=True)
        v[:] = 999.0
        np.testing.assert_array_equal(db._vectors["new"], X[9])


class TestR124EmbeddingsSync:
    """Criterio 4: tras update(id, vector=v), engine.embeddings[pos] == normalize_sphere(v)."""

    def test_update_vector_updates_engine_embeddings(self):
        db, ids = _db(40)
        newv = _rng.standard_normal(D).astype(np.float32)
        pos = db._splat_id_order.index(ids[3])
        db.update(ids[3], vector=newv)
        s = db.engine.m2m.splats
        eng = s.engine
        assert eng.embeddings is not None and len(eng.embeddings) == s.n_active
        np.testing.assert_allclose(eng.embeddings[pos], _sphere(newv), atol=1e-6)
        assert not np.allclose(eng.embeddings[pos], _sphere(X[3]))

    def test_update_vector_same_cluster_refreshes_cluster_embeddings(self):
        # sin cambio de cluster la fila de coarse_cluster_embeddings también debe refrescarse
        db, ids = _db(40)
        eng = db.engine.m2m.splats.engine
        pos = db._splat_id_order.index(ids[5])
        old_cid = int(eng.coarse_assignments[pos])
        newv = X[5] + 1e-3 * _rng.standard_normal(D).astype(np.float32)
        db.update(ids[5], vector=newv)
        assert int(eng.coarse_assignments[pos]) == old_cid  # precondición: no se movió
        cid = old_cid
        row = int(np.flatnonzero(eng.coarse_cluster_indices[cid] == pos)[0])
        np.testing.assert_allclose(
            eng.coarse_cluster_embeddings[cid][row], _sphere(newv), atol=1e-6
        )
        np.testing.assert_allclose(eng.embeddings[pos], _sphere(newv), atol=1e-6)


class TestR126CapacityHonesty:
    """Criterio 6: al llegar a max_splats, los descartados no se guardan, hay warning con ids
    y get_stats() cuenta solo indexados."""

    def test_rejected_not_stored_not_persisted_and_warning_logged(self, tmp_path, caplog):
        db = m2m.SimpleVectorDB(
            latent_dim=D, max_splats=10, storage_path=str(tmp_path / "st"), enable_wal=False
        )
        r1 = db.add(ids=[f"s{i}" for i in range(6)], vectors=X[:6])
        with caplog.at_level(logging.WARNING, logger="m2m"):
            r2 = db.add(ids=[f"s{i}" for i in range(6, 16)], vectors=X[6:16])
        assert (r1, r2) == (6, 4)
        accepted = {f"s{i}" for i in range(10)}
        assert set(db._vectors) == accepted
        assert set(db._metadata) == accepted
        assert len(db._splat_id_order) == 10 == db.engine.m2m.splats.n_active
        stats = db.get_stats()
        assert stats["total_documents"] == 10 and stats["active_documents"] == 10
        assert sorted(r.id for r in db.search(X[7], k=10, include_metadata=True)) == sorted(
            accepted
        )
        # storage: solo los indexados (shard_002 = 4 filas, no 10)
        assert set(db.storage.get_all_ids()) == accepted
        assert db.storage.get_metadata("s10") is None
        assert db.storage.load_vectors("shard_002").shape == (4, D)
        # warning con los ids descartados
        msgs = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
        assert msgs, "se esperaba un warning de capacidad"
        assert all(f"s{i}" in msgs for i in range(10, 16)), msgs
        assert "max_splats=10" in msgs

    def test_add_when_already_full_returns_zero_without_storing(self, caplog):
        db = m2m.SimpleVectorDB(latent_dim=D, max_splats=5)
        assert db.add(ids=[f"a{i}" for i in range(5)], vectors=X[:5]) == 5
        with caplog.at_level(logging.WARNING, logger="m2m"):
            assert db.add(ids=["b0", "b1"], vectors=X[5:7]) == 0
        assert set(db._vectors) == {f"a{i}" for i in range(5)}
        assert db.get_stats()["active_documents"] == 5
        assert "b1" in " ".join(r.getMessage() for r in caplog.records)


def _read_stored(p, doc_id):
    meta = p.get_metadata(doc_id)
    assert meta is not None, f"{doc_id} sin fila en metadata.db"
    shard = p.load_vectors(f"shard_{int(meta['shard_idx']):03d}")
    return shard[int(meta["vector_idx"])], meta


class TestR109UpdatedVectorPersisted:
    """Criterio 5: tras update(id, vector=v) el vector nuevo queda escrito en storage
    (verificado con una instancia fresca de M2MPersistence)."""

    @pytest.mark.parametrize("enable_wal", [False, True])
    def test_update_vector_writes_new_vector_to_storage(self, tmp_path, enable_wal):
        st = str(tmp_path / "st")
        db = m2m.SimpleVectorDB(latent_dim=D, storage_path=st, enable_wal=enable_wal)
        db.add(ids=["a", "b", "c"], vectors=X[:3])
        newv = X[40].copy()
        assert db.update("a", vector=newv, metadata={"g": 99}, document="updated").success
        fresh = M2MPersistence(st, enable_wal=False)
        try:
            got, meta = _read_stored(fresh, "a")
            np.testing.assert_allclose(got, newv, atol=1e-6)
            assert (meta["metadata"], meta["document"]) == ({"g": 99}, "updated")
            np.testing.assert_array_equal(_read_stored(fresh, "b")[0], X[1])
            np.testing.assert_array_equal(_read_stored(fresh, "c")[0], X[2])
            assert len(fresh.list_shards()) == 1  # reescritura in-place: el layout no cambia
        finally:
            fresh.close()

    def test_update_metadata_only_keeps_stored_vector(self, tmp_path):
        st = str(tmp_path / "st")
        db = m2m.SimpleVectorDB(latent_dim=D, storage_path=st, enable_wal=False)
        db.add(ids=["a", "b"], vectors=X[:2])
        db.update("b", metadata={"k": 1})
        fresh = M2MPersistence(st, enable_wal=False)
        try:
            got, meta = _read_stored(fresh, "b")
            np.testing.assert_array_equal(got, X[1])
            assert meta["metadata"] == {"k": 1}
        finally:
            fresh.close()

    def test_update_vector_method_fresh_instance_reads_new_value(self, tmp_path):
        st = str(tmp_path / "st")
        p = M2MPersistence(st, enable_wal=False)
        p.save_vectors(X[:3], ["a", "b", "c"])
        for i, d in enumerate("abc"):
            p.save_metadata(d, shard_idx=1, vector_idx=i)
        newv = X[41].copy()
        assert p.update_vector("b", newv) is True
        p.close()
        fresh = M2MPersistence(st, enable_wal=False)
        try:
            np.testing.assert_array_equal(_read_stored(fresh, "b")[0], newv)
            np.testing.assert_array_equal(_read_stored(fresh, "a")[0], X[0])
            np.testing.assert_array_equal(_read_stored(fresh, "c")[0], X[2])
            assert fresh.list_shards() == ["shard_001"]
        finally:
            fresh.close()

    def test_update_vector_falls_back_to_new_shard_when_row_missing(self, tmp_path):
        st = str(tmp_path / "st")
        p = M2MPersistence(st, enable_wal=False)
        p.save_vectors(X[:2], ["a", "b"])
        p.save_metadata("a", shard_idx=1, vector_idx=0)
        p.save_metadata("orphan", shard_idx=7, vector_idx=3)  # shard inexistente
        p.save_metadata("oob", shard_idx=1, vector_idx=99)  # fila fuera de rango
        assert p.update_vector("orphan", X[10]) is True
        assert p.update_vector("oob", X[11]) is True
        p.close()
        fresh = M2MPersistence(st, enable_wal=False)
        try:
            np.testing.assert_array_equal(_read_stored(fresh, "orphan")[0], X[10])
            np.testing.assert_array_equal(_read_stored(fresh, "oob")[0], X[11])
            np.testing.assert_array_equal(_read_stored(fresh, "a")[0], X[0])  # intacto
            np.testing.assert_array_equal(fresh.load_vectors("shard_001")[1], X[1])
        finally:
            fresh.close()

    def test_update_vector_unknown_id_returns_false(self, tmp_path):
        p = M2MPersistence(str(tmp_path / "st"), enable_wal=False)
        try:
            assert p.update_vector("ghost", X[0]) is False
            assert p.list_shards() == []
        finally:
            p.close()
