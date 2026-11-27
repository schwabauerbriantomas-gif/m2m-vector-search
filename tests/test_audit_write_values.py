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
        np.testing.assert_allclose(eng.coarse_cluster_embeddings[cid][row], _sphere(newv), atol=1e-6)
        np.testing.assert_allclose(eng.embeddings[pos], _sphere(newv), atol=1e-6)
