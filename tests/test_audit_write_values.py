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
X = _rng.standard_normal((16, D)).astype(np.float32)


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
