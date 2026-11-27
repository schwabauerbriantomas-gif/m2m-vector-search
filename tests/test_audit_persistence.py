"""Audit r1-01 / r1-17 / r3a-17 — persistencia y round trip (pack P4).

Cada test mapea 1:1 a un criterio de aceptación del BACKLOG (P4 `fix/p4-restore`).
Base de las sondas: ~/swarm_m2m_vs_audit/work/orch/verify_hot.py (V1) y
~/swarm_m2m_vs_audit/work/r1/p6_persist.py.
"""

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pytest

import m2m
from m2m.storage.persistence import M2MPersistence

D = 32
_rng = np.random.default_rng(4)
X = _rng.standard_normal((64, D)).astype(np.float32)
IDS = [f"a{i}" for i in range(50)]
METAS = [{"g": i % 3, "txt": f"m{i}"} for i in range(50)]
DOCS = [f"document {i}" for i in range(50)]


def _db(path, **kw):
    return m2m.SimpleVectorDB(latent_dim=D, storage_path=str(path), enable_wal=False, **kw)


def _top(db, q, k=1):
    return [r.id for r in db.search(q, k=k, include_metadata=True)[:k]]


def _build(path, **kw):
    db = _db(path, **kw)
    db.add(ids=IDS, vectors=X[:50], metadata=METAS, documents=DOCS)
    return db


def _stats(db):
    return {k: v for k, v in db.get_stats().items() if k != "storage"}


# ---------------------------------------------------------------------------
# Criterio 1 (r1-01): add → save → reopen reconstruye el índice
# ---------------------------------------------------------------------------


class TestCriterio1RoundTrip:
    def test_reopen_rebuilds_index_and_finds_each_doc(self, tmp_path):
        st = tmp_path / "st"
        db = _build(st)
        db.delete(id="a7")
        stats_before, order_before = _stats(db), list(db._splat_id_order)
        db.save(str(st))

        db2 = _db(st)
        store = db2.engine.m2m.splats
        # índice reconstruido + _splat_id_order alineado 1:1 con las posiciones
        assert store.n_active == 50 == len(db2._splat_id_order)
        assert store.engine._is_indexed
        assert db2._splat_id_order == order_before
        # soft-delete restaurado (no reaparece) y stats iguales
        assert db2._deleted == {"a7"}
        assert _stats(db2) == stats_before
        assert _stats(db2)["active_documents"] == 49
        # top-1 = el propio vector para cada doc activo
        for i in range(50):
            if i == 7:
                continue
            assert _top(db2, X[i])[0] == f"a{i}", i
        assert "a7" not in [r.id for r in db2.search(X[7], k=10, include_metadata=True)]
        # metadata / documents restaurados
        assert db2._metadata["a3"] == {"g": 0, "txt": "m3"}
        assert db2._documents["a3"] == "document 3"
        res = db2.search(X[3], k=1, include_metadata=True)[0]
        assert res.id == "a3" and res.metadata == {"g": 0, "txt": "m3"}
        assert res.document == "document 3"

    def test_batch_search_after_reopen(self, tmp_path):
        st = tmp_path / "st"
        db = _build(st)
        before = db.search_batch(X[:8], k=3)
        db.save(str(st))
        db2 = _db(st)
        after = db2.search_batch(X[:8], k=3)
        assert after.shape == (8, 3)
        assert (after >= 0).all()
        np.testing.assert_array_equal(after, before)
        assert int(after[3, 0]) == 3  # posición de a3 en _splat_id_order
        allowed = np.array([db2._metadata[d]["g"] == 1 for d in db2._splat_id_order], dtype=bool)
        filt = db2.search_batch_filtered(X[:2], k=3, filter_fn=lambda i: bool(allowed[i]))
        assert filt.shape == (2, 3) and (filt >= 0).all()
        assert all(allowed[int(p)] for p in filt.ravel())

    def test_order_is_insertion_order_across_shards(self, tmp_path):
        st = tmp_path / "st"
        db = _db(st)
        db.add(ids=["b0", "b1", "b2"], vectors=X[50:53])
        db.add(ids=IDS[:20], vectors=X[:20], metadata=METAS[:20], documents=DOCS[:20])
        db.add(ids=["c0", "c1"], vectors=X[53:55])
        order = ["b0", "b1", "b2"] + IDS[:20] + ["c0", "c1"]
        vecs = [X[50], X[51], X[52]] + [X[i] for i in range(20)] + [X[53], X[54]]
        assert list(db._splat_id_order) == order
        db.save(str(st))
        db2 = _db(st)
        assert db2._splat_id_order == order
        for doc_id, v in zip(order, vecs):
            np.testing.assert_allclose(db2._vectors[doc_id], v, atol=1e-6)
        store = db2.engine.m2m.splats
        norm = np.stack(vecs) / np.linalg.norm(np.stack(vecs), axis=1, keepdims=True)
        np.testing.assert_allclose(store.mu[: len(order)], norm, atol=1e-5)
        # KMeans completo, igual que un add() del mismo tamaño
        assert db2.engine.m2m.splats._last_trained_n == len(order)

    # Criterio 3: update(vector) → save → reopen devuelve el vector nuevo
    def test_update_vector_survives_reopen(self, tmp_path):
        st = tmp_path / "st"
        db = _build(st)
        assert db.update("a1", vector=X[60], metadata={"g": 99}, document="updated").success
        db.save(str(st))
        db2 = _db(st)
        np.testing.assert_allclose(db2._vectors["a1"], X[60], atol=1e-6)
        assert (db2._metadata["a1"], db2._documents["a1"]) == ({"g": 99}, "updated")
        assert _top(db2, X[60])[0] == "a1"
        assert _top(db2, X[1])[0] != "a1"

    def test_soft_delete_survive_reopen(self, tmp_path):
        st = tmp_path / "st"
        db = _build(st)
        db.delete(id="a2")
        db.save(str(st))
        db2 = _db(st)
        assert db2._deleted == {"a2"}
        s = db2.get_stats()
        assert (s["total_documents"], s["active_documents"], s["deleted_documents"]) == (50, 49, 1)
        # a2 NO reaparece: su splat sigue en el índice (misma posición que antes del
        # save) pero search lo salta por _deleted. (k>1 porque search aún no hace
        # backfill: r1-05 es del pack P3.)
        res = db2.search(X[2], k=5, include_metadata=True)
        assert res and "a2" not in [r.id for r in res]
        # el ciclo de vida sigue: upsert sobre el borrado tras el reopen
        assert db2.update("a2", vector=X[61], upsert=True).success
        assert "a2" not in db2._deleted

    def test_legacy_db_without_deleted_column_restores(self, tmp_path):
        st = tmp_path / "legacy"
        p = M2MPersistence(str(st), enable_wal=False)
        p.save_vectors(X[:5], IDS[:5])
        for i, d in enumerate(IDS[:5]):
            p.save_metadata(d, shard_idx=1, vector_idx=i, metadata=METAS[i], document=DOCS[i])
        p.close()
        db_path = Path(st) / "data" / "metadata" / "metadata.db"
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            "SELECT id, shard_idx, vector_idx, metadata, document, created_at, updated_at "
            "FROM documents ORDER BY rowid"
        ).fetchall()
        conn.execute("DROP TABLE documents")
        conn.execute(
            "CREATE TABLE documents (id TEXT PRIMARY KEY, shard_idx INTEGER, vector_idx INTEGER, "
            "metadata TEXT, document TEXT, created_at REAL, updated_at REAL)"
        )
        conn.executemany("INSERT INTO documents VALUES (?,?,?,?,?,?,?)", rows)
        conn.commit()
        conn.close()
        db = _db(st)
        assert set(db._vectors) == set(IDS[:5]) and db._deleted == set()
        assert _top(db, X[2])[0] == "a2"

    def test_restore_over_capacity_warns_and_keeps_first_max_splats(self, tmp_path, caplog):
        st = tmp_path / "st"
        db = _build(st, max_splats=1000)
        order = list(db._splat_id_order)
        db.save(str(st))
        with caplog.at_level(logging.WARNING, logger="m2m"):
            db2 = _db(st, max_splats=40)
        store = db2.engine.m2m.splats
        assert store.n_active == 40
        assert db2._splat_id_order == order[:40]
        assert set(db2._vectors) == set(order[:40])
        msgs = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
        assert "max_splats=40" in msgs, msgs
        assert _top(db2, X[39])[0] == "a39"

    def test_lsh_mode_reopen(self, tmp_path):
        st = tmp_path / "st"
        kw = dict(latent_dim=D, storage_path=str(st), enable_wal=False, enable_lsh_fallback=True)
        db = m2m.SimpleVectorDB(**kw)
        db.add(ids=IDS, vectors=X[:50], metadata=METAS, documents=DOCS)
        assert db._use_lsh  # precondición: distribución uniforme → LSH
        db.save(str(st))
        db2 = m2m.SimpleVectorDB(**kw)
        assert db2._use_lsh and db2.lsh is not None
        assert db2._lsh_id_map == IDS == db2._splat_id_order
        res = db2.search(X[3], k=5, include_metadata=True)
        ids = [r.id for r in res]
        assert ids and set(ids) <= set(IDS) and ids[0] == "a3"
        assert res[0].metadata == {"g": 0, "txt": "m3"} and res[0].document == "document 3"
        out = db2.search_batch(X[:2], k=3)
        assert out.shape == (2, 3) and (out >= 0).all()


