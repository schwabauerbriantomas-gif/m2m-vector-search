"""Tests T2: search_batch() en modo LSH (bug -1/[]) y parity vs search().

Dataset real: /tmp/m2mvs/embeddings.npy (659x384, rag_test). Si no existe,
los tests que lo usan se saltan (los de sintetico corren igual).
"""
import os

import numpy as np
import pytest

from m2m import SimpleVectorDB

EMB_PATH = "/tmp/m2mvs/embeddings.npy"
K = 10
N_QUERIES = 100
SEED = 42

HAS_DATASET = os.path.exists(EMB_PATH)


def _load_dataset():
    emb = np.load(EMB_PATH).astype(np.float32)
    rng = np.random.default_rng(SEED)
    qidx = rng.choice(emb.shape[0], size=N_QUERIES, replace=False)
    return emb, emb[qidx]


def _build_db(emb, enable_lsh_fallback):
    db = SimpleVectorDB(
        latent_dim=emb.shape[1],
        storage_path=None,
        enable_lsh_fallback=enable_lsh_fallback,
    )
    db.add(ids=[f"doc_{i}" for i in range(emb.shape[0])], vectors=emb)
    return db


def _brute_topk(emb, queries, k=K):
    sims = queries @ emb.T
    return np.argsort(-sims, axis=1)[:, :k]


# ---------------------------------------------------------------------------
# (c) Test ROJO: el bug documentado. Debe FALLAR con el codigo viejo.
# ---------------------------------------------------------------------------


def test_search_batch_lsh_returns_valid_results():
    """BUG 1: con config default (enable_lsh_fallback=True) y datos uniformes,
    search_batch() devolvia -1/[] para TODA query porque solo implementaba el
    path HRM2 (n_active=0 en splats cuando LSH esta activo)."""
    rng = np.random.RandomState(0)
    data = rng.randn(60, 16).astype(np.float32)
    db = _build_db(data, enable_lsh_fallback=True)
    assert db._use_lsh, "el dataset uniforme deberia activar LSH"

    queries = data[3:8]
    idx = db.search_batch(queries, k=5, include_metadata=False)
    assert isinstance(idx, np.ndarray)
    assert idx.shape == (5, 5)
    assert (idx >= 0).all(), f"search_batch devolvio indices invalidos: {idx}"

    res = db.search_batch(queries, k=5, include_metadata=True)
    assert len(res) == 5
    for row in res:
        assert len(row) == 5, f"search_batch devolvio fila vacia (bug -1/[]): {row}"
        assert all(r.id is not None for r in row)


def test_search_batch_lsh_indices_are_global_order():
    """Los indices de search_batch en modo LSH deben indexar _splat_id_order
    (misma semantica que el path HRM2)."""
    rng = np.random.RandomState(1)
    data = rng.randn(40, 12).astype(np.float32)
    db = _build_db(data, enable_lsh_fallback=True)
    assert db._use_lsh

    idx = db.search_batch(data[:3], k=5, include_metadata=False)
    res = db.search_batch(data[:3], k=5, include_metadata=True)
    for i in range(3):
        for j in range(5):
            s = int(idx[i, j])
            assert 0 <= s < len(db._splat_id_order)
            assert res[i][j].id == db._splat_id_order[s]


# ---------------------------------------------------------------------------
# (a) Parity search() vs search_batch() sobre el dataset real
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_DATASET, reason="dataset real no disponible")
def test_parity_search_vs_batch_lsh_mode():
    """En modo LSH (config default), search_batch debe devolver los mismos
    top-10 que search() single para >=95% de 100 queries del dataset real."""
    emb, queries = _load_dataset()
    db = _build_db(emb, enable_lsh_fallback=True)
    assert db._use_lsh, "rag_test deberia activar LSH con config default"

    single = [set(r.id for r in db.search(q, k=K, include_metadata=True)) for q in queries]
    batch = [set(r.id for r in row) for row in db.search_batch(queries, k=K, include_metadata=True)]
    matches = sum(1 for s, b in zip(single, batch) if s == b)
    assert matches >= 0.95 * N_QUERIES, (
        f"parity LSH {matches}/{N_QUERIES} < 95%"
    )


@pytest.mark.skipif(not HAS_DATASET, reason="dataset real no disponible")
def test_parity_search_vs_batch_hrm2_mode():
    """En modo HRM2 puro, search_batch debe devolver los mismos top-10 que
    search() single para >=95% de 100 queries del dataset real."""
    emb, queries = _load_dataset()
    db = _build_db(emb, enable_lsh_fallback=False)
    assert not db._use_lsh

    single = [set(r.id for r in db.search(q, k=K, include_metadata=True)) for q in queries]
    batch = [set(r.id for r in row) for row in db.search_batch(queries, k=K, include_metadata=True)]
    matches = sum(1 for s, b in zip(single, batch) if s == b)
    assert matches >= 0.95 * N_QUERIES, (
        f"parity HRM2 {matches}/{N_QUERIES} < 95%"
    )


# ---------------------------------------------------------------------------
# (b) Recall@10 de search_batch en modo HRM2 vs brute force
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_DATASET, reason="dataset real no disponible")
def test_search_batch_hrm2_recall():
    emb, queries = _load_dataset()
    db = _build_db(emb, enable_lsh_fallback=False)

    gt_sets = [set(f"doc_{i}" for i in g) for g in _brute_topk(emb, queries)]
    res = db.search_batch(queries, k=K, include_metadata=True)
    hits = sum(len(set(r.id for r in row) & g) for row, g in zip(res, gt_sets))
    recall = hits / (N_QUERIES * K)
    assert recall >= 0.99, f"recall@10 search_batch HRM2 = {recall:.4f} < 0.99"


@pytest.mark.skipif(not HAS_DATASET, reason="dataset real no disponible")
def test_search_batch_lsh_recall_floor():
    """Modo LSH: search_batch debe al menos igualar el recall del path single
    (mismo routing) — piso 0.70 segun baseline (0.716)."""
    emb, queries = _load_dataset()
    db = _build_db(emb, enable_lsh_fallback=True)
    assert db._use_lsh

    gt_sets = [set(f"doc_{i}" for i in g) for g in _brute_topk(emb, queries)]
    res = db.search_batch(queries, k=K, include_metadata=True)
    hits = sum(len(set(r.id for r in row) & g) for row, g in zip(res, gt_sets))
    recall = hits / (N_QUERIES * K)
    assert recall >= 0.70, f"recall@10 search_batch LSH = {recall:.4f} < 0.70"
