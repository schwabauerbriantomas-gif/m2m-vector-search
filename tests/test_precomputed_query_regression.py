"""Regression test: query should not crash with precomputed_embeddings (no splats)."""

import numpy as np
import pytest


def test_query_precomputed_embeddings():
    """query() with precomputed_embeddings should not raise IndexError."""
    from m2m.hrm2_engine import HRM2Engine

    emb = np.random.randn(100, 384).astype(np.float32)
    engine = HRM2Engine(metric="cosine")
    engine.index(precomputed_embeddings=emb)
    results = engine.query(emb[0], k=10)
    assert len(results) == 10
    for idx, dist in results:
        assert isinstance(idx, (int, np.integer))
        assert isinstance(dist, float)


def test_query_with_details_precomputed_embeddings():
    """query_with_details() with precomputed_embeddings should not raise IndexError."""
    from m2m.hrm2_engine import HRM2Engine

    emb = np.random.randn(100, 384).astype(np.float32)
    engine = HRM2Engine(metric="cosine")
    engine.index(precomputed_embeddings=emb)

    for lod in [0, 1, 2]:
        results = engine.query_with_details(emb[0], k=10, lod=lod)
        assert len(results) > 0, f"lod={lod} returned no results"
        for r in results:
            assert isinstance(r.splat_id, int)


def test_query_precomputed_euclidean():
    """Euclidean metric with precomputed_embeddings should also work."""
    from m2m.hrm2_engine import HRM2Engine

    emb = np.random.randn(50, 128).astype(np.float32)
    engine = HRM2Engine(metric="euclidean")
    engine.index(precomputed_embeddings=emb)
    results = engine.query(emb[0], k=5)
    assert len(results) == 5
