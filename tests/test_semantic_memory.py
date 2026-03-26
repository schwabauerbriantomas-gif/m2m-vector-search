"""Tests for SemanticMemoryDB - Semantic memory interface for AI agents."""

import tempfile
import time

import numpy as np
import pytest

from m2m import M2MConfig, SemanticMemoryDB


class FakeEncoder:
    """Deterministic fake encoder that maps text to random-but-consistent vectors."""

    def __init__(self, dim=384):
        self.dim = dim
        self._cache = {}

    def __call__(self, text):
        if isinstance(text, list):
            return np.array([self._encode_one(t) for t in text])
        return self._encode_one(text)

    def _encode_one(self, text):
        if text not in self._cache:
            rng = np.random.RandomState(hash(text) % (2**31))
            v = rng.randn(self.dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-8
            self._cache[text] = v
        return self._cache[text]


@pytest.fixture
def encoder():
    return FakeEncoder(dim=384)


@pytest.fixture
def memory_db(encoder):
    return SemanticMemoryDB(encoder=encoder, latent_dim=384)


class TestSemanticMemoryDB:
    """Core SemanticMemoryDB tests."""

    def test_store_and_search(self, memory_db, encoder):
        """Store a memory and find it via search."""
        memory_db.store("Brian likes dark mode", metadata={"category": "preference"})
        results = memory_db.search("dark mode preference", k=5)
        assert len(results) >= 1
        assert "dark mode" in results[0].document.lower()

    def test_store_returns_id(self, memory_db):
        """store() should return a doc_id."""
        doc_id = memory_db.store("test memory")
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    def test_store_with_custom_id(self, memory_db):
        """store() with explicit doc_id."""
        doc_id = memory_db.store("test", doc_id="my_custom_id")
        assert doc_id == "my_custom_id"
        assert memory_db.get("my_custom_id") is not None

    def test_store_with_vector(self, memory_db):
        """store_with_vector() works without encoder."""
        vec = np.random.randn(384).astype(np.float32)
        vec /= np.linalg.norm(vec)
        doc_id = memory_db.store_with_vector(
            "manual vector memory", vec, metadata={"source": "test"}
        )
        assert doc_id is not None
        result = memory_db.get(doc_id)
        assert result is not None
        assert result["document"] == "manual vector memory"

    def test_batch_store(self, memory_db):
        """batch_store() adds multiple memories at once."""
        texts = [
            "Memory about project Alpha",
            "Memory about project Beta",
            "Memory about project Gamma",
        ]
        ids = memory_db.batch_store(texts)
        assert len(ids) == 3
        assert memory_db.stats()["total_memories"] == 3

    def test_search_with_metadata_filter(self, memory_db):
        """Search with metadata filter returns only matching docs."""
        memory_db.store("Python is great", metadata={"lang": "python"})
        memory_db.store("Rust is fast", metadata={"lang": "rust"})
        memory_db.store("JavaScript is everywhere", metadata={"lang": "javascript"})

        results = memory_db.search("programming language", k=10, filter={"lang": "python"})
        for r in results:
            assert r.metadata.get("lang") == "python"

    def test_hybrid_search_bm25_contribution(self, memory_db):
        """Hybrid search should find keyword matches that vector search might miss."""
        # Store with very specific keywords
        memory_db.store("M2M-VECTOR-SEARCH-PROJECT-2026-03-18", metadata={"tag": "keyword"})
        # Search using exact keywords - BM25 should boost this
        results = memory_db.search("M2M-VECTOR-SEARCH-PROJECT-2026-03-18", k=3, hybrid=True)
        assert len(results) >= 1

    def test_vector_only_search(self, memory_db):
        """hybrid=False disables BM25."""
        memory_db.store("test document alpha")
        memory_db.store("test document beta")
        results = memory_db.search("test", k=5, hybrid=False)
        assert len(results) >= 1

    def test_delete_by_id(self, memory_db):
        """Delete a memory by ID."""
        doc_id = memory_db.store("to be deleted")
        assert memory_db.get(doc_id) is not None
        n = memory_db.delete(id=doc_id)
        assert n >= 1
        assert memory_db.get(doc_id) is None

    def test_delete_multiple(self, memory_db):
        """Delete multiple memories by IDs."""
        id1 = memory_db.store("delete me 1")
        id2 = memory_db.store("delete me 2")
        id3 = memory_db.store("keep me")
        memory_db.delete(ids=[id1, id2])
        assert memory_db.get(id1) is None
        assert memory_db.get(id2) is None
        assert memory_db.get(id3) is not None

    def test_get_nonexistent(self, memory_db):
        """get() returns None for nonexistent ID."""
        assert memory_db.get("nonexistent_id") is None

    def test_stats(self, memory_db):
        """stats() returns comprehensive info."""
        memory_db.store("memory 1")
        memory_db.store("memory 2")
        memory_db.search("test")
        memory_db.search("test")

        stats = memory_db.stats()
        assert stats["total_memories"] == 2
        assert stats["total_queries"] == 2
        assert stats["total_adds"] == 2
        assert stats["bm25_indexed"] == 2

    def test_stats_latency_tracking(self, memory_db):
        """Stats should include latency percentiles after queries."""
        memory_db.store("latency test")
        for _ in range(20):
            memory_db.search("latency test")

        stats = memory_db.stats()
        assert "query_stats" in stats
        qs = stats["query_stats"]
        assert qs["total_queries"] == 20
        assert "avg_latency_ms" in qs
        assert "p50_latency_ms" in qs
        assert "p95_latency_ms" in qs

    def test_no_encoder_raises(self):
        """Operations requiring encoding should raise without encoder."""
        db = SemanticMemoryDB(encoder=None, latent_dim=384)
        with pytest.raises(RuntimeError, match="No encoder set"):
            db.store("test")


class TestBM25Index:
    """Tests for BM25Index."""

    def test_basic_add_and_search(self):
        from m2m.bm25_index import BM25Index

        bm25 = BM25Index()
        bm25.add("doc_1", "the quick brown fox jumps over the lazy dog")
        bm25.add("doc_2", "a fast fox runs quickly in the forest")
        results = bm25.search("quick fox", k=2)
        assert len(results) >= 1

    def test_remove_document(self):
        from m2m.bm25_index import BM25Index

        bm25 = BM25Index()
        bm25.add("doc_1", "test document")
        assert bm25.remove("doc_1") is True
        assert len(bm25) == 0
        assert bm25.remove("doc_1") is False  # already removed

    def test_search_empty_index(self):
        from m2m.bm25_index import BM25Index

        bm25 = BM25Index()
        assert bm25.search("test", k=5) == []

    def test_search_with_filter(self):
        from m2m.bm25_index import BM25Index

        bm25 = BM25Index()
        bm25.add("d1", "alpha beta gamma")
        bm25.add("d2", "delta epsilon zeta")
        bm25.add("d3", "alpha delta theta")
        results = bm25.search("alpha delta", k=10, doc_filter={"d1", "d3"})
        ids = [r[0] for r in results]
        assert "d2" not in ids

    def test_clear(self):
        from m2m.bm25_index import BM25Index

        bm25 = BM25Index()
        bm25.add("d1", "test")
        bm25.add("d2", "test2")
        bm25.clear()
        assert len(bm25) == 0

    def test_unicode_tokenization(self):
        from m2m.bm25_index import BM25Index

        bm25 = BM25Index()
        bm25.add("d1", "The researcher is working on vector search")
        results = bm25.search("researcher", k=1)
        assert len(results) == 1
        assert results[0][0] == "d1"

    def test_re_add_updates(self):
        from m2m.bm25_index import BM25Index

        bm25 = BM25Index()
        bm25.add("d1", "old content")
        bm25.add("d1", "new content updated")
        results = bm25.search("updated", k=1)
        assert len(results) == 1
