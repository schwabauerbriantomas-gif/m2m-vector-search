"""
Tests for Phase 2 features: temporal decay, auto-categorize, fusion methods, chaos testing.

Research-backed via Z.AI tools investigation (2026-03-18).
"""
import math
import time
import uuid

import numpy as np
import pytest

from m2m.alfred_memory import AlfredMemoryDB, MemoryResult, auto_categorize
from m2m.bm25_index import BM25Index


# ---------------------------------------------------------------------------
# Mock encoder (deterministic, fast)
# ---------------------------------------------------------------------------
def _mock_encoder(text):
    """Deterministic encoder: hashes text into a vector."""
    if isinstance(text, list):
        return np.array([_mock_encoder(t).squeeze() for t in text], dtype=np.float32)
    text = text if isinstance(text, str) else str(text)
    # Deterministic hash-based encoding
    seed = abs(hash(text)) % (2**31)
    rng = np.random.RandomState(seed)
    vec = rng.randn(384).astype(np.float32)
    vec /= (np.linalg.norm(vec) + 1e-8)
    return vec


def _make_db(**kwargs):
    return AlfredMemoryDB(encoder=_mock_encoder, latent_dim=384, **kwargs)


# ===========================================================================
# Auto-categorize tests
# ===========================================================================
class TestAutoCategorize:
    def test_decision_category(self):
        assert auto_categorize("We decided to use Python for the project") == "decision"

    def test_preference_category(self):
        assert auto_categorize("User prefers dark mode for coding") == "preference"

    def test_project_category(self):
        assert auto_categorize("New feature sprint for M2M implementation") == "project"

    def test_error_category(self):
        assert auto_categorize("Fix the bug in the encoder module") == "error"

    def test_learning_category(self):
        assert auto_categorize("I learned that BM25 is better for exact matches") == "learning"

    def test_question_category(self):
        assert auto_categorize("Question: how to configure HNSW?") == "question"

    def test_conversation_category(self):
        assert auto_categorize("Brian told me about the new GPU setup") == "conversation"

    def test_task_category(self):
        assert auto_categorize("Todo: review PR #42 by Friday") == "task"

    def test_config_category(self):
        assert auto_categorize("Setup: installed sentence-transformers v3") == "config"

    def test_no_match(self):
        assert auto_categorize("The quick brown fox jumps over the lazy dog") is None

    def test_empty_string(self):
        assert auto_categorize("") is None

    def test_unicode_text(self):
        result = auto_categorize("Decisión: usar M2M para memoria semántica 🎩")
        assert result == "decision"


# ===========================================================================
# Temporal decay tests
# ===========================================================================
class TestTemporalDecay:
    def test_recent_memory_boosted(self):
        """Recent memories should score higher than old ones."""
        db = _make_db(temporal_decay=True, temporal_half_life_days=7.0)

        # Store two similar memories
        id1 = db.store("User prefers Python for ML", metadata={"date": "2026-03-18"})
        time.sleep(0.01)
        id2 = db.store("User likes Java for enterprise", metadata={"date": "2026-03-18"})

        # Artificially age id1
        db._timestamps[id1] = time.time() - (14 * 86400)  # 14 days ago (2 half-lives)

        results = db.search("user preference", k=5)
        assert len(results) >= 2

        # The recent one (id2) should rank higher
        ids = [r.id for r in results]
        assert ids.index(id2) < ids.index(id1)

    def test_no_decay_when_disabled(self):
        """Without temporal_decay, order should be based on relevance only."""
        db = _make_db(temporal_decay=False)

        id1 = db.store("Python programming language")
        time.sleep(0.01)
        id2 = db.store("Python snake animal")

        # Age id1
        db._timestamps[id1] = time.time() - (100 * 86400)

        results = db.search("Python programming", k=5)
        # id1 should rank first (more relevant text)
        assert results[0].id == id1

    def test_half_life_math(self):
        """Verify the exponential decay math."""
        db = _make_db(temporal_decay=True, temporal_half_life_days=10.0)

        doc_id = "test_doc"
        now = time.time()
        db._timestamps[doc_id] = now

        # At t=0, boost should be ~1.0
        boost_0 = db._compute_temporal_boost(doc_id)
        assert abs(boost_0 - 1.0) < 0.01

        # At 1 half-life, boost should be ~0.5
        db._timestamps[doc_id] = now - (10 * 86400)
        boost_half = db._compute_temporal_boost(doc_id)
        assert abs(boost_half - 0.5) < 0.05

        # At 2 half-lives, boost should be ~0.25
        db._timestamps[doc_id] = now - (20 * 86400)
        boost_double = db._compute_temporal_boost(doc_id)
        assert abs(boost_double - 0.25) < 0.05

    def test_temporal_boost_clamp(self):
        """Boost should never go below 0.01."""
        db = _make_db(temporal_decay=True, temporal_half_life_days=1.0)
        db._timestamps["old_doc"] = time.time() - (365 * 86400)  # 1 year
        boost = db._compute_temporal_boost("old_doc")
        assert boost >= 0.01

    def test_unknown_doc_boost(self):
        """Docs without timestamp get boost=1.0."""
        db = _make_db(temporal_decay=True)
        boost = db._compute_temporal_boost("nonexistent")
        assert boost == 1.0


# ===========================================================================
# Fusion method tests
# ===========================================================================
class TestFusionMethods:
    def test_rrf_fusion(self):
        db = _make_db(fusion_method="rrf")
        db.batch_store([f"Document about topic alpha {i}" for i in range(10)])
        results = db.search("topic alpha", k=5)
        assert len(results) >= 1
        assert all(r.score > 0 for r in results)

    def test_weighted_fusion(self):
        db = _make_db(fusion_method="weighted", hybrid_weight=0.7)
        db.batch_store([f"Machine learning with Python {i}" for i in range(10)])
        results = db.search("machine learning Python", k=5)
        assert len(results) >= 1

    def test_vector_only(self):
        db = _make_db(fusion_method="vector_only")
        db.store("quantum physics theory of everything")
        db.store("baking chocolate cake recipe")
        results = db.search("quantum theory", k=2)
        assert len(results) >= 1
        # First result should be about quantum
        assert "quantum" in results[0].document.lower()

    def test_bm25_only(self):
        db = _make_db(fusion_method="bm25_only")
        db.store("Python programming tutorial guide")
        db.store("Snake animal reptile zoo")
        results = db.search("Python programming tutorial", k=2)
        assert len(results) >= 1
        # BM25 should rank the one with matching keywords
        assert "programming" in results[0].document.lower()

    def test_invalid_fusion_method(self):
        with pytest.raises(ValueError, match="fusion_method"):
            _make_db(fusion_method="invalid")

    def test_hybrid_false_overrides(self):
        db = _make_db(fusion_method="rrf")
        db.store("test document alpha")
        results = db.search("test", k=5, hybrid=False)
        assert len(results) >= 1


# ===========================================================================
# Score normalization tests
# ===========================================================================
class TestScoreNormalization:
    def test_normalize_basic(self):
        db = _make_db()
        scores = {"a": 0.0, "b": 0.5, "c": 1.0}
        norm = db._normalize_scores(scores)
        assert norm["a"] == 0.0
        assert norm["c"] == 1.0
        assert 0 < norm["b"] < 1

    def test_normalize_equal_scores(self):
        db = _make_db()
        scores = {"a": 5.0, "b": 5.0, "c": 5.0}
        norm = db._normalize_scores(scores)
        assert all(v == 0.5 for v in norm.values())

    def test_normalize_empty(self):
        db = _make_db()
        assert db._normalize_scores({}) == {}

    def test_normalize_single(self):
        db = _make_db()
        norm = db._normalize_scores({"a": 42.0})
        assert norm["a"] == 0.5  # Single value maps to midpoint


# ===========================================================================
# Auto-categorize integration tests
# ===========================================================================
class TestAutoCategorizeIntegration:
    def test_store_auto_categorizes(self):
        db = _make_db(auto_categorize=True)
        doc_id = db.store("We decided to switch to Qdrant for production")
        mem = db.get(doc_id)
        assert mem is not None
        assert mem["metadata"]["category"] == "decision"

    def test_store_preserves_existing_category(self):
        db = _make_db(auto_categorize=True)
        doc_id = db.store("We decided to switch", metadata={"category": "custom"})
        mem = db.get(doc_id)
        assert mem["metadata"]["category"] == "custom"

    def test_store_auto_date(self):
        db = _make_db()
        doc_id = db.store("Test memory without date")
        mem = db.get(doc_id)
        assert "date" in mem["metadata"]
        assert len(mem["metadata"]["date"]) == 10  # YYYY-MM-DD

    def test_store_preserves_existing_date(self):
        db = _make_db()
        doc_id = db.store("Test", metadata={"date": "2020-01-01"})
        mem = db.get(doc_id)
        assert mem["metadata"]["date"] == "2020-01-01"

    def test_batch_store_auto_categorize(self):
        db = _make_db(auto_categorize=True)
        ids = db.batch_store([
            "Fix the encoding bug in BM25",
            "Learned that RRF is score-agnostic",
            "Question: what is the best vector DB?",
        ])
        for doc_id in ids:
            mem = db.get(doc_id)
            assert "category" in mem["metadata"]


# ===========================================================================
# Validation tests
# ===========================================================================
class TestValidation:
    def test_store_empty_text_raises(self):
        db = _make_db()
        with pytest.raises(ValueError, match="non-empty"):
            db.store("")

    def test_store_whitespace_text_raises(self):
        db = _make_db()
        with pytest.raises(ValueError, match="non-empty"):
            db.store("   \n\t  ")

    def test_store_with_vector_empty_text_raises(self):
        db = _make_db()
        with pytest.raises(ValueError, match="non-empty"):
            db.store_with_vector("", np.zeros(384))

    def test_batch_store_empty_list(self):
        db = _make_db()
        result = db.batch_store([])
        assert result == []


# ===========================================================================
# CHAOS TESTS — Phase 2
# ===========================================================================
class TestChaosUnicode:
    """Test with unusual Unicode inputs."""

    def test_emoji_in_text(self):
        db = _make_db()
        doc_id = db.store("Preferences: 🎩 dark mode, ☕ coffee, 🐍 Python 🚀")
        results = db.search("dark mode preferences", k=5)
        assert any(r.id == doc_id for r in results)

    def test_mixed_scripts(self):
        db = _make_db()
        db.store("Hello 世界 مرحبا Привет 🌍")
        results = db.search("hello world", k=5)
        assert len(results) >= 1

    def test_rare_unicode(self):
        db = _make_db()
        db.store("Test with special chars: ñ é í ó ú ü ß ø æ ð þ")
        results = db.search("special chars", k=5)
        assert len(results) >= 1

    def test_zero_width_chars(self):
        db = _make_db()
        text = "Test\u200Bzero\u200Cwidth\u200Dchars"
        doc_id = db.store(text)
        results = db.search("test", k=5)
        assert any(r.id == doc_id for r in results)

    def test_very_long_text(self):
        db = _make_db()
        long_text = "important keyword " * 500  # ~8000 chars
        doc_id = db.store(long_text)
        results = db.search("important keyword", k=5)
        assert any(r.id == doc_id for r in results)

    def test_single_char_text(self):
        db = _make_db()
        doc_id = db.store("X")
        results = db.search("X", k=5)
        assert any(r.id == doc_id for r in results)

    def test_numbers_only(self):
        db = _make_db()
        doc_id = db.store("42 is the answer to everything")
        results = db.search("42 answer", k=5)
        assert any(r.id == doc_id for r in results)


class TestChaosBM25EdgeCases:
    """Edge cases for BM25 index."""

    def test_bm25_empty_index_search(self):
        bm25 = BM25Index()
        results = bm25.search("anything", k=5)
        assert results == []

    def test_bm25_single_document(self):
        bm25 = BM25Index()
        bm25.add("doc1", "Python programming")
        results = bm25.search("Python", k=5)
        assert len(results) == 1
        assert results[0][0] == "doc1"

    def test_bm25_duplicate_documents(self):
        bm25 = BM25Index()
        bm25.add("doc1", "Python programming")
        bm25.add("doc2", "Python programming")
        bm25.add("doc3", "Python programming")
        results = bm25.search("Python", k=5)
        assert len(results) == 3
        # All should have the same score
        scores = [s for _, s in results]
        assert all(abs(s - scores[0]) < 1e-10 for s in scores)

    def test_bm25_repeated_terms(self):
        bm25 = BM25Index()
        bm25.add("doc1", "Python Python Python")
        bm25.add("doc2", "Python")
        results = bm25.search("Python", k=5)
        assert len(results) == 2
        # doc1 should rank higher (more occurrences)
        assert results[0][0] == "doc1"

    def test_bm25_no_matching_terms(self):
        bm25 = BM25Index()
        bm25.add("doc1", "Python programming")
        results = bm25.search("quantum physics", k=5)
        assert results == []

    def test_bm25_remove_and_search(self):
        bm25 = BM25Index()
        bm25.add("doc1", "Python programming")
        bm25.add("doc2", "Java enterprise")
        bm25.remove("doc1")
        results = bm25.search("Python", k=5)
        assert results == []

    def test_bm25_remove_nonexistent(self):
        bm25 = BM25Index()
        assert bm25.remove("nonexistent") is False

    def test_bm25_clear_and_search(self):
        bm25 = BM25Index()
        bm25.add("doc1", "Python programming")
        bm25.add("doc2", "Java enterprise")
        bm25.clear()
        results = bm25.search("Python", k=5)
        assert results == []

    def test_bm25_unicode_tokens(self):
        bm25 = BM25Index()
        bm25.add("doc1", "café résumé naïve")
        results = bm25.search("café", k=5)
        assert len(results) == 1

    def test_bm25_doc_filter(self):
        bm25 = BM25Index()
        bm25.add("doc1", "Python programming")
        bm25.add("doc2", "Python snake")
        bm25.add("doc3", "Python tutorial")
        results = bm25.search("Python", k=5, doc_filter={"doc1", "doc3"})
        assert len(results) == 2
        ids = [r[0] for r in results]
        assert "doc2" not in ids


class TestChaosMemoryDB:
    """Chaos tests for AlfredMemoryDB."""

    def test_store_and_search_many(self):
        """Store 500 memories and search."""
        db = _make_db()
        texts = [f"Memory number {i} about topic_{i % 10}" for i in range(500)]
        ids = db.batch_store(texts)
        assert len(ids) == 500

        results = db.search("topic_5", k=10)
        assert len(results) == 10
        assert all("topic_5" in r.document for r in results)

    def test_delete_all_and_search(self):
        """Delete everything and search returns empty."""
        db = _make_db()
        db.store("doc1")
        db.store("doc2")
        db.clear()
        results = db.search("doc", k=5)
        assert results == []

    def test_store_duplicate_ids(self):
        """Storing with same ID should update."""
        db = _make_db()
        doc_id = "fixed_id"
        db.store("original text", doc_id=doc_id)
        db.store("updated text", doc_id=doc_id)
        mem = db.get(doc_id)
        assert mem["document"] == "updated text"

    def test_stats_after_operations(self):
        """Stats should be consistent after operations."""
        db = _make_db()
        db.store("a")
        db.store("b")
        db.store("c")
        db.search("a", k=5)
        db.search("b", k=5)

        stats = db.stats()
        assert stats["total_adds"] == 3
        assert stats["total_queries"] == 2
        assert stats["bm25_indexed"] == 3

    def test_timestamps_cleanup_on_clear(self):
        """Timestamps should be cleaned on clear."""
        db = _make_db(temporal_decay=True)
        db.store("test")
        assert len(db._timestamps) == 1
        db.clear()
        assert len(db._timestamps) == 0

    def test_timestamps_cleanup_on_delete(self):
        """Timestamps should be cleaned on delete."""
        db = _make_db(temporal_decay=True)
        doc_id = db.store("test")
        assert doc_id in db._timestamps
        db.delete(id=doc_id)
        assert doc_id not in db._timestamps


class TestChaosConcurrent:
    """Basic concurrent safety tests."""

    def test_rapid_store_search_cycle(self):
        """Rapid store/search cycles shouldn't crash."""
        db = _make_db()
        for i in range(100):
            db.store(f"text_{i}")
            results = db.search(f"text_{i}", k=1)
            assert len(results) >= 1

    def test_interleaved_store_delete(self):
        """Interleave store and delete operations."""
        db = _make_db()
        stored = []
        for i in range(50):
            doc_id = db.store(f"keep_{i}")
            stored.append(doc_id)
            if i % 3 == 0 and i > 0:
                db.delete(id=stored.pop(0))

        results = db.search("keep", k=50)
        assert len(results) > 0


# ===========================================================================
# BM25 tokenizer tests
# ===========================================================================
class TestBM25Tokenizer:
    def test_spanish_chars(self):
        """BM25 should handle Spanish characters."""
        bm25 = BM25Index()
        bm25.add("doc1", "España México Argentina Chile Perú")
        results = bm25.search("españa", k=5)
        assert len(results) == 1

    def test_numbers_in_tokens(self):
        """BM25 should handle numbers."""
        bm25 = BM25Index()
        bm25.add("doc1", "Python 3.12 is the version")
        results = bm25.search("3.12", k=5)
        assert len(results) == 1
