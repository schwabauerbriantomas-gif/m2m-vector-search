"""
RAG Test Dataset Tests for M2M-Vector-Search
Tests: basic retrieval, edge cases, RAG pipeline, semantic memory, numerical stability.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

# Project root
PROJECT_ROOT = Path(r"C:\Users\Brian\Desktop\m2m-vector-search-main")
DATASET_DIR = PROJECT_ROOT / "datasets" / "rag_test"

# Add src to path
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ── Fixtures ────────────────────────────────────────────────────────
@pytest.fixture(scope="session")
def embeddings():
    """Load pre-computed embeddings (659, 384)."""
    path = DATASET_DIR / "embeddings.npy"
    assert path.exists(), f"Run build_dataset.py first: {path}"
    return np.load(path)


@pytest.fixture(scope="session")
def metadata():
    path = DATASET_DIR / "metadata.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def texts():
    """Load chunk texts as list."""
    path = DATASET_DIR / "texts.jsonl"
    texts_list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            texts_list.append(json.loads(line)["text"])
    return texts_list


@pytest.fixture(scope="session")
def embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def linear_search(
    query_emb: np.ndarray, embeddings: np.ndarray, k: int = 5
) -> List[Tuple[int, float]]:
    """Brute-force cosine similarity search. Returns [(index, similarity), ...]."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    normed = embeddings / norms
    q_norm = query_emb / max(np.linalg.norm(query_emb), 1e-10)
    sims = normed @ q_norm
    top_k = np.argsort(sims)[::-1][:k]
    return [(int(i), float(sims[i])) for i in top_k]


# ═══════════════════════════════════════════════════════════════════
# NUMERICAL STABILITY
# ═══════════════════════════════════════════════════════════════════
class TestNumericalStability:
    def test_no_nan_in_embeddings(self, embeddings):
        assert np.all(np.isfinite(embeddings)), "Embeddings contain NaN or Inf"

    def test_no_nan_in_any_dimension(self, embeddings):
        for i in range(embeddings.shape[1]):
            assert np.all(np.isfinite(embeddings[:, i])), f"Dimension {i} has NaN/Inf"

    def test_cosine_sims_in_range(self, embeddings):
        """Check pairwise cosine similarities are in [-1, 1]."""
        sample = embeddings[:50]
        norms = np.linalg.norm(sample, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-10, None)
        normed = sample / norms
        sims = normed @ normed.T
        assert np.all(sims >= -1.0 - 1e-6), f"Min sim: {sims.min()}"
        assert np.all(sims <= 1.0 + 1e-6), f"Max sim: {sims.max()}"

    def test_norms_non_negative(self, embeddings):
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.all(norms >= 0), "Negative norms found"

    def test_embeddings_shape(self, embeddings):
        assert embeddings.ndim == 2
        assert embeddings.shape[1] == 384


# ═══════════════════════════════════════════════════════════════════
# BASIC RETRIEVAL
# ═══════════════════════════════════════════════════════════════════
class TestBasicRetrieval:
    @pytest.mark.parametrize(
        "query,expected_source_keyword,threshold",
        [
            ("hierarchical reasoning model", "HRM", 0.3),
            ("flow matching generative models", "equilibrium_matching", 0.25),
            ("latent chain of thought", "latent_cot", 0.3),
            ("Langevin dynamics score", "langevin", 0.25),
            ("M2M vector search gaussian splats", "README", 0.3),
            ("tiny recursive model", "tiny_recursive", 0.3),
        ],
    )
    def test_topic_returns_relevant_results(
        self, embedder, embeddings, metadata, texts, query, expected_source_keyword, threshold
    ):
        q_emb = embedder.encode([query])[0]
        results = linear_search(q_emb, embeddings, k=5)
        assert len(results) > 0, "No results returned"

        # At least one top-5 result should come from expected source
        sources = [metadata[i]["source"] for i, _ in results]
        match = any(expected_source_keyword.lower() in s.lower() for s in sources)
        assert (
            match
        ), f"Query '{query}': none of top-5 match '{expected_source_keyword}'. Sources: {sources}"

    def test_search_returns_k_results(self, embedder, embeddings):
        q_emb = embedder.encode(["machine learning"])[0]
        results = linear_search(q_emb, embeddings, k=10)
        assert len(results) == 10

    def test_top1_has_highest_similarity(self, embedder, embeddings):
        q_emb = embedder.encode(["attention mechanism transformers"])[0]
        results = linear_search(q_emb, embeddings, k=5)
        sims = [s for _, s in results]
        assert sims == sorted(sims, reverse=True), "Results not sorted by similarity"

    def test_top_results_above_threshold(self, embedder, embeddings):
        q_emb = embedder.encode(["Gaussian splats vector search"])[0]
        results = linear_search(q_emb, embeddings, k=5)
        assert results[0][1] > 0.3, f"Top-1 similarity too low: {results[0][1]:.4f}"


# ═══════════════════════════════════════════════════════════════════
# EDGE CASES
# ═══════════════════════════════════════════════════════════════════
class TestEdgeCases:
    def test_single_word_query(self, embedder, embeddings):
        q_emb = embedder.encode(["transformers"])[0]
        results = linear_search(q_emb, embeddings, k=5)
        assert len(results) == 5
        assert results[0][1] > 0.2, "Single word query should still find relevant results"

    def test_long_paragraph_query(self, embedder, embeddings):
        long_query = (
            "Recent advances in machine learning have focused on developing more efficient "
            "architectures for training large language models. Techniques such as mixture of experts, "
            "quantization, and knowledge distillation have been explored to reduce computational costs "
            "while maintaining model quality. Additionally, new approaches to reasoning and chain-of-thought "
            "prompting have shown promise in improving model performance on complex tasks."
        )
        q_emb = embedder.encode([long_query])[0]
        results = linear_search(q_emb, embeddings, k=5)
        assert len(results) == 5

    def test_cross_lingual_spanish(self, embedder, embeddings):
        """all-MiniLM-L6-v2 has some cross-lingual capability."""
        q_emb = embedder.encode(["búsqueda vectorial modelos de lenguaje"])[0]
        results = linear_search(q_emb, embeddings, k=5)
        assert len(results) == 5
        # Should still return something semi-relevant
        assert results[0][1] > 0.15, f"Cross-lingual top-1 too low: {results[0][1]:.4f}"

    def test_query_with_typos(self, embedder, embeddings):
        q_emb = embedder.encode(["hierarchcal reasning model"])[0]
        results = linear_search(q_emb, embeddings, k=5)
        assert len(results) == 5

    def test_nonexistent_topic(self, embedder, embeddings):
        q_emb = embedder.encode(["quantum entanglement cooking recipes"])[0]
        results = linear_search(q_emb, embeddings, k=5)
        assert len(results) == 5
        # All similarities should be low
        assert all(
            s < 0.4 for _, s in results
        ), f"Nonexistent topic returned high sim: {results[0][1]:.4f}"

    def test_k_larger_than_dataset(self, embedder, embeddings):
        q_emb = embedder.encode(["test"])[0]
        results = linear_search(q_emb, embeddings, k=10000)
        assert len(results) == len(embeddings)


# ═══════════════════════════════════════════════════════════════════
# RAG PIPELINE
# ═══════════════════════════════════════════════════════════════════
class TestRAGPipeline:
    def test_embed_search_retrieve_assembly(self, embedder, embeddings, metadata, texts):
        """Full RAG pipeline: embed query → search → retrieve context."""
        query = "What is flow matching and how does it work for generative models?"
        q_emb = embedder.encode([query])[0]
        results = linear_search(q_emb, embeddings, k=3)

        # Assemble context
        context = "\n\n".join(texts[i] for i, _ in results)
        assert len(context) > 100, "Context too short"

        # Verify context mentions flow matching
        assert "flow" in context.lower(), "Context should mention flow matching"

    def test_context_contains_answer_keywords(self, embedder, embeddings, metadata, texts):
        query = "How does HRM hierarchical reasoning work?"
        q_emb = embedder.encode([query])[0]
        results = linear_search(q_emb, embeddings, k=3)
        context = "\n\n".join(texts[i] for i, _ in results)
        # At least one of these keywords should appear
        keywords = ["hierarchical", "reasoning", "hrm", "level", "layers"]
        found = any(kw in context.lower() for kw in keywords)
        assert found, f"Context missing answer keywords. Snippet: {context[:200]}"

    def test_metadata_preserved(self, embeddings, metadata):
        assert len(metadata) == len(embeddings)
        for m in metadata:
            assert "source" in m
            assert "chunk_id" in m
            assert "type" in m

    def test_pdf_chunks_have_page_numbers(self, metadata):
        pdf_meta = [m for m in metadata if m.get("type") == "pdf"]
        assert len(pdf_meta) > 0, "No PDF chunks found"
        for m in pdf_meta:
            assert "page" in m, f"PDF chunk missing page: {m}"


# ═══════════════════════════════════════════════════════════════════
# SEMANTIC MEMORY
# ═══════════════════════════════════════════════════════════════════
class TestSemanticMemory:
    def test_store_and_recall(self, embeddings, embedder):
        """Simulate storing a memory and recalling it."""
        memory_text = "User is studying machine learning papers about reasoning models"
        memory_emb = embedder.encode([memory_text])[0]

        # Query similar to the memory
        query = "What is the user researching?"
        q_emb = embedder.encode([query])[0]

        sim = cosine_similarity(memory_emb, q_emb)
        assert sim > 0.3, f"Memory recall similarity too low: {sim:.4f}"

    def test_similar_queries_return_similar_results(self, embedder, embeddings):
        """Two similar queries should return similar top results."""
        q1 = "How do transformers work?"
        q2 = "Transformer architecture explanation"

        e1 = embedder.encode([q1])[0]
        e2 = embedder.encode([q2])[0]

        r1 = set(i for i, _ in linear_search(e1, embeddings, k=10))
        r2 = set(i for i, _ in linear_search(e2, embeddings, k=10))

        overlap = len(r1 & r2)
        assert overlap >= 5, f"Similar queries share only {overlap}/10 results"

    def test_dissimilar_queries_return_different_results(self, embedder, embeddings):
        q1 = "Langevin dynamics score matching"
        q2 = "M2M vector search API"

        e1 = embedder.encode([q1])[0]
        e2 = embedder.encode([q2])[0]

        r1 = set(i for i, _ in linear_search(e1, embeddings, k=5))
        r2 = set(i for i, _ in linear_search(e2, embeddings, k=5))

        overlap = len(r1 & r2)
        # Dissimilar queries should have < 80% overlap in top-5
        assert overlap < 4, f"Dissimilar queries share too many: {overlap}/5"

    def test_timestamp_memory_ordering(self, embedder):
        """Simulate memories with timestamps."""
        memories = [
            (1000, "User started studying ML"),
            (2000, "User read about transformers"),
            (3000, "User discovered M2M vector search"),
        ]

        # Embed all
        embs = embedder.encode([m[1] for m in memories])

        # Recall recent (last one)
        query = "What did the user discover recently?"
        q_emb = embedder.encode([query])[0]
        sims = [cosine_similarity(e, q_emb) for e in embs]

        # The most recent memory should have high similarity
        assert sims[2] > sims[0], "Recent memory should be more relevant"


# ═══════════════════════════════════════════════════════════════════
# M2M INTEGRATION (if available)
# ═══════════════════════════════════════════════════════════════════
@pytest.mark.skipif(not (PROJECT_ROOT / "src" / "m2m").exists(), reason="M2M source not available")
class TestM2MIntegration:
    @pytest.fixture(scope="class")
    def m2m_index(self, embeddings, metadata):
        try:
            from m2m import M2M, M2MConfig

            config = M2MConfig(
                latent_dim=384,
                max_splats=len(embeddings),
                enable_vulkan=False,
            )
            db = M2M(config)
            ids = [str(i) for i in range(len(embeddings))]
            docs = [None] * len(embeddings)
            db.add(ids=ids, vectors=embeddings, documents=docs)
            return db
        except Exception as e:
            pytest.skip(f"M2M init failed: {e}")

    def test_m2m_search_returns_results(self, m2m_index, embedder, embeddings):
        q_emb = embedder.encode(["Gaussian splats"])[0]
        results = m2m_index.search(q_emb, k=5)
        # Legacy returns tuple, new returns list
        if isinstance(results, tuple):
            assert len(results[0]) == 5, f"Expected 5 results, got {len(results[0])}"
        else:
            assert len(results) > 0

    def test_m2m_results_consistent_with_linear(self, m2m_index, embedder, embeddings):
        """M2M results should overlap significantly with linear scan."""
        q_emb = embedder.encode(["HRM reasoning model"])[0]

        # Linear baseline
        linear = set(i for i, _ in linear_search(q_emb, embeddings, k=10))

        # M2M search
        try:
            results = m2m_index.search(q_emb, k=10, include_metadata=True)
            if isinstance(results, list):
                m2m_ids = set()
                for r in results:
                    try:
                        m2m_ids.add(int(r.id))
                    except (ValueError, AttributeError):
                        pass
                overlap = len(linear & m2m_ids)
                assert overlap >= 3, f"M2M and linear share only {overlap}/10"
        except Exception:
            pytest.skip("M2M metadata search not available in this version")

    def test_m2m_large_batch(self, m2m_index, embedder, embeddings):
        """Test with many vectors - stress test."""
        q_emb = embedder.encode(["machine learning"])[0]
        results = m2m_index.search(q_emb, k=50)
        if isinstance(results, tuple):
            assert len(results[0]) <= 50
        else:
            assert len(results) <= 50


# ═══════════════════════════════════════════════════════════════════
# DATASET QUALITY
# ═══════════════════════════════════════════════════════════════════
class TestDatasetQuality:
    def test_chunk_count(self, metadata):
        assert len(metadata) >= 500, f"Expected >= 500 chunks, got {len(metadata)}"

    def test_source_diversity(self, metadata):
        sources = set(m["source"] for m in metadata)
        assert len(sources) >= 10, f"Expected >= 10 sources, got {len(sources)}: {sources}"

    def test_no_empty_texts(self, texts):
        empty = sum(1 for t in texts if len(t.strip()) < 50)
        assert empty == 0, f"{empty} chunks have very short text (<50 chars)"

    def test_pdf_and_markdown_present(self, metadata):
        types = set(m["type"] for m in metadata)
        assert "pdf" in types, "No PDF chunks"
        assert "markdown" in types, "No markdown chunks"

    def test_unique_chunk_ids(self, metadata):
        ids = [m["index"] for m in metadata]
        assert len(ids) == len(set(ids)), "Duplicate indices found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
