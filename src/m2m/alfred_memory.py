"""
AlfredMemoryDB — Semantic Memory for Alfred 🎩

Convenience wrapper around M2M Vector Search designed specifically for
Alfred's personal memory use case.

Phase 2 Features (research-backed via Z.AI tools):
- Auto-embedding via sentence-transformers or custom encoder
- Hybrid search: BM25 (keyword) + Vector (semantic) with multiple fusion methods
- Fusion methods: RRF (default), Weighted Score, and Composable (vector/BM25 only)
- Temporal decay: recent memories rank higher (configurable half-life)
- Auto-categorization: infer category from text keywords
- Simple API: store(), search(), delete(), batch_store()
- Transparent persistence: save/load to disk
- SOC consolidation for memory cleanup
- Low resource: CPU-first, designed for ~1-10K memories

Research Sources (Z.AI web_search):
- RRF vs Weighted Score vs Cross-Encoder: RRF is industry standard (Elasticsearch,
  OpenSearch, Pinecone). Weighted Score needs normalization. Cross-Encoder for reranking.
- Embedding models: bge-small-en-v1.5 best all-rounder, gte-small best accuracy, 
  all-MiniLM-L6-v2 fastest but aging. All 384D.
- Compression: Scalar Quantization (int8) = 4x memory reduction with <1% recall loss.
  Binary Quantization = 32x reduction but needs oversampling+reranking.
- HNSW: best speed/accuracy, 30-50% memory overhead. IVF: good balance. PQ: memory only.
- Temporal decay: exponential decay based on age, configurable half-life.

Usage:
    >>> from m2m import AlfredMemoryDB
    >>> mem = AlfredMemoryDB(storage_path="./alfred_memory")
    >>> mem.store("Mr Schwabauer decided to use M2M for semantic memory",
    ...           metadata={"date": "2026-03-18", "category": "decision"})
    >>> results = mem.search("what did we decide about M2M?", k=5)
    >>> for r in results:
    ...     print(f"[{r.score:.3f}] {r.document[:80]}")
"""

from __future__ import annotations

import math
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .bm25_index import BM25Index
from .config import M2MConfig


# --- Category keywords for auto-categorization ---
_CATEGORY_KEYWORDS = {
    "decision": ["decided", "decisión", "chose", "elegido", "agreed", "acordado", "plan", "going to", "will use", "vamos a"],
    "preference": ["prefers", "prefiere", "likes", "gusta", "loves", "favorite", "favorito", "default", "always use"],
    "project": ["project", "proyecto", "implement", "implementar", "build", "construir", "feature", "featurea", "sprint", "milestone"],
    "error": ["error", "bug", "fallo", "failed", "falló", "crash", "exception", "broken", "roto", "fix", "arreglar"],
    "learning": ["learned", "aprendí", "lesson", "lección", "discovered", "descubrí", "realized", "me di cuenta", "insight"],
    "question": ["question", "pregunta", "how to", "cómo", "why", "por qué", "what is", "qué es", "wondering"],
    "conversation": ["said", "dijo", "told me", "me dijo", "discussed", "discutimos", "chat", "call", "llamada", "meeting"],
    "task": ["todo", "tarea", "task", "pending", "pendiente", "reminder", "recordar", "need to", "necesito"],
    "config": ["config", "configuración", "setup", "installed", "instalado", "settings", "ajustes", "environment"],
}


def auto_categorize(text: str) -> Optional[str]:
    """
    Infer a category from text using keyword matching.

    Based on research: simple keyword-based classification works well for
    personal memory systems where categories are few and distinct.

    Args:
        text: The text to categorize

    Returns:
        Category string or None if no match
    """
    text_lower = text.lower()
    best_cat = None
    best_count = 0
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > best_count:
            best_count = count
            best_cat = cat
    return best_cat if best_count > 0 else None


class MemoryResult:
    """A single memory search result."""

    __slots__ = ("id", "document", "metadata", "score", "vector_score", "bm25_score")

    def __init__(
        self,
        id: str,
        document: Optional[str] = None,
        metadata: Optional[Dict] = None,
        score: float = 0.0,
        vector_score: float = 0.0,
        bm25_score: float = 0.0,
    ):
        self.id = id
        self.document = document
        self.metadata = metadata or {}
        self.score = score
        self.vector_score = vector_score
        self.bm25_score = bm25_score

    def __repr__(self):
        doc_preview = (self.document[:60] + "...") if self.document and len(self.document) > 60 else self.document
        return f"MemoryResult(id={self.id!r}, score={self.score:.3f}, doc={doc_preview!r})"


class AlfredMemoryDB:
    """
    Semantic Memory Database for Alfred 🎩

    Combines vector search (M2M) with keyword search (BM25) via
    configurable fusion methods for optimal recall.

    Args:
        encoder: Callable that takes text (str or List[str]) and returns np.ndarray
                 of shape (D,) or (N, D). If None, vectors must be provided manually.
        latent_dim: Embedding dimension (default 384 for all-MiniLM-L6-v2)
        storage_path: Directory for persistence (None = in-memory only)
        hybrid_weight: Weight for vector vs BM25 in weighted fusion (0.0=bm25 only, 1.0=vector only, default 0.6)
        fusion_method: "rrf" (Reciprocal Rank Fusion, default), "weighted" (linear combination), 
                       "vector_only", or "bm25_only"
        bm25_k1: BM25 k1 parameter (default 1.5)
        bm25_b: BM25 b parameter (default 0.75)
        temporal_decay: Enable temporal decay (recent memories rank higher). Default False.
        temporal_half_life_days: Half-life for temporal decay in days. Default 30.
                                After this many days, a memory's time-boost halves.
        auto_categorize: Automatically infer category from text. Default False.
        device: Compute device for M2M (default "cpu")
        mode: M2M mode ("edge", "standard", "ebm")

    Example:
        >>> # With auto-encoder and temporal decay
        >>> mem = AlfredMemoryDB(storage_path="./mem", encoder=my_encoder,
        ...                      temporal_decay=True, auto_categorize=True)
        >>> mem.store("User prefers dark mode", metadata={"category": "preference"})
        >>> results = mem.search("dark mode preference", k=3)
    """

    def __init__(
        self,
        encoder: Optional[Callable] = None,
        latent_dim: int = 384,
        storage_path: Optional[str] = None,
        hybrid_weight: float = 0.6,
        fusion_method: str = "rrf",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        temporal_decay: bool = False,
        temporal_half_life_days: float = 30.0,
        auto_categorize: bool = False,
        device: str = "cpu",
        mode: str = "standard",
    ):
        if fusion_method not in ("rrf", "weighted", "vector_only", "bm25_only"):
            raise ValueError(f"fusion_method must be one of: rrf, weighted, vector_only, bm25_only. Got: {fusion_method}")

        self.encoder = encoder
        self.latent_dim = latent_dim
        self.storage_path = storage_path
        self.hybrid_weight = hybrid_weight
        self.fusion_method = fusion_method
        self.temporal_decay = temporal_decay
        self.temporal_half_life_days = temporal_half_life_days
        self.auto_categorize = auto_categorize

        # Import SimpleVectorDB from package
        from m2m import SimpleVectorDB

        # M2M Vector DB (disable LSH fallback — Alfred's memories are diverse)
        self._db = SimpleVectorDB(
            device=device,
            latent_dim=latent_dim,
            storage_path=storage_path,
            enable_lsh_fallback=False,
            mode=mode,
        )

        # BM25 Keyword Index
        self._bm25 = BM25Index(k1=bm25_k1, b=bm25_b)

        # Stats tracking
        self._query_count = 0
        self._query_latencies: List[float] = []
        self._add_count = 0

        # Store timestamps for temporal decay
        self._timestamps: Dict[str, float] = {}  # doc_id -> unix timestamp

    def _encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """Encode text to vector. Uses encoder if available, raises otherwise."""
        if self.encoder is None:
            raise RuntimeError(
                "No encoder set. Provide an encoder callable to AlfredMemoryDB, "
                "or use store_with_vector() to provide pre-computed vectors."
            )
        result = self.encoder(text)
        arr = np.asarray(result, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return arr

    def _compute_temporal_boost(self, doc_id: str) -> float:
        """
        Compute temporal decay boost for a document.

        Uses exponential decay: boost = exp(-λ * age_in_days)
        where λ = ln(2) / half_life_days

        A document added today gets boost=1.0.
        A document added half_life_days ago gets boost=0.5.
        A document added 2*half_life_days ago gets boost=0.25.

        Returns:
            Boost factor in (0.0, 1.0]
        """
        if not self.temporal_decay:
            return 1.0

        ts = self._timestamps.get(doc_id)
        if ts is None:
            return 1.0

        age_seconds = time.time() - ts
        age_days = age_seconds / 86400.0

        if age_days <= 0:
            return 1.0

        lam = math.log(2) / self.temporal_half_life_days
        boost = math.exp(-lam * age_days)

        # Clamp to avoid floating point issues
        return max(boost, 0.01)

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Min-Max normalize scores to [0, 1].

        Used for weighted score fusion where vector and BM25 scores
        have different scales (research: critical requirement).
        """
        if not scores:
            return {}
        values = list(scores.values())
        min_s = min(values)
        max_s = max(values)
        rng = max_s - min_s
        if rng < 1e-10:
            # All scores are equal
            return {k: 0.5 for k in scores}
        return {k: (v - min_s) / rng for k, v in scores.items()}

    def store(
        self,
        text: str,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Store a memory (text + metadata) with automatic embedding.

        Args:
            text: The text content to remember
            metadata: Optional dict of metadata (date, category, source, etc.)
            doc_id: Optional ID. Auto-generated if not provided.

        Returns:
            The document ID
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        if doc_id is None:
            import uuid
            doc_id = str(uuid.uuid4())

        meta = dict(metadata) if metadata else {}

        # Auto-categorize if enabled and no category set
        if self.auto_categorize and "category" not in meta:
            cat = auto_categorize(text)
            if cat:
                meta["category"] = cat

        # Auto-set date if not provided
        if "date" not in meta:
            meta["date"] = datetime.now().strftime("%Y-%m-%d")

        embedding = self._encode(text)  # (1, D)
        self._db.add(
            ids=[doc_id],
            vectors=embedding,
            metadata=[meta],
            documents=[text],
        )
        self._bm25.add(doc_id, text)
        self._timestamps[doc_id] = time.time()
        self._add_count += 1
        return doc_id

    def store_with_vector(
        self,
        text: str,
        vector: np.ndarray,
        metadata: Optional[Dict] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Store a memory with a pre-computed vector (no encoder needed).

        Args:
            text: The text content
            vector: Pre-computed embedding (D,) or (1, D)
            metadata: Optional metadata dict
            doc_id: Optional ID

        Returns:
            The document ID
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be a non-empty string")

        if doc_id is None:
            import uuid
            doc_id = str(uuid.uuid4())

        meta = dict(metadata) if metadata else {}

        if self.auto_categorize and "category" not in meta:
            cat = auto_categorize(text)
            if cat:
                meta["category"] = cat

        if "date" not in meta:
            meta["date"] = datetime.now().strftime("%Y-%m-%d")

        vec = np.asarray(vector, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec[np.newaxis, :]

        self._db.add(ids=[doc_id], vectors=vec, metadata=[meta], documents=[text])
        self._bm25.add(doc_id, text)
        self._timestamps[doc_id] = time.time()
        self._add_count += 1
        return doc_id

    def batch_store(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Store multiple memories at once.

        Args:
            texts: List of text contents
            metadatas: Optional list of metadata dicts (one per text)
            ids: Optional list of IDs

        Returns:
            List of document IDs
        """
        if not texts:
            return []

        n = len(texts)
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(n)]
        if metadatas is None:
            metadatas = [{}] * n

        # Auto-enrich metadatas
        now = datetime.now().strftime("%Y-%m-%d")
        for i in range(n):
            if self.auto_categorize and "category" not in metadatas[i]:
                cat = auto_categorize(texts[i])
                if cat:
                    metadatas[i] = dict(metadatas[i])
                    metadatas[i]["category"] = cat
            if "date" not in metadatas[i]:
                metadatas[i] = dict(metadatas[i])
                metadatas[i]["date"] = now

        embeddings = self._encode(texts)  # (N, D)
        self._db.add(ids=ids, vectors=embeddings, metadata=metadatas, documents=texts)

        store_time = time.time()
        for i, text in enumerate(texts):
            self._bm25.add(ids[i], text)
            self._timestamps[ids[i]] = store_time

        self._add_count += n
        return ids

    def search(
        self,
        query: Union[str, np.ndarray],
        k: int = 10,
        filter: Optional[Dict] = None,
        hybrid: bool = True,
    ) -> List[MemoryResult]:
        """
        Search memories using hybrid or vector-only search.

        Fusion method is determined by self.fusion_method:
        - "rrf": Reciprocal Rank Fusion (score-agnostic, no tuning needed)
        - "weighted": Linear combination with min-max normalization
        - "vector_only": Only vector search
        - "bm25_only": Only BM25 keyword search

        If temporal_decay is enabled, recent memories receive a boost.

        Args:
            query: Text query (auto-embedded) or pre-computed vector
            k: Number of results
            filter: Metadata filter dict
            hybrid: If True, use configured fusion. If False, vector only.

        Returns:
            List of MemoryResult sorted by fused score
        """
        t0 = time.perf_counter()

        # Determine effective fusion method
        if not hybrid or self.fusion_method == "vector_only":
            effective_fusion = "vector_only"
        elif self.fusion_method == "bm25_only":
            effective_fusion = "bm25_only"
        else:
            effective_fusion = self.fusion_method

        # --- Vector search ---
        vector_results = []
        vector_score_map: Dict[str, float] = {}  # id -> raw score

        if effective_fusion != "bm25_only":
            if isinstance(query, str):
                query_vec = self._encode(query)
            else:
                query_vec = np.asarray(query, dtype=np.float32)
                if query_vec.ndim == 1:
                    query_vec = query_vec[np.newaxis, :]

            vector_results = self._db.search(
                query_vec.squeeze(), k=k * 3, include_metadata=True, filter=filter
            )
            if isinstance(vector_results, tuple):
                vector_results = []

            for r in vector_results:
                score = getattr(r, 'score', 0.0) or 0.0
                vector_score_map[r.id] = float(score)

        # --- BM25 search ---
        bm25_score_map: Dict[str, float] = {}

        if effective_fusion != "vector_only" and isinstance(query, str):
            doc_filter = None
            if filter and vector_score_map:
                doc_filter = set(vector_score_map.keys())
            bm25_raw = self._bm25.search(query, k=k * 3, doc_filter=doc_filter)
            for doc_id, score in bm25_raw:
                bm25_score_map[doc_id] = float(score)

        # --- Fusion ---
        fused_scores: Dict[str, float] = {}
        all_ids = set(vector_score_map.keys()) | set(bm25_score_map.keys())

        if effective_fusion == "vector_only":
            fused_scores = dict(vector_score_map)

        elif effective_fusion == "bm25_only":
            fused_scores = dict(bm25_score_map)

        elif effective_fusion == "rrf":
            # Reciprocal Rank Fusion (industry standard, used by ES/OpenSearch/Pinecone)
            # Research: score-agnostic, no normalization needed, works with default k=60
            rrf_k = 60
            v_ranked = sorted(vector_score_map.items(), key=lambda x: x[1], reverse=True)
            b_ranked = sorted(bm25_score_map.items(), key=lambda x: x[1], reverse=True)

            v_rank_map = {doc_id: rank for rank, (doc_id, _) in enumerate(v_ranked)}
            b_rank_map = {doc_id: rank for rank, (doc_id, _) in enumerate(b_ranked)}

            for doc_id in all_ids:
                v_rank = v_rank_map.get(doc_id, len(v_ranked) + 1)
                b_rank = b_rank_map.get(doc_id, len(b_ranked) + 1)

                v_rrf = self.hybrid_weight / (rrf_k + v_rank + 1)
                b_rrf = (1 - self.hybrid_weight) / (rrf_k + b_rank + 1)
                fused_scores[doc_id] = v_rrf + b_rrf

        elif effective_fusion == "weighted":
            # Weighted Score Fusion with min-max normalization
            # Research: must normalize! BM25 can be 0-100+, cosine similarity is 0-1
            norm_v = self._normalize_scores(vector_score_map)
            norm_b = self._normalize_scores(bm25_score_map)

            for doc_id in all_ids:
                v = norm_v.get(doc_id, 0.0)
                b = norm_b.get(doc_id, 0.0)
                fused_scores[doc_id] = self.hybrid_weight * v + (1 - self.hybrid_weight) * b

        # --- Temporal decay boost ---
        if self.temporal_decay:
            for doc_id in fused_scores:
                boost = self._compute_temporal_boost(doc_id)
                fused_scores[doc_id] *= (0.5 + 0.5 * boost)  # Range [0.5, 1.0]

        # --- Build results ---
        ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # Build lookup for document text/metadata from vector results
        vec_lookup = {}
        for r in vector_results:
            vec_lookup[r.id] = {
                "document": getattr(r, 'document', None),
                "metadata": getattr(r, 'metadata', {}) or {},
            }

        results = []
        for doc_id, fused_score in ranked:
            v_score = vector_score_map.get(doc_id, 0.0)
            b_score = bm25_score_map.get(doc_id, 0.0)

            doc_text = None
            meta = {}

            if doc_id in vec_lookup:
                doc_text = vec_lookup[doc_id]["document"]
                meta = vec_lookup[doc_id]["metadata"]

            # Fallback: get from BM25
            if not doc_text and doc_id in self._bm25._docs:
                doc_text = self._bm25._docs[doc_id]

            results.append(MemoryResult(
                id=doc_id,
                document=doc_text,
                metadata=meta,
                score=fused_score,
                vector_score=v_score,
                bm25_score=b_score,
            ))

        # Track stats
        self._query_count += 1
        self._query_latencies.append((time.perf_counter() - t0) * 1000)
        if len(self._query_latencies) > 1000:
            self._query_latencies = self._query_latencies[-1000:]

        return results

    def delete(
        self,
        id: Optional[str] = None,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict] = None,
    ) -> int:
        """
        Delete memories by ID or filter.

        Args:
            id: Single document ID
            ids: List of document IDs
            filter: Metadata filter dict

        Returns:
            Number of documents deleted
        """
        result = self._db.delete(id=id, ids=ids, filter=filter, hard=True)
        n_deleted = result.deleted if hasattr(result, 'deleted') else 0

        # Remove from BM25 and timestamps
        to_remove = set()
        if id:
            to_remove.add(id)
        if ids:
            to_remove.update(ids)
        for doc_id in to_remove:
            self._bm25.remove(doc_id)
            self._timestamps.pop(doc_id, None)

        return n_deleted

    def save(self) -> str:
        """
        Persist current state to disk.

        Returns:
            Path where data was saved
        """
        if self.storage_path is None:
            raise RuntimeError("No storage_path configured. Set storage_path in constructor.")
        self._db.save(self.storage_path)
        return self.storage_path

    def consolidate(self, threshold: float = 0.8) -> int:
        """
        Run SOC consolidation to merge/remove low-quality memories.

        Args:
            threshold: Alpha threshold for consolidation (lower = more aggressive)

        Returns:
            Number of splats consolidated
        """
        if isinstance(self._db, type(self._db)):
            return self._db.engine.m2m.consolidate(threshold)
        return 0

    def stats(self) -> Dict[str, Any]:
        """
        Get memory system statistics.

        Returns:
            Dict with counts, latencies, and system health info
        """
        db_stats = self._db.get_stats()
        query_stats = {}
        if self._query_latencies:
            lat = np.array(self._query_latencies)
            query_stats = {
                "total_queries": self._query_count,
                "avg_latency_ms": round(float(np.mean(lat)), 2),
                "p50_latency_ms": round(float(np.percentile(lat, 50)), 2),
                "p95_latency_ms": round(float(np.percentile(lat, 95)), 2),
                "p99_latency_ms": round(float(np.percentile(lat, 99)), 2),
                "last_100_avg_ms": round(float(np.mean(lat[-100:])), 2) if len(lat) >= 100 else round(float(np.mean(lat)), 2),
            }

        return {
            "total_memories": db_stats.get("active_documents", 0),
            "total_stored": db_stats.get("total_documents", 0),
            "total_queries": self._query_count,
            "total_adds": self._add_count,
            "bm25_indexed": self._bm25.n_docs,
            "hybrid_weight": self.hybrid_weight,
            "fusion_method": self.fusion_method,
            "temporal_decay": self.temporal_decay,
            "temporal_half_life_days": self.temporal_half_life_days,
            "auto_categorize": self.auto_categorize,
            "storage_path": self.storage_path,
            "query_stats": query_stats,
            "db_stats": db_stats,
        }

    def get(self, doc_id: str) -> Optional[Dict]:
        """
        Retrieve a specific memory by ID.

        Args:
            doc_id: Document ID

        Returns:
            Dict with 'document', 'metadata', or None if not found
        """
        vec = self._db._vectors.get(doc_id)
        if vec is not None and doc_id not in self._db._deleted:
            return {
                "id": doc_id,
                "document": self._db._documents.get(doc_id),
                "metadata": self._db._metadata.get(doc_id, {}),
            }
        return None

    def clear(self):
        """Clear all memories (irreversible)."""
        self._db.delete(filter={"_all": True}, hard=True)
        self._bm25.clear()
        self._timestamps.clear()
        # Reset in-memory storage
        self._db._vectors.clear()
        self._db._metadata.clear()
        self._db._documents.clear()
        self._db._deleted.clear()
