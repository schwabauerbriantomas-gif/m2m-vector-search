"""
AlfredMemoryDB — Semantic Memory for Alfred 🎩

Convenience wrapper around M2M Vector Search designed specifically for
Alfred's personal memory use case.

Features:
- Auto-embedding via sentence-transformers or custom encoder
- Hybrid search: BM25 (keyword) + Vector (semantic) with RRF fusion
- Simple API: store(), search(), delete(), batch_store()
- Transparent persistence: save/load to disk
- SOC consolidation for memory cleanup
- Low resource: CPU-first, designed for ~1-10K memories

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

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from .bm25_index import BM25Index
from .config import M2MConfig


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
    Reciprocal Rank Fusion for optimal recall.

    Args:
        encoder: Callable that takes text (str or List[str]) and returns np.ndarray
                 of shape (D,) or (N, D). If None, vectors must be provided manually.
        latent_dim: Embedding dimension (default 384 for all-MiniLM-L6-v2)
        storage_path: Directory for persistence (None = in-memory only)
        hybrid_weight: Weight for vector vs BM25 in hybrid search (0.0=bm25 only, 1.0=vector only, default 0.6)
        bm25_k1: BM25 k1 parameter (default 1.5)
        bm25_b: BM25 b parameter (default 0.75)
        device: Compute device for M2M (default "cpu")
        mode: M2M mode ("edge", "standard", "ebm")

    Example:
        >>> # With auto-encoder
        >>> mem = AlfredMemoryDB(storage_path="./mem", encoder=my_encoder)
        >>> mem.store("User prefers dark mode", metadata={"category": "preference"})
        >>> results = mem.search("dark mode preference", k=3)
    """

    def __init__(
        self,
        encoder: Optional[Callable] = None,
        latent_dim: int = 384,
        storage_path: Optional[str] = None,
        hybrid_weight: float = 0.6,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        device: str = "cpu",
        mode: str = "standard",
    ):
        self.encoder = encoder
        self.latent_dim = latent_dim
        self.storage_path = storage_path
        self.hybrid_weight = hybrid_weight

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
        if doc_id is None:
            import uuid
            doc_id = str(uuid.uuid4())

        embedding = self._encode(text)  # (1, D)
        self._db.add(
            ids=[doc_id],
            vectors=embedding,
            metadata=[metadata or {}],
            documents=[text],
        )
        self._bm25.add(doc_id, text)
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
        if doc_id is None:
            import uuid
            doc_id = str(uuid.uuid4())

        vec = np.asarray(vector, dtype=np.float32)
        if vec.ndim == 1:
            vec = vec[np.newaxis, :]

        self._db.add(ids=[doc_id], vectors=vec, metadata=[metadata or {}], documents=[text])
        self._bm25.add(doc_id, text)
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
        n = len(texts)
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(n)]
        if metadatas is None:
            metadatas = [{}] * n

        embeddings = self._encode(texts)  # (N, D)
        self._db.add(ids=ids, vectors=embeddings, metadata=metadatas, documents=texts)

        for i, text in enumerate(texts):
            self._bm25.add(ids[i], text)

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
        Search memories using hybrid (vector + BM25) or vector-only search.

        Args:
            query: Text query (auto-embedded) or pre-computed vector
            k: Number of results
            filter: Metadata filter dict
            hybrid: If True, combine vector + BM25. If False, vector only.

        Returns:
            List of MemoryResult sorted by fused score
        """
        t0 = time.perf_counter()

        # Get vector results
        if isinstance(query, str):
            query_vec = self._encode(query)  # (1, D)
        else:
            query_vec = np.asarray(query, dtype=np.float32)
            if query_vec.ndim == 1:
                query_vec = query_vec[np.newaxis, :]

        vector_results = self._db.search(
            query_vec.squeeze(), k=k * 3, include_metadata=True, filter=filter
        )
        # Handle legacy tuple return
        if isinstance(vector_results, tuple):
            vector_results = []

        # Build vector score map: id -> (rank, score)
        vector_scores: Dict[str, Tuple[int, float]] = {}
        for i, r in enumerate(vector_results):
            score = getattr(r, 'score', 0.0) or 0.0
            vector_scores[r.id] = (i, float(score))

        # Get BM25 results (only if query is text)
        bm25_scores: Dict[str, Tuple[int, float]] = {}
        if hybrid and isinstance(query, str):
            # Apply filter if needed
            doc_filter = None
            if filter:
                # Get active IDs that match filter
                active = set(vector_scores.keys())
                doc_filter = active
            bm25_raw = self._bm25.search(query, k=k * 3, doc_filter=doc_filter)
            for i, (doc_id, score) in enumerate(bm25_raw):
                bm25_scores[doc_id] = (i, score)

        # Reciprocal Rank Fusion
        all_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        rrf_scores: Dict[str, float] = {}

        rrf_k = 60  # RRF constant
        for doc_id in all_ids:
            v_rank, v_score = vector_scores.get(doc_id, (len(vector_scores) + 1, 0.0))
            b_rank, b_score = bm25_scores.get(doc_id, (len(bm25_scores) + 1, 0.0))

            v_rrf = self.hybrid_weight / (rrf_k + v_rank + 1)
            b_rrf = (1 - self.hybrid_weight) / (rrf_k + b_rank + 1) if hybrid else 0

            rrf_scores[doc_id] = v_rrf + b_rrf

        # Build results
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        results = []
        for doc_id, fused_score in ranked:
            v_rank, v_score = vector_scores.get(doc_id, (999, 0.0))
            b_rank, b_score = bm25_scores.get(doc_id, (999, 0.0))

            # Get full metadata from vector DB
            doc_text = None
            meta = {}
            if vector_scores:
                for r in vector_results:
                    if r.id == doc_id:
                        doc_text = getattr(r, 'document', None)
                        meta = getattr(r, 'metadata', {}) or {}
                        break

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
        # Keep only last 1000 latencies
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

        # Remove from BM25
        to_remove = set()
        if id:
            to_remove.add(id)
        if ids:
            to_remove.update(ids)
        for doc_id in to_remove:
            self._bm25.remove(doc_id)

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
            # Access internal engine for consolidation
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
            import numpy as np
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
        # Check vector DB
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
        # Reset in-memory storage
        self._db._vectors.clear()
        self._db._metadata.clear()
        self._db._documents.clear()
        self._db._deleted.clear()
