"""
M2M VectorIndex Interface & Strategy Pattern

Defines the abstract VectorIndex interface that all backends implement,
plus the IndexSelector for auto-detection logic.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class IndexSearchResult:
    """Unified search result from any VectorIndex backend."""

    indices: np.ndarray  # [K] indices into the stored vectors
    distances: np.ndarray  # [K] distances (lower = closer)
    ids: Optional[List[str]] = None  # optional doc IDs


class VectorIndex(ABC):
    """Abstract interface for vector search backends.

    All backends (BruteForce, HRM2, HNSW) must implement this interface.
    """

    @abstractmethod
    def build(self, vectors: np.ndarray) -> None:
        """Build index from vectors [N, D]."""
        ...

    @abstractmethod
    def search(self, query: np.ndarray, k: int = 10) -> IndexSearchResult:
        """Search for k nearest neighbors of query [D]."""
        ...

    @abstractmethod
    def add(self, vectors: np.ndarray) -> None:
        """Add new vectors [M, D] to existing index."""
        ...

    @abstractmethod
    def remove(self, indices: np.ndarray) -> None:
        """Remove vectors at given indices."""
        ...

    @property
    @abstractmethod
    def n_items(self) -> int:
        """Number of items currently indexed."""
        ...

    @property
    @abstractmethod
    def supports_remove(self) -> bool:
        """Whether the index supports efficient removal."""
        ...


class BruteForceIndex(VectorIndex):
    """Linear scan index — exact search, O(N) per query."""

    def __init__(self, metric: str = "cosine"):
        self._vectors: Optional[np.ndarray] = None
        self.metric = metric

    def build(self, vectors: np.ndarray) -> None:
        self._vectors = np.asarray(vectors, dtype=np.float32)

    def search(self, query: np.ndarray, k: int = 10) -> IndexSearchResult:
        query = np.asarray(query, dtype=np.float32).ravel()
        n = len(self._vectors)
        k = min(k, n)

        if self.metric == "cosine":
            # Cosine similarity -> convert to distance
            q_norm = query / (np.linalg.norm(query) + 1e-10)
            v_norms = self._vectors / (np.linalg.norm(self._vectors, axis=1, keepdims=True) + 1e-10)
            sims = v_norms @ q_norm  # [N]
            top_k = np.argpartition(-sims, k)[:k]
            top_k = top_k[np.argsort(-sims[top_k])]
            return IndexSearchResult(
                indices=top_k,
                distances=1.0 - sims[top_k],
            )
        else:
            # Euclidean distance
            diffs = self._vectors - query[np.newaxis, :]
            dists = np.sum(diffs**2, axis=1)  # [N]
            top_k = np.argpartition(dists, k)[:k]
            top_k = top_k[np.argsort(dists[top_k])]
            return IndexSearchResult(
                indices=top_k,
                distances=np.sqrt(dists[top_k]),
            )

    def add(self, vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])

    def remove(self, indices: np.ndarray) -> None:
        mask = np.ones(len(self._vectors), dtype=bool)
        mask[indices] = False
        self._vectors = self._vectors[mask]

    @property
    def n_items(self) -> int:
        return 0 if self._vectors is None else len(self._vectors)

    @property
    def supports_remove(self) -> bool:
        return True


@dataclass
class IndexSelectionResult:
    """Result of the auto-detection index selection."""

    recommended: str  # 'bruteforce', 'hrm2', 'hnsw'
    silhouette: float
    distance_cv: float  # coefficient of variation of distances
    n_vectors: int
    dim: int
    reason: str


def select_index_strategy(
    vectors: np.ndarray,
    force: Optional[str] = None,
) -> IndexSelectionResult:
    """Auto-detect the best index strategy based on data characteristics.

    Logic:
    - < 15K vectors -> BruteForce (always exact, fast enough)
    - 15K-100K with good cluster structure (silhouette > 0.15, CV > 0.2) -> HRM2
    - 15K-100K with poor cluster structure (silhouette < 0.15) -> HNSW
    - > 100K -> HNSW
    """
    n, dim = vectors.shape

    if force:
        return IndexSelectionResult(
            recommended=force,
            silhouette=-1,
            distance_cv=-1,
            n_vectors=n,
            dim=dim,
            reason="forced by user",
        )

    if n < 15_000:
        return IndexSelectionResult(
            recommended="bruteforce",
            silhouette=-1,
            distance_cv=-1,
            n_vectors=n,
            dim=dim,
            reason=f"{n} < 15K -> linear scan is sufficient",
        )

    # Compute data structure diagnostics
    silhouette = _compute_silhouette_safe(vectors)
    distance_cv = _compute_distance_cv(vectors)

    if n >= 100_000:
        return IndexSelectionResult(
            recommended="hnsw",
            silhouette=silhouette,
            distance_cv=distance_cv,
            n_vectors=n,
            dim=dim,
            reason=f"{n} >= 100K -> HNSW for scalability",
        )

    # Medium dataset: check clustering quality
    if silhouette > 0.15 and distance_cv > 0.2:
        return IndexSelectionResult(
            recommended="hrm2",
            silhouette=silhouette,
            distance_cv=distance_cv,
            n_vectors=n,
            dim=dim,
            reason=f"Good cluster structure (sil={silhouette:.3f}, cv={distance_cv:.3f}) -> HRM2",
        )
    else:
        return IndexSelectionResult(
            recommended="hnsw",
            silhouette=silhouette,
            distance_cv=distance_cv,
            n_vectors=n,
            dim=dim,
            reason=f"Poor cluster structure (sil={silhouette:.3f}, cv={distance_cv:.3f}) -> HNSW",
        )


def _compute_silhouette_safe(vectors: np.ndarray, sample_size: int = 1000) -> float:
    """Compute silhouette score safely, returning -1 on failure."""
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        n = len(vectors)
        if n < 3:
            return 1.0

        idx = np.random.choice(n, min(sample_size, n), replace=False)
        sample = vectors[idx]
        n_clusters = max(2, int(np.sqrt(len(sample))))
        labels = KMeans(n_clusters=n_clusters, n_init=1, random_state=42).fit_predict(sample)
        return float(silhouette_score(sample, labels))
    except Exception:
        return -1.0


def _compute_distance_cv(vectors: np.ndarray, sample_size: int = 500) -> float:
    """Compute coefficient of variation of pairwise distances."""
    n = len(vectors)
    sample_n = min(sample_size, n)
    idx = np.random.choice(n, sample_n, replace=False)
    sample = vectors[idx]

    # Compute distances to a few reference points
    refs = sample[: min(5, sample_n)]
    all_dists = []
    for ref in refs:
        dists = np.linalg.norm(sample - ref[np.newaxis, :], axis=1)
        all_dists.extend(dists.tolist())

    if not all_dists:
        return 0.0
    arr = np.array(all_dists)
    mean = np.mean(arr)
    if mean < 1e-10:
        return 0.0
    return float(np.std(arr) / mean)
