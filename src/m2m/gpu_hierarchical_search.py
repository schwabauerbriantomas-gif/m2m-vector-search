"""
HierarchicalGPUSearch — Two-stage ANN search, both stages on GPU.

Stage 1 (Coarse):  Query vs cluster centroids — O(Q × C), C << N
Stage 2 (Fine):    Query vs candidate cluster members — O(Q × n_probe × M)

Both stages use GPUVectorIndex for persistent GPU buffers.
Follows the pattern in the reference code provided by the user.
"""

import numpy as np
import time
from typing import Optional, Tuple


class HierarchicalGPUSearch:
    """
    Two-stage GPU hierarchical vector search.

    Architecture:
        ┌──────────────────────────────────────────────────────┐
        │  Stage 1 — Coarse   (GPU)                           │
        │  Compare Q queries vs C centroids                   │
        │  O(Q × C)    C = n_clusters (typically 100–1000)   │
        │  → returns n_probe closest cluster ids per query    │
        ├──────────────────────────────────────────────────────┤
        │  Stage 2 — Fine     (GPU)                           │
        │  Compare Q queries vs members of selected clusters  │
        │  O(Q × n_probe × M)   M = avg cluster size         │
        │  → returns top-k results                            │
        └──────────────────────────────────────────────────────┘

    The full vector set is partitioned once at build() time.
    Centroids are stored in a persistent GPU index.
    Each cluster's members are stored contiguously for cache efficiency.
    """

    def __init__(self, n_clusters: int = 100, n_probe: int = 5,
                 max_batch_size: int = 100):
        """
        Args:
            n_clusters:    Number of coarse clusters (C).
            n_probe:       Number of clusters to probe per query.
            max_batch_size: Max queries per batch_search() call.
        """
        self.n_clusters = n_clusters
        self.n_probe = min(n_probe, n_clusters)
        self.max_batch_size = max_batch_size

        self._is_built = False
        self._gpu_centroids: Optional[object] = None     # GPUVectorIndex of centroids
        self._cluster_members: list = []                 # list of np arrays per cluster
        self._cluster_gpu_indices: list = []             # GPUVectorIndex per cluster
        self._all_original_ids: list = []                # original vector ids per cluster

    # ─────────────────────────────────────────────────────────────────
    # Index building
    # ─────────────────────────────────────────────────────────────────

    def build(self, vectors: np.ndarray, use_gpu: bool = True):
        """
        Partition vectors into clusters and build GPU indices.

        Args:
            vectors:  shape [N, D] float32 — the full vector set.
            use_gpu:  Try to use GPUVectorIndex; fall back to CPU if unavailable.
        """
        t0 = time.perf_counter()
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        n, d = vectors.shape

        print(f"[HierarchicalGPUSearch] Building index: {n:,} vectors × {d} dims, "
              f"C={self.n_clusters}, n_probe={self.n_probe}")

        # ── KMeans clustering (CPU — done once at build time) ────────
        centroids, labels = self._kmeans(vectors, self.n_clusters)

        # ── Group vectors by cluster ──────────────────────────────────
        self._cluster_members = []
        self._all_original_ids = []
        for c in range(self.n_clusters):
            mask = labels == c
            members = vectors[mask]
            ids = np.where(mask)[0]
            self._cluster_members.append(members)
            self._all_original_ids.append(ids)

        # ── Build GPU index for centroids ─────────────────────────────
        self._gpu_centroids = self._make_gpu_index(centroids, use_gpu)

        # ── Build GPU index per cluster ───────────────────────────────
        self._cluster_gpu_indices = []
        for members in self._cluster_members:
            if len(members) == 0:
                self._cluster_gpu_indices.append(None)
            else:
                self._cluster_gpu_indices.append(
                    self._make_gpu_index(members, use_gpu)
                )

        self._is_built = True
        build_time = time.perf_counter() - t0
        print(f"[HierarchicalGPUSearch] Built in {build_time:.2f}s  "
              f"| avg cluster size: {n // self.n_clusters}")

    def _make_gpu_index(self, vectors: np.ndarray, use_gpu: bool):
        """Try to create GPUVectorIndex; fall back to numpy wrapper."""
        if use_gpu:
            try:
                from gpu_vector_index import GPUVectorIndex
                return GPUVectorIndex(vectors, max_batch_size=self.max_batch_size)
            except Exception as e:
                print(f"[HierarchicalGPUSearch] GPU init failed ({e}), using CPU fallback.")
        return _CPUFallbackIndex(vectors)

    # ─────────────────────────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────────────────────────

    def batch_search(self, queries: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Two-stage GPU hierarchical search for a batch of queries.

        Args:
            queries: shape [B, D] float32
            k: number of nearest neighbours per query

        Returns:
            indices   shape [B, k]  int64  (original index positions)
            distances shape [B, k]  float32
        """
        assert self._is_built, "Call build() before search()."
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        batch_size, d = queries.shape

        # ── Stage 1: Coarse — queries vs centroids (GPU) ──────────────
        _, coarse_dists = self._gpu_centroids.batch_search(queries, k=self.n_probe)
        coarse_ids, _ = self._gpu_centroids.batch_search(queries, k=self.n_probe)

        # ── Stage 2: Fine — queries vs candidate clusters (GPU) ───────
        all_indices = [[] for _ in range(batch_size)]
        all_dists   = [[] for _ in range(batch_size)]

        for cluster_id in range(self.n_clusters):
            # Which queries probe this cluster?
            query_mask = np.any(coarse_ids == cluster_id, axis=1)
            probing_queries = np.where(query_mask)[0]
            if len(probing_queries) == 0:
                continue

            gpu_idx = self._cluster_gpu_indices[cluster_id]
            if gpu_idx is None or len(self._cluster_members[cluster_id]) == 0:
                continue

            q_batch = queries[probing_queries]
            local_ids, local_dists = gpu_idx.batch_search(q_batch, k=k)

            # Map local cluster ids back to global ids
            orig_ids = self._all_original_ids[cluster_id]
            for i, qi in enumerate(probing_queries):
                global_ids = orig_ids[local_ids[i]]
                all_indices[qi].append(global_ids)
                all_dists[qi].append(local_dists[i])

        # ── Merge and top-k ───────────────────────────────────────────
        final_ids  = np.zeros((batch_size, k), dtype=np.int64)
        final_dists = np.full((batch_size, k), np.inf, dtype=np.float32)

        for qi in range(batch_size):
            if all_indices[qi]:
                merged_ids  = np.concatenate(all_indices[qi])
                merged_dists = np.concatenate(all_dists[qi])
                # Deduplicate by keeping lowest dist per id
                _, uniq = np.unique(merged_ids, return_index=True)
                merged_ids  = merged_ids[uniq]
                merged_dists = merged_dists[uniq]
                # Top-k
                kk = min(k, len(merged_ids))
                order = np.argpartition(merged_dists, kk - 1)[:kk]
                order = order[np.argsort(merged_dists[order])]
                final_ids[qi, :kk]   = merged_ids[order]
                final_dists[qi, :kk] = merged_dists[order]

        return final_ids, final_dists

    def search_single(self, query: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Single-query convenience wrapper."""
        ids, dists = self.batch_search(query.reshape(1, -1), k=k)
        return ids[0], dists[0]

    # ─────────────────────────────────────────────────────────────────
    # KMeans (CPU — only at build time)
    # ─────────────────────────────────────────────────────────────────

    def _kmeans(self, vectors: np.ndarray, n_clusters: int,
                max_iter: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Simple Lloyd's KMeans — used once at build time."""
        rng = np.random.default_rng(42)
        n, d = vectors.shape
        n_clusters = min(n_clusters, n)

        # Random init
        idx = rng.choice(n, n_clusters, replace=False)
        centroids = vectors[idx].copy()

        for _ in range(max_iter):
            # Assign: find nearest centroid for each vector
            diffs = vectors[:, None, :] - centroids[None, :, :]   # [N, C, D]
            dists = np.linalg.norm(diffs, axis=-1)                 # [N, C]
            labels = np.argmin(dists, axis=-1)                     # [N]

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for c in range(n_clusters):
                members = vectors[labels == c]
                if len(members) > 0:
                    new_centroids[c] = members.mean(axis=0)
                else:
                    new_centroids[c] = centroids[c]

            if np.allclose(new_centroids, centroids, atol=1e-6):
                break
            centroids = new_centroids

        return centroids, labels


# ─────────────────────────────────────────────────────────────────
# CPU fallback (when Vulkan unavailable)
# ─────────────────────────────────────────────────────────────────

class _CPUFallbackIndex:
    """NumPy brute-force fallback when GPU is unavailable."""

    def __init__(self, vectors: np.ndarray):
        self._vecs = vectors.astype(np.float32)

    def batch_search(self, queries: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        queries = queries.astype(np.float32)
        # [B, N] distance matrix
        diff = queries[:, None, :] - self._vecs[None, :, :]   # [B, N, D]
        dists = np.linalg.norm(diff, axis=-1)                 # [B, N]
        k = min(k, self._vecs.shape[0])
        ids = np.argpartition(dists, k - 1, axis=1)[:, :k]
        top_dists = np.take_along_axis(dists, ids, axis=1)
        order = np.argsort(top_dists, axis=1)
        ids = np.take_along_axis(ids, order, axis=1)
        top_dists = np.take_along_axis(top_dists, order, axis=1)
        return ids, top_dists
