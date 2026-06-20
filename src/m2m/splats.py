from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from .gaussian_scoring import gaussian_score, two_phase_search
from .hrm2_engine import HRM2Engine
from .online_updates import FeedbackEvent, OnlineUpdater
from .splat_types import GaussianSplat

if TYPE_CHECKING:
    from .config import M2MConfig

import logging

logger = logging.getLogger(__name__)


class SplatStore:
    """
    Gaussian Splat vector store with probabilistic scoring and online learning.

    Each stored vector is a Gaussian in embedding space:
        Gᵢ(x) = αᵢ · exp(-κᵢ · ‖x - μᵢ‖²)

    Search uses two-phase retrieval:
        1. L2 candidate retrieval (fast pruning)
        2. Gaussian re-ranking (probabilistic scoring)

    Parameters adapt via user feedback:
        - α (amplitude): importance of this memory, decays over time
        - κ (concentration): specificity, increases with relevant hits
        - μ (center): drifts toward queries that find it relevant
    """

    def __init__(self, config: M2MConfig) -> None:
        self.config = config

        self.max_splats = config.max_splats
        self.n_active = 0

        # Determine number of clusters based on config
        n_coarse = max(10, int(np.sqrt(self.max_splats) / 10))
        n_fine = max(100, int(self.max_splats / n_coarse))

        self.engine = HRM2Engine(
            n_coarse=n_coarse,
            n_fine=n_fine,
            embedding_dim=config.latent_dim,
            batch_size=min(10000, self.max_splats),
            config=config,
        )

        # Keep track of tensors for Energy / SOC functions that access properties directly
        self.mu = np.zeros((self.max_splats, config.latent_dim), dtype=np.float32)
        self.alpha = np.ones((self.max_splats,), dtype=np.float32) * config.init_alpha
        self.kappa = np.ones((self.max_splats,), dtype=np.float32) * config.init_kappa
        self.frequency = np.zeros((self.max_splats,), dtype=np.float32)

        # Internal splat counter for ID generation
        self._next_id = 0

        # GPUVectorIndex: lazy-init on first batch_find_neighbors when Vulkan enabled.
        # Rebuilds automatically when index changes (dirty flag).
        self._gpu_index = None
        self._gpu_index_dirty = True

        # CUDASearcher: lazy-init for CUDA brute-force path.
        self._cuda_searcher = None
        self._cuda_dirty = True

        # Online updater for feedback-driven learning
        self._updater = OnlineUpdater()

    @property
    def updater(self) -> OnlineUpdater:
        """Access the online parameter updater for feedback."""
        return self._updater

    def add_splat(self, x: np.ndarray) -> bool:
        """Add a batch of splats or a single splat."""
        if x.ndim == 1:
            x = x[np.newaxis, :]

        n_new = x.shape[0]
        if self.n_active + n_new > self.max_splats:
            return False

        new_splats = []
        for i in range(n_new):
            idx = self._next_id
            self._next_id += 1

            # Update tensor tracking
            self.mu[self.n_active] = x[i]
            self.alpha[self.n_active] = self.config.init_alpha
            self.kappa[self.n_active] = self.config.init_kappa
            self.frequency[self.n_active] = 1.0  # initial access

            self.n_active += 1

            splat = GaussianSplat(id=idx)
            new_splats.append(splat)

        # Add dummy splats to engine so it knows the size
        self.engine.add_splats(new_splats)
        self._cuda_dirty = True
        return True

    def build_index(self) -> None:
        """Build the semantic router index from active vectors."""
        if self.n_active == 0:
            return
        embeddings = self.mu[: self.n_active]
        self.engine.index(precomputed_embeddings=embeddings)
        self._gpu_index_dirty = True
        self._cuda_dirty = True

    def find_neighbors(
        self, query: np.ndarray, k: int = 64, lod: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find k-nearest neighbors using two-phase Gaussian search.

        Phase 1: L2 candidate retrieval via HRM2 or brute-force
        Phase 2: Gaussian re-ranking with α·exp(-κ·‖x-μ‖²)

        Returns (mu, alpha, kappa) for the top-k results, ranked by
        Gaussian score (highest first = best match).
        """
        query_np = query
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)

        batch_size = query_np.shape[0]
        dim = query_np.shape[1]
        n = self.n_active

        # -- Input validation ---------------------------------
        if k < 1:
            k = 1
        if query_np.size == 0:
            raise ValueError("Query vector must not be empty")
        if not np.all(np.isfinite(query_np)):
            raise ValueError("Query vector contains NaN or Inf values")
        if dim != self.config.latent_dim:
            raise ValueError(
                f"Query dimension mismatch: expected {self.config.latent_dim}, got {dim}"
            )

        k = min(k, max(1, n))

        if n == 0:
            # No splats: return empty arrays, don't raise
            mu_out = np.zeros((batch_size, 0, dim), dtype=np.float32)
            alpha_out = np.zeros((batch_size, 0), dtype=np.float32)
            kappa_out = np.zeros((batch_size, 0), dtype=np.float32)
            return mu_out, alpha_out, kappa_out

        # Auto-build index if splats exist but haven't been indexed yet
        if not self.engine._is_indexed:
            self.build_index()

        # Active data slices
        index_data = self.mu[:n]
        index_alpha = self.alpha[:n]
        index_kappa = self.kappa[:n]

        mu_out = np.zeros((batch_size, k, dim), dtype=np.float32)
        alpha_out = np.zeros((batch_size, k), dtype=np.float32)
        kappa_out = np.zeros((batch_size, k), dtype=np.float32)

        # Use HRM2 clustering pruning when dataset is large enough
        use_hrm2 = n > 15000 and self.engine.coarse_model is not None

        for i in range(batch_size):
            q = query_np[i].astype(np.float32)

            if use_hrm2 and lod == 2:
                # HRM2 accelerated path for large datasets
                n_probe = self.engine.n_probe
                coarse_dists = self.engine.coarse_model.transform(q.reshape(1, -1))[0]

                if coarse_dists.shape[0] > n_probe:
                    closest_coarse = np.argpartition(coarse_dists, n_probe - 1)[:n_probe]
                else:
                    closest_coarse = np.argsort(coarse_dists)

                candidate_lists = []
                for c in closest_coarse:
                    cidx = self.engine.coarse_cluster_indices.get(c)
                    if cidx is not None and len(cidx) > 0:
                        candidate_lists.append(cidx)

                if not candidate_lists:
                    continue

                candidates = np.concatenate(candidate_lists)
            else:
                # All points are candidates for small datasets
                candidates = np.arange(n)

            # Two-phase Gaussian search on candidates
            cand_mu = index_data[candidates]
            cand_alpha = index_alpha[candidates]
            cand_kappa = index_kappa[candidates]

            _, scores, _, _ = two_phase_search(
                q, cand_mu, cand_alpha, cand_kappa, k=k, overfetch=1.5
            )

            # Top-k by Gaussian score (already sorted by two_phase_search)
            # Re-compute to get the right candidate indices
            top_local = np.argsort(-scores)[:k]

            for j, local_j in enumerate(top_local):
                idx = candidates[local_j]
                mu_out[i, j] = index_data[idx]
                alpha_out[i, j] = index_alpha[idx]
                kappa_out[i, j] = index_kappa[idx]

        return mu_out, alpha_out, kappa_out

    def batch_find_neighbors(
        self,
        queries: np.ndarray,
        k: int = 64,
        lod: int = 2,
        max_batch_size: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Batch k-NN search with Gaussian re-ranking.

        Uses GPUVectorIndex (Vulkan) or CUDASearcher when available,
        falls back to CPU with Gaussian re-ranking.
        """
        if queries.ndim == 1:
            queries = queries[np.newaxis, :]
        batch_size = queries.shape[0]
        dim = queries.shape[1]
        k = min(k, max(1, self.n_active))

        # -- CUDA path: CUDASearcher (SPEC 1) -------------------------
        cuda_enabled = getattr(self.config, "enable_cuda", False)
        if cuda_enabled and self.n_active > 0:
            try:
                if self._cuda_searcher is None or self._cuda_dirty:
                    from .cuda_search import CUDASearcher

                    index_vecs = self.mu[: self.n_active]
                    metric = getattr(self.config, "cuda_metric", "cosine")
                    self._cuda_searcher = CUDASearcher(index_vecs, metric=metric)
                    self._cuda_dirty = False

                queries_np = queries.astype(np.float32)
                if queries_np.ndim == 1:
                    queries_np = queries_np[np.newaxis, :]

                if queries_np.shape[0] == 1:
                    gpu_ids, gpu_dists = self._cuda_searcher.search(queries_np[0], k=k)
                    gpu_ids = gpu_ids[np.newaxis, :]
                    gpu_dists = gpu_dists[np.newaxis, :]
                else:
                    gpu_ids, gpu_dists = self._cuda_searcher.search_batch(queries_np, k=k)

                # Gaussian re-ranking of GPU candidates
                return self._gaussian_rerank_gpu_results(gpu_ids, gpu_dists, queries_np, k, dim)

            except Exception as e:
                logger.warning("CUDA batch search failed (%s), falling back.", e)

        # -- GPU path: GPUVectorIndex (Vulkan) ------------------------
        vulkan_enabled = getattr(self.config, "enable_vulkan", False)
        if vulkan_enabled and self.n_active > 0:
            try:
                if self._gpu_index is None or self._gpu_index_dirty:
                    from gpu_vector_index import GPUVectorIndex

                    index_vecs = self.mu[: self.n_active]
                    self._gpu_index = GPUVectorIndex(index_vecs, max_batch_size=max_batch_size)
                    self._gpu_index_dirty = False

                queries_np = queries.astype(np.float32)
                gpu_ids, gpu_dists = self._gpu_index.batch_search(queries_np, k=k)

                return self._gaussian_rerank_gpu_results(gpu_ids, gpu_dists, queries_np, k, dim)

            except Exception as e:
                logger.warning("GPU batch search failed (%s), falling back to CPU.", e)

        # -- CPU fallback: two-phase Gaussian search -------------------
        n = self.n_active
        index_data = self.mu[:n].astype(np.float32)
        index_alpha = self.alpha[:n]
        index_kappa = self.kappa[:n]
        queries_np = queries.astype(np.float32)

        mu_out = np.zeros((batch_size, k, dim), dtype=np.float32)
        alpha_out = np.zeros((batch_size, k), dtype=np.float32)
        kappa_out = np.zeros((batch_size, k), dtype=np.float32)

        for i in range(batch_size):
            q = queries_np[i]
            _, scores, _, _ = two_phase_search(
                q, index_data, index_alpha, index_kappa, k=k, overfetch=2.0
            )
            top_local = np.argsort(-scores)[:k]

            for j, local_j in enumerate(top_local):
                mu_out[i, j] = index_data[local_j]
                alpha_out[i, j] = index_alpha[local_j]
                kappa_out[i, j] = index_kappa[local_j]

        return mu_out, alpha_out, kappa_out

    def _gaussian_rerank_gpu_results(
        self,
        gpu_ids: np.ndarray,
        gpu_dists: np.ndarray,
        queries_np: np.ndarray,
        k: int,
        dim: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Re-rank GPU results using Gaussian scoring."""
        batch_size = queries_np.shape[0]
        n = self.n_active

        mu_out = np.zeros((batch_size, k, dim), dtype=np.float32)
        alpha_out = np.zeros((batch_size, k), dtype=np.float32)
        kappa_out = np.zeros((batch_size, k), dtype=np.float32)

        for i in range(batch_size):
            # Get candidate indices from GPU results
            valid_mask = gpu_ids[i] < n
            candidates = gpu_ids[i][valid_mask].astype(int)

            if len(candidates) == 0:
                continue

            # Gaussian re-ranking
            cand_mu = self.mu[candidates]
            cand_alpha = self.alpha[candidates]
            cand_kappa = self.kappa[candidates]

            scores = gaussian_score(queries_np[i], cand_mu, cand_alpha, cand_kappa)
            top_local = np.argsort(-scores)[:k]

            for j, local_j in enumerate(top_local[:k]):
                idx = candidates[local_j]
                mu_out[i, j] = self.mu[idx]
                alpha_out[i, j] = self.alpha[idx]
                kappa_out[i, j] = self.kappa[idx]

        return mu_out, alpha_out, kappa_out

    def feedback(
        self,
        query: np.ndarray,
        relevant_ids: Optional[List[int]] = None,
        irrelevant_ids: Optional[List[int]] = None,
    ) -> Dict[str, int]:
        """
        Submit feedback for a query to adapt splat parameters.

        This is the key learning mechanism: after a search, the user
        confirms which results were relevant and which weren't.
        The splat parameters (α, κ, μ) are updated accordingly.

        Args:
            query: [D] the query vector
            relevant_ids: indices of relevant results
            irrelevant_ids: indices of irrelevant results

        Returns:
            Summary of updates applied.
        """
        relevant_ids = relevant_ids or []
        irrelevant_ids = irrelevant_ids or []

        query = np.asarray(query, dtype=np.float32).flatten()

        self._updater.apply_batch_feedback(
            query=query,
            relevant_indices=relevant_ids,
            irrelevant_indices=irrelevant_ids,
            mu=self.mu,
            alpha=self.alpha,
            kappa=self.kappa,
        )

        # Update frequency counters for relevant hits
        for idx in relevant_ids:
            if 0 <= idx < self.n_active:
                self.frequency[idx] += 1.0

        return self._updater.get_feedback_summary()

    def entropy(self, x: Optional[np.ndarray] = None) -> float:
        """
        Compute Shannon entropy of the active kappa (concentration) distribution.

        Normalizes kappa values to a probability distribution and computes:
            H = -sum(p * log(p))

        Returns:
            float in [0.0, 1.0] — 0.0 = all identical, 1.0 = uniform distribution.
        """
        if self.n_active == 0:
            return 0.0

        kappa = self.kappa[: self.n_active]
        kappa = kappa[kappa > 0]

        if len(kappa) == 0:
            return 0.0

        total = kappa.sum()
        if total <= 0:
            return 0.0

        p = kappa / total
        h = -np.sum(p * np.log(p))

        max_h = np.log(len(p))
        if max_h <= 0:
            return 0.0

        return float(np.clip(h / max_h, 0.0, 1.0))

    def compact(self) -> int:
        """
        Remove dead splats and recompact arrays in-place.

        Dead splats are those with:
        - alpha below minimum threshold (< 0.01)
        - mu containing NaN or Inf

        Returns:
            Number of splats removed.
        """
        if self.n_active == 0:
            return 0

        n = self.n_active
        mu = self.mu[:n]
        alpha = self.alpha[:n]
        kappa = self.kappa[:n]
        frequency = self.frequency[:n]

        # Build mask: keep splats that are valid
        valid_alpha = alpha >= 0.01
        valid_mu = np.all(np.isfinite(mu), axis=1)
        mask = valid_alpha & valid_mu

        n_kept = int(mask.sum())
        removed = n - n_kept

        if removed == 0:
            return 0

        # Reassign compacted data
        self.mu[:n_kept] = mu[mask]
        self.alpha[:n_kept] = alpha[mask]
        self.kappa[:n_kept] = kappa[mask]
        self.frequency[:n_kept] = frequency[mask]

        # Zero out the rest
        self.mu[n_kept:n] = 0.0
        self.alpha[n_kept:n] = 0.0
        self.kappa[n_kept:n] = 0.0
        self.frequency[n_kept:n] = 0.0

        self.n_active = n_kept
        return removed

    def get_statistics(self) -> Dict[str, object]:
        n = self.n_active
        stats = {
            "n_active": n,
            "max_splats": self.max_splats,
            "alpha_mean": float(self.alpha[:n].mean()) if n > 0 else 0.0,
            "alpha_std": float(self.alpha[:n].std()) if n > 0 else 0.0,
            "kappa_mean": float(self.kappa[:n].mean()) if n > 0 else 0.0,
            "kappa_std": float(self.kappa[:n].std()) if n > 0 else 0.0,
            "entropy": self.entropy(),
            "hrm2_stats": self.engine.get_stats(),
            "feedback_stats": self._updater.get_feedback_summary(),
        }
        return stats

    def _build_hrm2_from_splats(self, splats: List[GaussianSplat]) -> None:
        """Construye índice HRM2 desde splats pre-computados."""
        from .splat_types import GaussianSplat as HrmSplat

        n_new = len(splats)
        if n_new > self.max_splats:
            raise ValueError(f"Too many splats to load ({n_new} > {self.max_splats})")

        new_splats = []
        for i, s in enumerate(splats):
            self.mu[i] = np.array(s.mu, dtype=np.float32)
            self.alpha[i] = s.alpha
            self.kappa[i] = s.kappa
            self.frequency[i] = 1.0

            splat_obj = HrmSplat(id=i)
            new_splats.append(splat_obj)

        self.n_active = n_new
        self._next_id = n_new

        # Clear and rebuild engine
        self.engine.clear()
        self.engine.add_splats(new_splats)

        # Bypass encoder
        embeddings = self.mu[: self.n_active]
        self.engine.index(precomputed_embeddings=embeddings)
        self._gpu_index_dirty = True
