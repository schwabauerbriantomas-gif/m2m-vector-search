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

        # Ranking mode: "gaussian" (default, probabilistic re-ranking) or
        # "l2" (pure L2 distance ranking, for standard ANN benchmarks).
        # "l2" matches FAISS IVFFlat behavior exactly.
        self.rank_by = "gaussian"

        # Cached squared norms of mu[:n_active] (Gram-trick term ||mu_i||^2).
        # Invalidated by any mutator of mu (add_splat, compact, feedback drift).
        self._m_sq_cache: Optional[np.ndarray] = None
        self._m_sq_dirty = False

        # NOTE: n_coarse is re-derived from the actual dataset size at build time
        # (build_index -> _recompute_n_coarse). This initial value only sizes the
        # engine before any data is seen.
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

    def add_splat(self, x: np.ndarray) -> int:
        """Add a batch of splats or a single splat (vectorized)."""
        if x.ndim == 1:
            x = x[np.newaxis, :]

        n_new = x.shape[0]
        if self.n_active + n_new > self.max_splats:
            # Accept as many as fit
            n_new = self.max_splats - self.n_active
            if n_new <= 0:
                return 0
            x = x[:n_new]

        # Vectorized tensor updates (no Python loop)
        start = self.n_active
        end = start + n_new
        self.mu[start:end] = x
        self.alpha[start:end] = self.config.init_alpha
        self.kappa[start:end] = self.config.init_kappa
        self.frequency[start:end] = 1.0
        self.n_active = end
        self._next_id += n_new

        self._m_sq_dirty = True
        self._cuda_dirty = True
        return n_new

    def build_index(self) -> None:
        """Build the semantic router index from active vectors."""
        if self.n_active == 0:
            return
        self._recompute_n_coarse()
        embeddings = self.mu[: self.n_active]
        self.engine.index(precomputed_embeddings=embeddings)
        self._gpu_index_dirty = True
        self._cuda_dirty = True

    def _recompute_n_coarse(self) -> None:
        """Derive n_coarse from the ACTUAL number of active vectors (IVF rule ~sqrt(N)).

        The old heuristic derived it from max_splats (capacity), which degenerates:
        10-53 clusters regardless of the real dataset size, so n_probe=5 scanned
        25-50% of the dataset and the index lost against brute force below 1M rows.
        """
        n_coarse = max(2, int(np.sqrt(self.n_active)))
        # Never shrink below the configured floor of 10 (preserves old behavior
        # for tiny datasets); explicit user overrides win over the heuristic.
        user_cfg = getattr(self.config, "n_coarse_override", None)
        if user_cfg is not None and user_cfg > 0:
            n_coarse = int(user_cfg)
        else:
            n_coarse = max(10, n_coarse)
        self.engine.n_coarse = n_coarse

    def find_neighbors(
        self, query: np.ndarray, k: int = 64, lod: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Find k-nearest neighbors using two-phase Gaussian search.

        Phase 1: L2 candidate retrieval via HRM2 or brute-force
        Phase 2: Gaussian re-ranking with α·exp(-κ·‖x-μ‖²)

        Returns (mu, alpha, kappa, splat_indices) for the top-k results,
        ranked by Gaussian score (highest first = best match).
        splat_indices are positions into self.mu[:n_active].
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
            idx_out = np.zeros((batch_size, 0), dtype=np.int64)
            return mu_out, alpha_out, kappa_out, idx_out

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
        idx_out = np.full((batch_size, k), -1, dtype=np.int64)

        # Use HRM2 clustering pruning when the dataset is large enough to pay off.
        # 3000: below ~sqrt(N) clusters, scan cost ~ N * probe/sqrt(N); the crossover
        # with brute force (which now runs view-based, no copy) sits around 2-3k rows
        # on CPU. Old threshold (15000) left 99% of realistic workloads unpruned.
        use_hrm2 = n > 3000 and self.engine.coarse_model is not None

        # Use default n_probe — HRM2 auto-detection handles cluster count.
        # Silhouette-based scaling was tested and found unnecessary:
        # n_probe=5 already achieves recall≥0.9995 even with sil=0.13.
        if use_hrm2:
            effective_n_probe = self.engine.n_probe
        else:
            effective_n_probe = 0

        # Cached squared norms (Gram-trick term ||mu_i||^2), invalidated on mutation
        if self._m_sq_cache is None or self._m_sq_dirty or len(self._m_sq_cache) != n:
            self._m_sq_cache = np.einsum("ij,ij->i", index_data, index_data)
            self._m_sq_dirty = False
        _non_hrm2_m_sq = self._m_sq_cache
        assert _non_hrm2_m_sq is not None

        for i in range(batch_size):
            q = query_np[i].astype(np.float32)

            if use_hrm2 and lod == 2:
                # HRM2 accelerated path for large datasets
                n_probe = effective_n_probe
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
                full_scan = False
            else:
                # All points are candidates for small datasets.
                # Use a view (no O(n*d) copy); fancy-indexing arange would copy.
                candidates = None
                full_scan = True

            if full_scan:
                cand_mu = index_data
                cand_alpha = index_alpha
                cand_kappa = index_kappa
            else:
                cand_mu = index_data[candidates]
                cand_alpha = index_alpha[candidates]
                cand_kappa = index_kappa[candidates]

            q_sq = float(np.dot(q, q))

            if self.rank_by == "l2":
                # Pure L2 ranking — matches FAISS IVFFlat behavior exactly.
                # No Gaussian re-ranking, no score distortion.
                assert candidates is not None or full_scan
                if full_scan:
                    m_sq = _non_hrm2_m_sq
                    n_cand = n
                else:
                    assert candidates is not None
                    m_sq = _non_hrm2_m_sq[candidates]
                    n_cand = len(candidates)
                cross = cand_mu @ q
                dist_sq = q_sq + m_sq - 2.0 * cross
                np.maximum(dist_sq, 0.0, out=dist_sq)

                n_results = min(k, n_cand)
                if n_cand > n_results:
                    top_local = np.argpartition(dist_sq, n_results - 1)[:n_results]
                    top_local = top_local[np.argsort(dist_sq[top_local])]
                else:
                    top_local = np.argsort(dist_sq)[:n_results]

                if full_scan:
                    global_indices = top_local
                else:
                    assert candidates is not None
                    global_indices = candidates[top_local]
                mu_out[i, :n_results] = index_data[global_indices]
                alpha_out[i, :n_results] = index_alpha[global_indices]
                kappa_out[i, :n_results] = index_kappa[global_indices]
                idx_out[i, :n_results] = global_indices
            else:
                # Gaussian two-phase search (default, probabilistic re-ranking)
                result_indices, result_scores, _, _ = two_phase_search(
                    q, cand_mu, cand_alpha, cand_kappa, k=k, overfetch=1.5,
                    precomputed_m_sq=_non_hrm2_m_sq if full_scan else None,
                    with_ranks=False,
                )

                n_results = min(k, len(result_indices))
                if n_results > 0:
                    local_idx = result_indices[:n_results]
                    if full_scan:
                        global_indices = local_idx
                    else:
                        assert candidates is not None
                        global_indices = candidates[local_idx]
                    mu_out[i, :n_results] = index_data[global_indices]
                    alpha_out[i, :n_results] = index_alpha[global_indices]
                    kappa_out[i, :n_results] = index_kappa[global_indices]
                    idx_out[i, :n_results] = global_indices

        return mu_out, alpha_out, kappa_out, idx_out

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
                    from .gpu_vector_index import GPUVectorIndex

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

        # Pre-compute norms once for all queries
        _m_sq = np.einsum("ij,ij->i", index_data, index_data)

        mu_out = np.zeros((batch_size, k, dim), dtype=np.float32)
        alpha_out = np.zeros((batch_size, k), dtype=np.float32)
        kappa_out = np.zeros((batch_size, k), dtype=np.float32)

        for i in range(batch_size):
            q = queries_np[i]
            result_indices, scores, _, _ = two_phase_search(
                q, index_data, index_alpha, index_kappa, k=k, overfetch=2.0,
                precomputed_m_sq=_m_sq,
                with_ranks=False,
            )
            # result_indices are absolute positions into index_data, already
            # sorted by Gaussian score (descending) — use them directly.
            # (Previously re-ranked `scores` positions 0..k-1 were used as
            # absolute indices: returned vectors 0..k-1, not the k-NN.)
            n_results = min(k, len(result_indices))
            mu_out[i, :n_results] = index_data[result_indices[:n_results]]
            alpha_out[i, :n_results] = index_alpha[result_indices[:n_results]]
            kappa_out[i, :n_results] = index_kappa[result_indices[:n_results]]

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

            # Gaussian re-ranking — vectorized
            cand_mu = self.mu[candidates]
            cand_alpha = self.alpha[candidates]
            cand_kappa = self.kappa[candidates]

            scores = gaussian_score(queries_np[i], cand_mu, cand_alpha, cand_kappa)

            if len(scores) > k:
                top_local = np.argpartition(-scores, k - 1)[:k]
                top_local = top_local[np.argsort(-scores[top_local])]
            else:
                top_local = np.argsort(-scores)

            n_results = min(k, len(top_local))
            for j in range(n_results):
                idx = candidates[top_local[j]]
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
        if relevant_ids or irrelevant_ids:
            # apply_feedback may drift mu in-place (Hebbian update) — invalidate
            # the cached squared norms.
            self._m_sq_dirty = True

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
        self._m_sq_dirty = True
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
