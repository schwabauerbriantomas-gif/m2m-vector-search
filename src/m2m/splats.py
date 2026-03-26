from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from .hrm2_engine import HRM2Engine
from .splat_types import GaussianSplat

if TYPE_CHECKING:
    from .config import M2MConfig

import logging

logger = logging.getLogger(__name__)


class SplatStore:
    """Wrapper around the CPU-optimized HRM2Engine to interface with NumPy."""

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

    def add_splat(self, x: np.ndarray) -> bool:
        """Add a batch of splats or a single splat."""
        if x.ndim == 1:
            x = x[np.newaxis, :]

        n_new = x.shape[0]
        if self.n_active + n_new > self.max_splats:
            return False

        # Data is already numpy

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

            # Create GaussianSplat dummy (we don't have full 3D parsing yet, we use the vector as a proxy or just store defaults)
            # In a real system, x_np[i] would be decoded or we'd store it in the splat object
            splat = GaussianSplat(id=idx)
            # We will hack the embedding index later, for now we just add it to HRM2Engine
            # HRM2Engine expects we generate embeddings, but here our 'x' is ALREADY the embedding!
            new_splats.append(splat)

        # Add dummy splats to engine so it knows the size
        self.engine.add_splats(new_splats)
        self._cuda_dirty = True
        return True

    def build_index(self) -> None:
        """Build the semantic router index from active vectors."""
        if self.n_active == 0:
            return
        # Pass raw active vectors directly into HRM2 so we bypass the slow encoder
        embeddings = self.mu[: self.n_active]
        self.engine.index(precomputed_embeddings=embeddings)
        # Mark GPU index dirty so it is rebuilt on next batch_find_neighbors call
        self._gpu_index_dirty = True
        self._cuda_dirty = True

    def find_neighbors(
        self, query: np.ndarray, k: int = 64, lod: int = 2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find k-nearest neighbors using fast vectorized search with optional HRM2 routing."""
        query_np = query
        if query_np.ndim == 1:
            query_np = query_np.reshape(1, -1)

        batch_size = query_np.shape[0]
        dim = query_np.shape[1]
        n = self.n_active

        # -- CHAOS FIX: Validate inputs ---------------------------------
        if k < 1:  # C-01 fix: k=0 causes crash
            k = 1
        if query_np.size == 0:  # C-02 fix: empty query
            raise ValueError("Query vector must not be empty")
        if not np.all(np.isfinite(query_np)):  # C-03 fix: NaN/Inf detection
            raise ValueError("Query vector contains NaN or Inf values")
        if dim != self.config.latent_dim:  # H-02 fix: dimension validation
            raise ValueError(
                f"Query dimension mismatch: expected {self.config.latent_dim}, got {dim}"
            )

        k = min(k, max(1, n))

        if not self.engine._is_indexed or n == 0:
            mu_out = np.random.randn(batch_size, k, dim).astype(np.float32)
            alpha_out = np.ones((batch_size, k), dtype=np.float32)
            kappa_out = np.ones((batch_size, k), dtype=np.float32) * 10.0
            return mu_out, alpha_out, kappa_out

        # Precompute index embeddings slice for fast access
        index_data = self.mu[:n]  # [n, dim], already float32
        index_alpha = self.alpha[:n]
        index_kappa = self.kappa[:n]

        # Use HRM2 clustering pruning when dataset is large enough
        # For small N, vectorized brute-force is faster due to lower Python overhead
        use_hrm2 = n > 15000 and self.engine.coarse_model is not None

        mu_out = np.zeros((batch_size, k, dim), dtype=np.float32)
        alpha_out = np.zeros((batch_size, k), dtype=np.float32)
        kappa_out = np.zeros((batch_size, k), dtype=np.float32)

        if use_hrm2 and lod == 2:
            # HRM2 accelerated path for large datasets
            queries_np = query_np.astype(np.float32)
            # Precompute coarse distances for all queries at once
            coarse_dists = self.engine.coarse_model.transform(queries_np)  # [B, n_coarse]
            n_probe = self.engine.n_probe

            for i in range(batch_size):
                q = queries_np[i]
                # Find nearest coarse clusters
                if coarse_dists.shape[1] > n_probe:
                    closest_coarse = np.argpartition(coarse_dists[i], n_probe - 1)[:n_probe]
                else:
                    closest_coarse = np.argsort(coarse_dists[i])

                # Gather candidate indices from probed clusters
                candidate_lists = []
                for c in closest_coarse:
                    cidx = self.engine.coarse_cluster_indices.get(c)
                    if cidx is not None and len(cidx) > 0:
                        candidate_lists.append(cidx)

                if not candidate_lists:
                    continue

                candidates = np.concatenate(candidate_lists)
                # Vectorized squared L2 distance via einsum (no sqrt needed for ranking)
                diff = index_data[candidates] - q
                dists_sq = np.einsum("ij,ij->i", diff, diff)

                if len(dists_sq) > k:
                    topk_local = np.argpartition(dists_sq, k - 1)[:k]
                    sort_order = np.argsort(dists_sq[topk_local])
                    topk_local = topk_local[sort_order]
                else:
                    topk_local = np.argsort(dists_sq)

                for j, local_j in enumerate(topk_local[:k]):
                    idx = candidates[local_j]
                    mu_out[i, j] = index_data[idx]
                    alpha_out[i, j] = index_alpha[idx]
                    kappa_out[i, j] = index_kappa[idx]
        else:
            # Fast vectorized brute-force path (optimal for N ≤ 15K)
            queries_np = query_np.astype(np.float32)
            for i in range(batch_size):
                q = queries_np[i]
                diff = index_data - q  # [n, dim]
                dists_sq = np.einsum("ij,ij->i", diff, diff)  # [n]

                if n > k:
                    topk = np.argpartition(dists_sq, k - 1)[:k]
                    sort_order = np.argsort(dists_sq[topk])
                    topk = topk[sort_order]
                else:
                    topk = np.argsort(dists_sq)

                for j, idx in enumerate(topk[:k]):
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
        Batch k-NN search — uses GPUVectorIndex (persistent index, single dispatch)
        when Vulkan is enabled, falls back to sequential find_neighbors() on CPU.

        [OK] CORRECT pattern (reference implementation):
          - Index uploaded to GPU ONCE (or when dirty after rebuild)
          - Only queries (small) are transferred per call
          - All B queries dispatched in one vkCmdDispatch(ceil(N/256), B, 1)

        Args:
            queries: [B, D] tensor
            k:       number of neighbours
            lod:     level of detail (CPU path only)
            max_batch_size: max queries per GPU dispatch

        Returns:
            mu_out    [B, k, D]
            alpha_out [B, k]
            kappa_out [B, k]
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
                # Handle single query
                if queries_np.ndim == 1:
                    queries_np = queries_np[np.newaxis, :]

                if queries_np.shape[0] == 1:
                    gpu_ids, gpu_dists = self._cuda_searcher.search(queries_np[0], k=k)
                    gpu_ids = gpu_ids[np.newaxis, :]
                    gpu_dists = gpu_dists[np.newaxis, :]
                else:
                    gpu_ids, gpu_dists = self._cuda_searcher.search_batch(queries_np, k=k)

                mu_out = np.zeros((batch_size, k, dim), dtype=np.float32)
                alpha_out = np.zeros((batch_size, k), dtype=np.float32)
                kappa_out = np.zeros((batch_size, k), dtype=np.float32)

                for i in range(batch_size):
                    for j, idx in enumerate(gpu_ids[i]):
                        idx = int(idx)
                        if idx < self.n_active:
                            mu_out[i, j] = self.mu[idx]
                            alpha_out[i, j] = self.alpha[idx]
                            kappa_out[i, j] = self.kappa[idx]

                return mu_out, alpha_out, kappa_out

            except Exception as e:
                logger.warning("CUDA batch search failed (%s), falling back.", e)

        # -- GPU path: GPUVectorIndex (Vulkan) ------------------------
        vulkan_enabled = getattr(self.config, "enable_vulkan", False)
        if vulkan_enabled and self.n_active > 0:
            try:
                # Lazy init / rebuild when index vectors changed
                if self._gpu_index is None or self._gpu_index_dirty:
                    from gpu_vector_index import GPUVectorIndex

                    index_vecs = self.mu[: self.n_active]
                    self._gpu_index = GPUVectorIndex(index_vecs, max_batch_size=max_batch_size)
                    self._gpu_index_dirty = False

                queries_np = queries.astype(np.float32)
                gpu_ids, gpu_dists = self._gpu_index.batch_search(queries_np, k=k)

                mu_out = np.zeros((batch_size, k, dim), dtype=np.float32)
                alpha_out = np.zeros((batch_size, k), dtype=np.float32)
                kappa_out = np.zeros((batch_size, k), dtype=np.float32)

                for i in range(batch_size):
                    for j, idx in enumerate(gpu_ids[i]):
                        if idx < self.n_active:
                            mu_out[i, j] = self.mu[idx]
                            alpha_out[i, j] = self.alpha[idx]
                            kappa_out[i, j] = self.kappa[idx]

                return mu_out, alpha_out, kappa_out

            except Exception as e:
                logger.warning("GPU batch search failed (%s), falling back to CPU.", e)

        # -- CPU fallback: vectorized brute-force -------------------
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
            diff = index_data - q
            dists_sq = np.einsum("ij,ij->i", diff, diff)
            if n > k:
                topk = np.argpartition(dists_sq, k - 1)[:k]
                sort_order = np.argsort(dists_sq[topk])
                topk = topk[sort_order]
            else:
                topk = np.argsort(dists_sq)
            for j, idx in enumerate(topk[:k]):
                mu_out[i, j] = index_data[idx]
                alpha_out[i, j] = index_alpha[idx]
                kappa_out[i, j] = index_kappa[idx]

        return mu_out, alpha_out, kappa_out

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
        # Remove non-positive kappa (invalid concentration)
        kappa = kappa[kappa > 0]

        if len(kappa) == 0:
            return 0.0

        # Normalize to probability distribution
        total = kappa.sum()
        if total <= 0:
            return 0.0

        p = kappa / total
        # Shannon entropy (nats)
        h = -np.sum(p * np.log(p))

        # Normalize by max entropy (uniform distribution)
        max_h = np.log(len(p))
        if max_h <= 0:
            return 0.0

        return float(np.clip(h / max_h, 0.0, 1.0))

    def compact(self) -> None:
        """
        Remove invalid/dead splats and recompact arrays in-place.

        Removes splats where:
        - alpha ≈ 0 (< 1e-6)
        - mu contains NaN or Inf

        After removal, shifts remaining splats to contiguous indices.
        """
        if self.n_active == 0:
            return

        n = self.n_active
        mu = self.mu[:n]
        alpha = self.alpha[:n]
        kappa = self.kappa[:n]
        frequency = self.frequency[:n]

        # Build mask: keep splats that are valid
        valid_alpha = alpha >= 1e-6
        valid_mu = np.all(np.isfinite(mu), axis=1)
        mask = valid_alpha & valid_mu

        n_kept = int(mask.sum())
        if n_kept == n:
            return  # Nothing to compact

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

    def get_statistics(self) -> Dict[str, object]:
        return {
            "n_active": self.n_active,
            "max_splats": self.max_splats,
            "hrm2_stats": self.engine.get_stats(),
        }

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
