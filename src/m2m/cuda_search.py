#!/usr/bin/env python3
"""
CUDA Brute-Force Search — PyTorch CUDA backend for M2M.

SPEC 1: CUDA brute-force search via torch.matmul (cosine similarity / dot product).
SPEC 2: Precomputed L2 norms — zero allocation at query time.
SPEC 4: Multi-start search with majority voting.

Target: <2ms per query on 10K embeddings × 640D (RTX 3090).
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Lazy torch import at module level for method usage
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


class CUDASearcher:
    """
    High-performance CUDA brute-force k-NN search with precomputed norms.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        metric: str = "cosine",
        device: str = "cuda",
    ):
        """
        Args:
            embeddings: shape [N, D] float32 — the index vectors.
            metric: 'cosine' or 'l2'.
            device: torch device string (default 'cuda').
        """
        import torch

        if not _has_cuda():
            raise RuntimeError("CUDA not available via PyTorch.")

        self._device = torch.device(device)
        self._metric = metric

        if (
            embeddings is None or embeddings.nelement() == 0
            if hasattr(embeddings, "nelement")
            else (embeddings is None or embeddings.size == 0)
        ):
            raise ValueError("embeddings must be a non-empty array")

        self._n, self._dim = embeddings.shape

        # Upload index to GPU — ONCE
        idx_np = np.ascontiguousarray(embeddings, dtype=np.float32)
        try:
            self._index = torch.from_numpy(idx_np).to(self._device)  # [N, D]
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError(
                f"CUDA OOM: cannot allocate {idx_np.nbytes / 1024**2:.1f} MB for "
                f"{self._n}×{self._dim} index. Try reducing dataset size or using a smaller device."
            )

        # SPEC 2: Precompute ‖x_i‖² for all embeddings
        with torch.no_grad():
            self._norms_sq = (self._index**2).sum(dim=1)  # [N] on GPU

        logger.info(
            "[CUDASearcher] Ready — %d vectors × %dD, metric=%s, "
            "%.1f MB on GPU, norms precomputed",
            self._n,
            self._dim,
            self._metric,
            self._index.nbytes / 1024**2,
        )

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for k nearest neighbors.

        Args:
            query: shape [D] or [1, D] float32
            k: number of neighbors

        Returns:
            indices:   shape [k] int64
            distances: shape [k] float32
        """
        import torch

        # Ensure query shape [1, D]
        if query.ndim == 1:
            query = query[np.newaxis, :]
        if query.shape[-1] != self._dim:
            raise ValueError(
                f"Query dimension mismatch: expected {self._dim}, got {query.shape[-1]}"
            )
        try:
            q = torch.from_numpy(np.ascontiguousarray(query, dtype=np.float32)).to(
                self._device
            )  # [1, D]
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError(
                "CUDA OOM during query upload. Reduce batch size or free GPU memory."
            )

        k = min(k, self._n)

        try:
            if self._metric == "cosine":
                scores = self._cosine_scores(q)
                # torch.topk on GPU for largest scores
                topk_scores, topk_idx = torch.topk(scores, k, dim=1)
                topk_scores = topk_scores.cpu().numpy().flatten()
                topk_idx = topk_idx.cpu().numpy().flatten()
                return topk_idx.astype(np.int64), (1.0 - topk_scores).astype(np.float32)

            else:  # l2
                dists = self._l2_distances(q)
                topk_dists, topk_idx = torch.topk(dists, k, dim=1, largest=False)
                return (
                    topk_idx.cpu().numpy().flatten().astype(np.int64),
                    topk_dists.cpu().numpy().flatten().astype(np.float32),
                )
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError("CUDA OOM during search. Reduce k or free GPU memory.")

    def search_batch(
        self,
        queries: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch search for B queries.

        Returns:
            indices:   shape [B, k] int64
            distances: shape [B, k] float32
        """
        import torch

        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[np.newaxis, :]
        B = queries.shape[0]
        k = min(k, self._n)

        try:
            q = torch.from_numpy(queries).to(self._device)  # [B, D]
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError("CUDA OOM during batch query upload. Reduce batch size.")

        try:
            if self._metric == "cosine":
                scores = self._cosine_scores(q)  # [B, N]
                # topk for largest
                topk_scores, topk_idx = torch.topk(scores, k, dim=1)
                distances = (1.0 - topk_scores).cpu().numpy()
                indices = topk_idx.cpu().numpy()
                return indices.astype(np.int64), distances.astype(np.float32)
            else:
                dists = self._l2_distances(q)  # [B, N]
                topk_dists, topk_idx = torch.topk(dists, k, dim=1, largest=False)
                return topk_idx.cpu().numpy().astype(np.int64), topk_dists.cpu().numpy().astype(
                    np.float32
                )
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError("CUDA OOM during batch search. Reduce batch size or k.")

    def _cosine_scores(self, q: "torch.Tensor") -> "torch.Tensor":
        """Compute cosine similarity: q @ index^T / (‖q‖ * ‖x_i‖)."""
        # ‖q‖ per query row
        q_norms = torch.norm(q, dim=1, keepdim=True)  # [B, 1]
        # Dot product: [B, D] @ [D, N] -> [B, N]
        dots = q @ self._index.T
        # Precomputed ‖x_i‖
        x_norms = torch.sqrt(self._norms_sq).unsqueeze(0)  # [1, N]
        denom = q_norms * x_norms
        denom = torch.clamp(denom, min=1e-8)
        return dots / denom

    def _l2_distances(self, q: "torch.Tensor") -> "torch.Tensor":
        """
        L2 distances using precomputed norms (SPEC 2).
        dist²(q, x_i) = ‖q‖² + ‖x_i‖² - 2·q·x_i
        """
        q_sq = (q**2).sum(dim=1, keepdim=True)  # [B, 1]
        x_sq = self._norms_sq.unsqueeze(0)  # [1, N]
        dots = q @ self._index.T  # [B, N]
        dist_sq = q_sq + x_sq - 2 * dots
        dist_sq = torch.clamp(dist_sq, min=0.0)
        return torch.sqrt(dist_sq)

    def rebuild(self, new_embeddings: np.ndarray):
        """Re-upload index (call when embeddings change)."""
        import torch

        idx_np = np.ascontiguousarray(new_embeddings, dtype=np.float32)
        self._n, self._dim = idx_np.shape
        self._index = torch.from_numpy(idx_np).to(self._device)
        with torch.no_grad():
            self._norms_sq = (self._index**2).sum(dim=1)
        logger.info("[CUDASearcher] Index rebuilt — %d vectors", self._n)

    @property
    def n_vectors(self) -> int:
        return self._n


class MultiStartSearcher:
    """
    SPEC 4: Multi-start search with majority voting.

    Runs k independent searches from perturbed query points and aggregates
    results via majority voting (frequency of appearance in top-k lists).

    Can improve recall at the cost of latency.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        n_starts: int = 3,
        noise_scale: float = 0.01,
        metric: str = "cosine",
        device: str = "cuda",
    ):
        import torch

        self._searcher = CUDASearcher(embeddings, metric=metric, device=device)
        self._n_starts = n_starts
        self._noise_scale = noise_scale
        self._device = torch.device(device)

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multi-start search: n_starts perturbed queries + majority vote.

        Returns:
            indices:   shape [k] int64  — aggregated top-k
            distances: shape [k] float32 — average distance from votes
        """
        import torch

        query = np.asarray(query, dtype=np.float32).flatten()

        # Score each index by frequency of appearing across n_starts
        vote_counts = {}
        vote_dists = {}

        for start in range(self._n_starts):
            if start == 0:
                perturbed = query
            else:
                # Add small noise
                noise = np.random.randn(len(query)).astype(np.float32) * self._noise_scale
                perturbed = query + noise

            indices, distances = self._searcher.search(perturbed, k=k)

            for idx, dist in zip(indices, distances):
                idx = int(idx)
                if idx not in vote_counts:
                    vote_counts[idx] = 0
                    vote_dists[idx] = 0.0
                vote_counts[idx] += 1
                vote_dists[idx] += dist

        if not vote_counts:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        # Sort by vote count (desc), then by average distance (asc)
        candidates = sorted(
            vote_counts.items(),
            key=lambda x: (-x[1], vote_dists[x[0]] / x[1]),
        )

        top_k = candidates[:k]
        result_indices = np.array([c[0] for c in top_k], dtype=np.int64)
        result_dists = np.array([vote_dists[c[0]] / c[1] for c in top_k], dtype=np.float32)

        return result_indices, result_dists

    def rebuild(self, new_embeddings: np.ndarray):
        self._searcher.rebuild(new_embeddings)
