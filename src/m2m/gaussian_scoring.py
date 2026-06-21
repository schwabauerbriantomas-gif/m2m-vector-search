"""
Gaussian Splat Scoring Engine.

Replaces flat L2 distance with probabilistic Gaussian mixture scoring:
    score(x, i) = αᵢ · exp(-κᵢ · ‖x - μᵢ‖²)

This is the core differentiator: each memory is a Gaussian in embedding space,
not a point. Amplitude (α) encodes importance, concentration (κ) encodes
specificity, and the center (μ) encodes the embedding itself.

Two-phase search:
  1. Candidate retrieval via L2 (fast, prunes the search space)
  2. Gaussian re-ranking (scores candidates by probabilistic density)

This keeps performance competitive while using the Gaussian model for ranking.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def gaussian_score(
    query: np.ndarray,
    mu: np.ndarray,
    alpha: np.ndarray,
    kappa: np.ndarray,
    precomputed_dist_sq: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute Gaussian mixture scores for a single query against all splats.

    score_i = αᵢ · exp(-κᵢ · ‖q - μᵢ‖²)

    Higher score = better match (closer to splat center, higher amplitude).

    Args:
        query: Query vector [D]
        mu: Splat centers [N, D]
        alpha: Splat amplitudes [N]
        kappa: Splat concentrations [N]
        precomputed_dist_sq: Pre-computed squared L2 distances [N] to avoid
            recomputing them. When provided, skips the distance calculation
            entirely (useful when Phase 1 already computed them).

    Returns:
        scores: [N] — Gaussian mixture scores (higher = better match)
    """
    if precomputed_dist_sq is not None:
        dist_sq = precomputed_dist_sq
    else:
        # Gram-matrix trick: ‖q-μ‖² = ‖q‖² + ‖μ‖² - 2·q·μ
        # Avoids materializing the [N, D] diff array
        q_sq = np.dot(query, query)  # scalar
        m_sq = np.einsum("ij,ij->i", mu, mu)  # [N]
        cross = mu @ query  # [N]
        dist_sq = q_sq + m_sq - 2.0 * cross
        np.maximum(dist_sq, 0.0, out=dist_sq)

    # Clamp kappa to prevent numerical overflow
    exponent = -kappa * dist_sq
    np.clip(exponent, -100.0, 0.0, out=exponent)

    return alpha * np.exp(exponent)


def gaussian_score_batch(
    queries: np.ndarray,
    mu: np.ndarray,
    alpha: np.ndarray,
    kappa: np.ndarray,
    chunk_size: int = 4096,
) -> np.ndarray:
    """
    Compute Gaussian mixture scores for a batch of queries.

    Memory-efficient chunked implementation: processes queries in chunks of
    ``chunk_size`` to avoid materialising the full [B, N] distance matrix
    when B or N is large.

    Args:
        queries: [B, D]
        mu: [N, D]
        alpha: [N]
        kappa: [N]
        chunk_size: number of queries per chunk (memory vs speed tradeoff)

    Returns:
        scores: [B, N]
    """
    B = queries.shape[0]
    N = mu.shape[0]

    # Precompute splat norms (reused across all chunks)
    m_sq = np.sum(mu**2, axis=1)  # [N]

    scores = np.empty((B, N), dtype=np.float32)

    for start in range(0, B, chunk_size):
        end = min(start + chunk_size, B)
        q_chunk = queries[start:end]  # [C, D]

        q_sq = np.sum(q_chunk**2, axis=1, keepdims=True)  # [C, 1]
        cross = q_chunk @ mu.T  # [C, N]
        dist_sq = q_sq + m_sq[np.newaxis, :] - 2.0 * cross  # [C, N]
        np.maximum(dist_sq, 0.0, out=dist_sq)

        exponent = -kappa[np.newaxis, :] * dist_sq  # [C, N]
        np.clip(exponent, -100.0, 0.0, out=exponent)

        scores[start:end] = alpha[np.newaxis, :] * np.exp(exponent)

    return scores


def two_phase_search(
    query: np.ndarray,
    mu: np.ndarray,
    alpha: np.ndarray,
    kappa: np.ndarray,
    k: int = 64,
    overfetch: float = 2.0,
    precomputed_m_sq: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Two-phase search: L2 candidate retrieval + Gaussian re-ranking.

    Phase 1: L2 distance to get top-(k * overfetch) candidates (fast)
    Phase 2: Gaussian scoring to re-rank and select top-k (accurate)

    Args:
        query: [D] query vector
        mu: [N, D] splat centers
        alpha: [N] splat amplitudes
        kappa: [N] splat concentrations
        k: number of results
        overfetch: how many extra candidates to retrieve in phase 1
        precomputed_m_sq: Pre-computed ||μ_i||² [N] to avoid recomputing
            for every query in a batch. When provided, skips the einsum.

    Returns:
        indices: [k] — splat indices (sorted by Gaussian score, descending)
        scores: [k] — Gaussian mixture scores
        distances: [k] — L2 distances (for diagnostics)
        gaussian_ranks: [k] — rank change from L2 to Gaussian (positive = promoted)
    """
    N = mu.shape[0]
    n_fetch = min(int(k * overfetch), N)

    # Phase 1: Fast L2 retrieval via Gram-matrix trick
    q_sq = np.dot(query, query)
    if precomputed_m_sq is not None:
        m_sq = precomputed_m_sq
    else:
        m_sq = np.einsum("ij,ij->i", mu, mu)
    cross = mu @ query
    dist_sq = q_sq + m_sq - 2.0 * cross
    np.maximum(dist_sq, 0.0, out=dist_sq)

    if N > n_fetch:
        candidates = np.argpartition(dist_sq, n_fetch - 1)[:n_fetch]
    else:
        candidates = np.arange(N)

    # Phase 2: Gaussian scoring on candidates
    # Reuse pre-computed distances instead of recomputing them
    cand_dist_sq = dist_sq[candidates]
    cand_alpha = alpha[candidates]
    cand_kappa = kappa[candidates]

    scores = gaussian_score(query, mu[candidates], cand_alpha, cand_kappa,
                           precomputed_dist_sq=cand_dist_sq)

    # L2 ranks among candidates (for rank change tracking)
    l2_ranks = np.argsort(np.argsort(cand_dist_sq))

    # Gaussian ranking (descending score = best first)
    gauss_order = np.argsort(-scores)
    top_k = gauss_order[:k]

    # Compute rank changes
    gaussian_ranks = np.zeros(k, dtype=np.int32)
    for new_rank, idx in enumerate(top_k):
        gaussian_ranks[new_rank] = int(l2_ranks[idx]) - new_rank

    result_indices = candidates[top_k]
    result_scores = scores[top_k]
    result_distances = np.sqrt(cand_dist_sq[top_k])

    return result_indices, result_scores, result_distances, gaussian_ranks


def gaussian_energy(
    query: np.ndarray,
    mu: np.ndarray,
    alpha: np.ndarray,
    kappa: np.ndarray,
) -> float:
    """
    Compute energy landscape value for a query point.

    E(x) = -log(Σᵢ αᵢ · exp(-κᵢ · ‖x - μᵢ‖²))

    Lower energy = the query is well-covered by existing splats (high confidence).
    Higher energy = the query is in a sparse region (exploration needed).

    Args:
        query: [D]
        mu: [N, D]
        alpha: [N]
        kappa: [N]

    Returns:
        energy: scalar (lower = more confident)
    """
    scores = gaussian_score(query, mu, alpha, kappa)
    total_density = np.sum(scores)
    return -float(np.log(max(total_density, 1e-30)))
