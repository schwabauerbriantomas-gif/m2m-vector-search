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

from typing import Tuple

import numpy as np


def gaussian_score(
    query: np.ndarray,
    mu: np.ndarray,
    alpha: np.ndarray,
    kappa: np.ndarray,
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

    Returns:
        scores: [N] — Gaussian mixture scores (higher = better match)
    """
    diff = mu - query  # [N, D]
    dist_sq = np.einsum("ij,ij->i", diff, diff)  # [N]

    # Clamp kappa to prevent numerical overflow
    exponent = -kappa * dist_sq
    exponent = np.clip(exponent, -100.0, 0.0)

    return alpha * np.exp(exponent)


def gaussian_score_batch(
    queries: np.ndarray,
    mu: np.ndarray,
    alpha: np.ndarray,
    kappa: np.ndarray,
) -> np.ndarray:
    """
    Compute Gaussian mixture scores for a batch of queries.

    Args:
        queries: [B, D]
        mu: [N, D]
        alpha: [N]
        kappa: [N]

    Returns:
        scores: [B, N]
    """
    B = queries.shape[0]
    N = mu.shape[0]

    # Compute all pairwise squared distances: [B, N]
    # Using the expansion: ||q - m||² = ||q||² + ||m||² - 2·q·m
    q_sq = np.sum(queries**2, axis=1, keepdims=True)  # [B, 1]
    m_sq = np.sum(mu**2, axis=1, keepdims=True).T  # [1, N]
    cross = queries @ mu.T  # [B, N]
    dist_sq = q_sq + m_sq - 2.0 * cross  # [B, N]
    dist_sq = np.maximum(dist_sq, 0.0)  # Numerical stability

    # Gaussian scores: α · exp(-κ · d²) for each (query, splat) pair
    exponent = -kappa[np.newaxis, :] * dist_sq  # [B, N]
    exponent = np.clip(exponent, -100.0, 0.0)

    return alpha[np.newaxis, :] * np.exp(exponent)


def two_phase_search(
    query: np.ndarray,
    mu: np.ndarray,
    alpha: np.ndarray,
    kappa: np.ndarray,
    k: int = 64,
    overfetch: float = 2.0,
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

    Returns:
        indices: [k] — splat indices (sorted by Gaussian score, descending)
        scores: [k] — Gaussian mixture scores
        distances: [k] — L2 distances (for diagnostics)
        gaussian_ranks: [k] — rank change from L2 to Gaussian (positive = promoted)
    """
    N = mu.shape[0]
    n_fetch = min(int(k * overfetch), N)

    # Phase 1: Fast L2 retrieval
    diff = mu - query
    dist_sq = np.einsum("ij,ij->i", diff, diff)

    if N > n_fetch:
        candidates = np.argpartition(dist_sq, n_fetch - 1)[:n_fetch]
    else:
        candidates = np.arange(N)

    # Phase 2: Gaussian scoring on candidates
    cand_mu = mu[candidates]
    cand_alpha = alpha[candidates]
    cand_kappa = kappa[candidates]

    scores = gaussian_score(query, cand_mu, cand_alpha, cand_kappa)

    # L2 ranks among candidates (for rank change tracking)
    cand_dist_sq = dist_sq[candidates]
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
