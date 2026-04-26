"""
Online Update Rules for Gaussian Splats.

Implements Hebbian-inspired learning rules that adapt splat parameters
based on user feedback (relevant / not_relevant) after each query.

Three update mechanisms:
  1. Frequency boost: α increases when a splat is confirmed relevant
  2. Concentration sharpening: κ increases for hits, decreases for misses
  3. Mean drift: μ moves slightly toward queries that find it relevant

All updates are bounded and numerically stable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FeedbackEvent:
    """A single feedback event from a user."""

    query: np.ndarray
    splat_index: int
    relevant: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class UpdateStats:
    """Statistics about online updates."""

    total_feedback: int = 0
    total_alpha_up: int = 0
    total_alpha_down: int = 0
    total_kappa_up: int = 0
    total_kappa_down: int = 0
    total_mu_drifts: int = 0


class OnlineUpdater:
    """
    Online parameter updates for Gaussian Splats via user feedback.

    The update rules are:
      - Relevant hit: α += η_α, κ += η_κ, μ += η_μ * (query - μ)
      - Not relevant: α *= decay, κ -= η_κ (bounded below)
      - Temporal decay: all α decay slowly over time (forgetting)

    All hyperparameters are configurable. Updates are applied in-place
    to the numpy arrays in SplatStore.
    """

    def __init__(
        self,
        lr_alpha: float = 0.1,
        lr_kappa: float = 0.5,
        lr_mu: float = 0.01,
        alpha_max: float = 10.0,
        alpha_min: float = 0.01,
        kappa_max: float = 100.0,
        kappa_min: float = 0.5,
        temporal_decay: float = 0.999,
        decay_interval: int = 100,
    ):
        self.lr_alpha = lr_alpha
        self.lr_kappa = lr_kappa
        self.lr_mu = lr_mu
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.kappa_max = kappa_max
        self.kappa_min = kappa_min
        self.temporal_decay = temporal_decay
        self.decay_interval = decay_interval

        self._feedback_buffer: List[FeedbackEvent] = []
        self._decay_counter = 0
        self._stats = UpdateStats()

    @property
    def stats(self) -> UpdateStats:
        return self._stats

    def apply_feedback(
        self,
        event: FeedbackEvent,
        mu: np.ndarray,
        alpha: np.ndarray,
        kappa: np.ndarray,
    ) -> None:
        """
        Apply a single feedback event to update splat parameters.

        Args:
            event: The feedback event (query, splat index, relevant/not)
            mu: [max_splats, D] — splat centers (modified in-place)
            alpha: [max_splats] — splat amplitudes (modified in-place)
            kappa: [max_splats] — splat concentrations (modified in-place)
        """
        idx = event.splat_index
        if idx < 0 or idx >= len(alpha):
            return

        self._stats.total_feedback += 1

        if event.relevant:
            # Positive feedback: strengthen the splat
            alpha[idx] = min(alpha[idx] + self.lr_alpha, self.alpha_max)
            kappa[idx] = min(kappa[idx] + self.lr_kappa, self.kappa_max)

            # Drift μ slightly toward the query (Hebbian association)
            diff = event.query - mu[idx]
            mu[idx] += self.lr_mu * diff

            self._stats.total_alpha_up += 1
            self._stats.total_kappa_up += 1
            self._stats.total_mu_drifts += 1
        else:
            # Negative feedback: weaken the splat
            alpha[idx] = max(alpha[idx] * (1.0 - self.lr_alpha), self.alpha_min)
            kappa[idx] = max(kappa[idx] - self.lr_kappa * 0.5, self.kappa_min)

            self._stats.total_alpha_down += 1
            self._stats.total_kappa_down += 1

        self._feedback_buffer.append(event)
        self._decay_counter += 1

        # Apply temporal decay periodically
        if self._decay_counter >= self.decay_interval:
            self._apply_temporal_decay(alpha)
            self._decay_counter = 0

    def apply_batch_feedback(
        self,
        query: np.ndarray,
        relevant_indices: List[int],
        irrelevant_indices: List[int],
        mu: np.ndarray,
        alpha: np.ndarray,
        kappa: np.ndarray,
    ) -> None:
        """
        Apply feedback for a batch of results from a single query.

        Args:
            query: [D] query vector
            relevant_indices: indices of confirmed-relevant splats
            irrelevant_indices: indices of confirmed-irrelevant splats
            mu, alpha, kappa: splat parameter arrays (modified in-place)
        """
        now = time.time()
        for idx in relevant_indices:
            self.apply_feedback(
                FeedbackEvent(query=query, splat_index=idx, relevant=True, timestamp=now),
                mu,
                alpha,
                kappa,
            )
        for idx in irrelevant_indices:
            self.apply_feedback(
                FeedbackEvent(query=query, splat_index=idx, relevant=False, timestamp=now),
                mu,
                alpha,
                kappa,
            )

    def _apply_temporal_decay(self, alpha: np.ndarray) -> None:
        """
        Apply exponential decay to all alpha values.

        This implements a forgetting mechanism: splats that are never
        confirmed as relevant gradually fade away.
        """
        # Only decay active splats (non-zero alpha)
        mask = alpha > self.alpha_min
        alpha[mask] *= self.temporal_decay
        # Clamp to minimum
        alpha[mask] = np.maximum(alpha[mask], self.alpha_min)

    def get_feedback_summary(self) -> Dict[str, int]:
        """Return summary of feedback processed."""
        return {
            "total_feedback": self._stats.total_feedback,
            "alpha_increases": self._stats.total_alpha_up,
            "alpha_decreases": self._stats.total_alpha_down,
            "kappa_increases": self._stats.total_kappa_up,
            "kappa_decreases": self._stats.total_kappa_down,
            "mu_drifts": self._stats.total_mu_drifts,
        }
